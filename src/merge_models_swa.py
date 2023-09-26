# =========================================================================================
# Libraries
# =========================================================================================
import os
import sys
import gc
import utils
import wandb
import shutil
import argparse
import datetime
import time
import argparse
import numpy as np
import pandas as pd
import torch
import random
from tqdm import tqdm
from transformers import (
    AutoTokenizer
)
import copy
import multiprocessing
from torch_ema import ExponentialMovingAverage


os.environ["TOKENIZERS_PARALLELISM"] = "true"
sys.path.append('../src')

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.multiprocessing
import re
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

from utils import load_filepaths
from utils import get_config, dictionary_to_namespace, get_logger, save_config, update_filepaths
from utils import str_to_bool, create_dirs_if_not_exists
from utils import AverageMeter, time_since, get_evaluation_steps
from utils import time_since
from criterion.score import get_score, get_score_single
from data.preprocessing import Preprocessor, make_folds, get_max_len_from_df, get_additional_special_tokens, preprocess_text, add_prompt_info
from data.preprocessing import get_input_text, split_prompt_text

from dataset.datasets import get_train_dataloader, get_valid_dataloader
from dataset.collators import collate
from models.utils import get_model
from optimizer.optimizer import get_optimizer
from scheduler.scheduler import get_scheduler
from adversarial_learning.awp import AWP
from criterion.criterion import get_criterion
from datetime import datetime
from models.swa import average_checkpoints
torch.multiprocessing.set_sharing_strategy("file_system")

tqdm.pandas()


fold_score_dict_swa = {
    0: 0.5,
    1: 0.6,
    2: 0.5,
    3: 0.6
}



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str)
    parser.add_argument('--run_id', type=str)
    parser.add_argument('--debug', type=str_to_bool, default=False)
    parser.add_argument('--use_wandb', type=str_to_bool, default=True)
    parser.add_argument('--fold', type=int)
    arguments = parser.parse_args()
    return arguments

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def check_arguments():
    all_folds = [i for i in range(config.dataset.n_folds)]
    assert args.fold in all_folds, \
        f'Invalid training fold, fold number must be in {all_folds}'

    if config.dataset.use_current_data_pseudo_labels and config.dataset.use_current_data_true_labels:
        logger.warning('Both use_current_data_pseudo_labels and use_current_data_true_labels are True. ')

    


def valid_fn(valid_loader, model, criterion, epoch, ema):    
    
    valid_losses = utils.AverageMeter()
    model.eval()
    predictions = []
    start = time.time()
    
    bar=tqdm(enumerate(valid_loader),total=len(valid_loader))
    
    valid_labels = []
    for step, (inputs, labels) in bar:
        inputs = collate(inputs)
        
        for k, v in inputs.items():
            inputs[k] = v.to(device)
            
        labels = labels.to(device)
        batch_size = labels.size(0)
        if config.dataset.input_cols[0] == "text":
            pred_indexes_start = 1

        with torch.no_grad(): 
            if hasattr(config.training, "ema"):
                with ema.average_parameters():
                    y_preds = model(inputs)
            else:
                y_preds = model(inputs)

            input_ids = inputs['input_ids']
            if config.architecture.pooling_type == "CLS":
                # Find the positions of start and end tokens
                start_tokens = (input_ids == config.text_start_token).nonzero()[:, 1]
                end_tokens = (input_ids == config.text_end_token).nonzero()[:, 1]
                mask = torch.arange(input_ids.size(1)).expand(input_ids.size(0), -1).to(input_ids.device)
                mask = (mask >= start_tokens.unsqueeze(1)) & (mask <= end_tokens.unsqueeze(1))
                masked_y_preds = y_preds * mask.unsqueeze(2).float()
                y_preds = masked_y_preds.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
            # preds.append(y_preds.to('cpu').numpy())
            # if config.architecture.pooling_type == "CLS":
            #     sample_pred = y_preds
            #     for index, sample in enumerate(inputs["input_ids"]):
            #         # pred_indexes_start = [i for i, k in enumerate(sample) if k == config.text_start_token][0]
            #         pred_indexes_end = [i for i, k in enumerate(sample) if k == config.text_end_token][0]
            #         x = y_preds[index, pred_indexes_start:pred_indexes_end+1, :]
            #         y_preds[index,:,:] =  torch.div(torch.sum(x, dim=0), x.shape[0]) # (y_preds[index][pred_indexes_start, :] + y_preds[index][pred_indexes_end+1, :]) / 2
                
            #     y_preds = torch.mean(y_preds, dim=1)


            loss = criterion(y_preds, labels)

        if config.training.gradient_accumulation_steps > 1:
            loss = loss / config.training.gradient_accumulation_steps
               
        valid_losses.update(loss.item(), batch_size)   

        # if config.architecture.pooling_type == "CLS":
        #     predictions.append(y_preds[:, 1].detach().cpu().numpy())  
        # else:
        predictions.append(y_preds.detach().cpu().numpy())  
        bar.set_postfix(loss=valid_losses.avg)
        
        if step % config.training.valid_print_frequency == 0 or step == (len(valid_loader) - 1):
            remain = time_since(start, float(step + 1) / len(valid_loader))
            logger.info('EVAL: [{0}][{1}/{2}] '
                        'Elapsed: {remain:s} '
                        'Loss: {loss.avg:.4f} '
                        .format(epoch+1, step+1, len(valid_loader),
                                remain=remain,
                                loss=valid_losses))
            
        if args.use_wandb: 
            wandb.log({f"Validation loss": valid_losses.val})

    predictions = np.concatenate(predictions)
    return valid_losses, predictions



def train_loop(train_folds, valid_folds, model_checkpoint_path=None):
    
    train_dataloader = get_train_dataloader(config, train_folds)
    valid_dataloader = get_valid_dataloader(config, valid_folds)
    
    valid_labels = valid_folds[config.dataset.target_cols].values
    
    model = get_model(config, model_checkpoint_path=model_checkpoint_path)
    torch.save(model.backbone_config, filepaths['backbone_config_fn_path'])
    model.to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    
    optimizer = get_optimizer(model, config)
    train_steps_per_epoch = int(len(train_folds) / config.training.train_batch_size)
    # num_train_steps = train_steps_per_epoch * config.training.epochs

    eval_steps = get_evaluation_steps(train_steps_per_epoch,
                                      config.training.val_check_interval)
    num_train_steps = int(len(train_folds))
    scheduler = get_scheduler(optimizer, config, num_train_steps)
    
    awp = AWP(model=model,
            optimizer=optimizer,
            adv_lr=config.adversarial_learning.adversarial_lr,
            adv_eps=config.adversarial_learning.adversarial_eps,
            adv_epoch=config.adversarial_learning.adversarial_epoch_start)
    
    criterion = get_criterion(config)

    best_score = np.inf
    for epoch in range(config.training.epochs):
        start_time = time.time()
        model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=config.training.apex)

        train_losses = utils.AverageMeter()
        valid_losses = utils.AverageMeter() #None
        score, scores = None, None
        start = time.time()
        global_step = 0
        bar=tqdm(enumerate(train_dataloader),total=len(train_dataloader))
        for step, (inputs, labels) in bar:
            inputs = collate(inputs)
            
            for k, v in inputs.items():
                inputs[k] = v.to(device)
                
            labels = labels.to(device)
            awp.perturb(epoch)
            batch_size = labels.size(0)    
            
            # # use cutmix but don't change labels
            if hasattr(config, "cutmix") and config.training.cut_ratio > 0:
                if np.random.uniform()<0.25:
                    cut=config.training.cut_ratio
                    perm=torch.randperm(inputs["input_ids"].shape[0]).cuda()
                    rand_len=int(inputs["input_ids"].shape[1]*cut)
                    start=np.random.randint(inputs["input_ids"].shape[1]-int(inputs["input_ids"].shape[1]*cut))
                    inputs["input_ids"][:,start:start+rand_len]=inputs["input_ids"][perm,start:start+rand_len]
                    inputs["attention_mask"][:,start:start+rand_len]=inputs["attention_mask"][perm,start:start+rand_len]
                    labels[:] = (labels[:] + labels[perm]) / 2
                    # print(labels)
                    
                    # labels[:,start:start+rand_len]=labels[perm,start:start+rand_len]
                                
            with torch.cuda.amp.autocast(enabled=config.training.apex):
                y_preds = model(inputs)
                input_ids = inputs['input_ids']
                if config.architecture.pooling_type == "CLS":
                    # Find the positions of start and end tokens
                    start_tokens = (input_ids == config.text_start_token).nonzero()[:, 1]
                    end_tokens = (input_ids == config.text_end_token).nonzero()[:, 1]
                    mask = torch.arange(input_ids.size(1)).expand(input_ids.size(0), -1).to(input_ids.device)
                    mask = (mask >= start_tokens.unsqueeze(1)) & (mask <= end_tokens.unsqueeze(1))
                    masked_y_preds = y_preds * mask.unsqueeze(2).float()
                    y_preds = masked_y_preds.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
                loss = criterion(y_preds, labels)

                # if config.architecture.pooling_type == "CLS":
                #     sample_pred = y_preds
                #     for index, sample in enumerate(inputs["input_ids"]):
                #         pred_indexes_start = [i for i, k in enumerate(sample) if k == config.text_start_token][0]
                #         pred_indexes_end = [i for i, k in enumerate(sample) if k == config.text_end_token][0]
                #         x = y_preds[index, pred_indexes_start:pred_indexes_end+1, :]
                #         sample_pred[index,:,:] =  torch.div(torch.sum(x, dim=0), x.shape[0]) # (y_preds[index][pred_indexes_start, :] + y_preds[index][pred_indexes_end+1, :]) / 2
                    
                #     sample_pred = torch.mean(sample_pred, dim=1)
                #     loss = criterion(sample_pred, labels)
                # else:
                #     loss = criterion(y_preds, labels)

            if config.training.gradient_accumulation_steps > 1:
                loss = loss / config.training.gradient_accumulation_steps
                
            train_losses.update(loss.item(), batch_size)   
            scaler.scale(loss).backward()
            awp.restore()
            
            if config.training.unscale:
                scaler.unscale_(optimizer)
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            
            if args.use_wandb:
                wandb.log({f"Training loss": train_losses.val})
            
            if (step + 1) % config.training.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                if config.scheduler.batch_scheduler:
                    scheduler.step()
                    
                if hasattr(config.training, "ema"):
                    ema.update()
                    
            end = time.time()  
            
            if step % config.training.train_print_frequency == 0 or \
                (step == (len(train_dataloader)-1)) or \
                (step + 1 in eval_steps) or \
                (step - 1 in eval_steps): 
                    
                remain = time_since(start, float(step + 1) / len(train_dataloader))
                logger.info(f'Epoch: [{epoch+1}][{step+1}/{len(train_dataloader)}] '
                            f'Elapsed {remain:s} '
                            f'Loss: {train_losses.val:.4f}({train_losses.avg:.4f}) '
                            f'Grad: {grad_norm:.4f}  '
                            f'LR: {scheduler.get_lr()[0]:.8f}  ')
                
                
                if (step + 1) in eval_steps:
                    print(eval_steps)
                    valid_losses, predictions = valid_fn(valid_dataloader, model, criterion, epoch, ema)
                    
                    if len(config.dataset.target_cols) > 1:
                        score, scores = get_score(valid_labels, predictions)
                    else:
                        score, scores = get_score_single(valid_labels, predictions)
                                                    
                    model.train()
                    logger.info(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {scores}')
                    if score < best_score:
                        best_score = score
                        if score <= fold_score_dict_swa[fold]: 
                            model_save_pth = filepaths['run_dir_path'] + f"/fold{fold}/" + f"{filepaths['model_fn_path'].split('/')[-1]}"
                            torch.save({'state_dict': model.state_dict()}, model_save_pth)
                            logger.info(f'\nEpoch {epoch + 1} - Save Best Score: {best_score:.4f} Model\n')

                        torch.save({'state_dict': model.state_dict()}, filepaths['model_fn_path'])
                    
                    if config.training.use_swa:
                        if score <= fold_score_dict_swa[fold]: 
                            swa_pth = filepaths['run_dir_path'] + f"/fold{fold}" + f"/{filepaths['model_fn_path'].split('/')[-1].split('.pth')[0]}_{epoch}_{step}.pth"
                            torch.save(
                                {'state_dict': model.state_dict()},
                                f"{swa_pth}"
                            )

                    unique_parameters = ['.'.join(name.split('.')[:4]) for name, _ in model.named_parameters()]
                    learning_rates = list(set(zip(unique_parameters, scheduler.get_lr())))
                    
                    if args.use_wandb:
                        wandb.log({f'{parameter} lr': lr for parameter, lr in learning_rates})
                        wandb.log({f'Best Score': best_score})
            
            
            if config.optimizer.use_swa:
                optimizer.swap_swa_sgd()
                
            elapsed = time.time() - start_time
            if args.use_wandb:
                wandb.log({f"Epoch": epoch + 1,
                        f"avg_train_loss": train_losses.avg,
                        f"avg_val_loss": valid_losses.avg,
                        f"Score": score,
                        f"content rmse": scores[0],
                        f"wording rmse": scores[1]})  
                

            bar.set_postfix({'train_loss': train_losses.avg}) 
        
        # if epoch >= 6:
        #     break
        
        valid_losses, predictions = valid_fn(valid_dataloader, model, criterion, epoch, ema)
        
        if len(config.dataset.target_cols) > 1:
            score, scores = get_score(valid_labels, predictions)
        else:
            score, scores = get_score_single(valid_labels, predictions)
        
        model.train()
        logger.info(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {scores}')
        if score < best_score:
            best_score = score
            if score <= fold_score_dict_swa[fold]:         
                model_save_pth = filepaths['run_dir_path'] + f"/fold{fold}/" + f"{filepaths['model_fn_path'].split('/')[-1]}"
                torch.save({'state_dict': model.state_dict()}, model_save_pth)
                logger.info(f'\nEpoch {epoch + 1} - Save Best Score: {best_score:.4f} Model\n')
                

        elapsed = time.time() - start_time
        logger.info(f'Epoch {epoch + 1} - avg_train_loss: {train_losses.avg:.4f} '
                f'avg_val_loss: {valid_losses.avg:.4f} time: {elapsed:.0f}s '
                f'Epoch {epoch + 1} - Score: {score:.4f}  Scores: {scores}\n'
                '=============================================================================\n') 
        
        
    fold_path = filepaths['run_dir_path'] + f"/fold{fold}"
    average_checkpoints(fold_path, filepaths['model_fn_path'])
    # logger.info(f'\n Final fold {fold} Score SWA - Save Best Score: {best_score:.4f} Model\n')
    shutil.rmtree(fold_path)

    torch.cuda.empty_cache()
    gc.collect()
        
    return valid_folds
       


def main():

    fold_path = filepaths['run_dir_path'] + f"/fold{fold}"
    average_checkpoints(fold_path, filepaths['model_fn_path'])

    # train = pd.read_csv(filepaths['TRAIN_FOLDS_CSV_PATH'])
    # train_prompt = pd.read_csv(filepaths['TRAIN_PROMPT_CSV_PATH'])
    

    # train = make_folds(train,
    #         target_cols=config.dataset.target_cols,
    #         n_splits=config.dataset.n_folds,
    # )
    # # train.to_csv("train_folds.csv", index=False)
    
    # special_tokens_replacement = get_additional_special_tokens()
    # all_special_tokens = list(special_tokens_replacement.values())
    
    # tokenizer = AutoTokenizer.from_pretrained(
    #     config.architecture.model_name,
    #     use_fast=True,
    #     additional_special_tokens=all_special_tokens
    # )
    # tokenizer.save_pretrained(filepaths['tokenizer_dir_path'])
    # config.tokenizer = tokenizer
    
    # config.text_start_token = tokenizer.convert_tokens_to_ids("[SUMMARY_START]")
    # config.text_end_token = tokenizer.convert_tokens_to_ids("[SUMMARY_END]")

    # print(config.text_start_token, config.text_end_token)



    # ### new code to test model_performance  
    # if not os.path.exists("../data/raw/train_folds_processed.csv"):
    #     preprocessor = Preprocessor(tokenizer)
    #     train = preprocessor.run(train_prompt, train, mode="train")
    #     train.to_csv("../data/raw/train_folds_processed.csv", index=False)
    # else:
    #     train = pd.read_csv("../data/raw/train_folds_processed.csv")
    


    # if hasattr(config.dataset, "prompt_text_sent_end_count"):
    #     train_prompt['prompt_text'] = train_prompt.apply(lambda x: split_prompt_text(config, x), axis=1)



    # if hasattr(config.dataset, "preprocess_cols"):
    #     for col in config.dataset.preprocess_cols:
    #         train[col] = train[col].apply(lambda x: preprocess_text(x))


    # if config.architecture.pooling_type == "CLS":
    #     train['text'] = train.text.apply(lambda x: f"[SUMMARY_START]{x}[SUMMARY_END]")
    # # input_cols =  get_input_cols(config=config)

    # train['input_text'] = train.progress_apply(lambda x: get_input_text(x, config), axis=1)


    # train_df = pd.DataFrame(columns=train.columns)
    # valid_df = train[train['fold'] == fold].reset_index(drop=True)
    # if config.dataset.use_current_data_true_labels:
    #     train_df = pd.concat([train_df, train[train['fold'] != fold].reset_index(drop=True)], axis=0)
    
    # if args.debug:
    #     logger.info('Debug mode: using only 50 samples')
    #     train_df = train_df.sample(n=50, random_state=config.general.seed).reset_index(drop=True)
    #     valid_df = valid_df.sample(n=50, random_state=config.general.seed).reset_index(drop=True)
        
    # logger.info(f'Train shape: {train_df.shape}')
    # logger.info(f'Valid shape: {valid_df.shape}')

    # if config.dataset.set_max_length_from_data:
    #     logger.info('Setting max length from data')
    #     config.dataset.max_length = get_max_len_from_df(train_df, tokenizer, config)
        
    # logger.info(f"Max tokenized sequence len: {config.dataset.max_length}")
    # logger.info(f"==================== fold: {fold} training ====================")

    # model_checkpoint_path = filepaths['model_checkpoint_fn_path'] if config.architecture.from_checkpoint else None
    # logger.info(f'Using model checkpoint from: {model_checkpoint_path}')
    
    # if args.debug:
    #     config.training.epochs = 1
        
    # fold_out = train_loop(train_df, valid_df, model_checkpoint_path=model_checkpoint_path)
        

if __name__ == "__main__":
    args = parse_args()
    filepaths = load_filepaths()
    config_path = os.path.join(filepaths['CONFIGS_DIR_PATH'], args.config_name)
    config = get_config(config_path)
    fold = args.fold
    
    # if args.use_wandb: 
    #     run = init_wandb()
        
    filepaths = update_filepaths(filepaths, config, args.run_id, fold)  
    create_dirs_if_not_exists(filepaths)
    if not os.path.exists(filepaths['run_dir_path']):
        os.makedirs(filepaths['run_dir_path'])

    os.makedirs(filepaths['run_dir_path'] + f"/fold{fold}", exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = get_logger(filename=filepaths['log_fn_path'])
    print(filepaths.keys())
  
    if os.path.isfile(filepaths['model_fn_path']):
        new_name = filepaths["model_fn_path"]+f'_renamed_at_{str(datetime.now())}'
        logger.warning(f'{filepaths["model_fn_path"]} is already exists, renaming this file to {new_name}')
        os.rename(filepaths["model_fn_path"], new_name)
   
    print(filepaths["model_fn_path"])
    config = dictionary_to_namespace(config)
    logger.info({"config": config})
    
    seed_everything(seed=config.environment.seed)
    check_arguments()
        
    main()