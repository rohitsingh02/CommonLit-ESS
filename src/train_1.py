# =========================================================================================
# Libraries
# =========================================================================================
import os
import sys 
import wandb
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import yaml
import utils
import time
import shutil
import pickle
import argparse
import importlib
import numpy as np
import pandas as pd
from types import SimpleNamespace
import torch
import random
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AutoTokenizer
)
import copy
import multiprocessing

from tez import Tez, TezConfig
from tez import enums
from tez.callbacks import Callback
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.multiprocessing
import re
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
import utils
torch.multiprocessing.set_sharing_strategy("file_system")
import warnings
warnings.filterwarnings("ignore")

sys.path.append('../src')
sys.path.append("model_factory")
sys.path.append("datasets")

            
            
def get_result(oof_df):
    labels = oof_df[utils.target_cols].values
    preds = oof_df[[f"pred_{c}" for c in utils.target_cols]].values
    score, scores = utils.get_score(labels, preds)
    cfg.logger.info(f'Score: {score:<.4f}  Scores: {scores}')

    

class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.0001, model_pth=""):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.model_pth = model_pth
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, cfg, epoch_score, model, predictions):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(cfg, epoch_score, model, predictions)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(cfg, epoch_score, model, predictions)
            self.counter = 0

    def save_checkpoint(self, cfg, epoch_score, model, predictions):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Saving model!".format(
                    self.val_score, epoch_score
                )
            )
            # if cfg.epoch >= cfg.training.swa_start:
            #     model_state_dict = model.state_dict() 
            #     del model_state_dict['n_averaged']
            #     model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
            # else:
            # model_state_dict = model.state_dict()
            # ## calculate and save statistics
            # torch.save({'model': model_state_dict}, self.model_pth) 
            
            torch.save( {'model': model.state_dict(), 'predictions': predictions}  , self.model_pth)
        self.val_score = epoch_score



def validate_epoch(cfg, model, test_df, valid_samples, data_loader):    
    
    losses = utils.AverageMeter()
    model.eval()
    bar=tqdm(enumerate(data_loader),total=len(data_loader))
    preds = []
    valid_labels = []
    for step, data_dict in bar:
        ids = data_dict['ids'].to(cfg.device, dtype = torch.long)
        mask = data_dict['mask'].to(cfg.device, dtype = torch.long)
        labels = data_dict['targets'].to(cfg.device, dtype = torch.long)
        
        batch_size = data_dict['targets'].size(0)
        
        with torch.no_grad():        
            logits, loss, rmse = model(ids, mask, targets=labels)
                
        losses.update(loss.item(), batch_size)   
        preds.append(logits.detach().cpu().numpy())  
        valid_labels.append(labels.detach().cpu().numpy())       
        bar.set_postfix(loss=losses.avg)

    predictions = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)
    
    score, scores = utils.get_score(valid_labels, predictions)
    # print("Score", score)
    cfg.logger.info(f"Fold: {cfg.fold}, epoch: {cfg.epoch}")  
    cfg.logger.info(f"mcrmse_score: {score}, scores: {scores}")  
    return score, scores, predictions



def save_targets(cfg, df_valid):
    predictions = torch.load(os.path.join(output_pth, f"model_{cfg.fold}.pth"), 
                        map_location=torch.device('cpu'))['predictions']
    df_valid[[f"pred_{c}" for c in utils.target_cols]] = predictions  
    return df_valid



def train_loop(cfg, df_valid, train_samples, valid_samples):
    
    train_dataset = CommonlitDataset(train_samples, cfg.dataset.max_len, tokenizer, mode="train")
    valid_dataset = CommonlitDataset(valid_samples, cfg.dataset.max_len, tokenizer, mode="val")
    num_train_steps = int(len(train_dataset) / cfg.training.batch_size / cfg.training.grad_accumulation * cfg.training.epochs)
        
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.environment.mixed_precision)

    # TRAIN DATASET AND VALID DATASET
    train_params = {
        'batch_size': cfg.training.batch_size,
        'shuffle': True,
        'num_workers': multiprocessing.cpu_count(),
        'pin_memory':True,
        'drop_last': True,
    }
    
    val_params = {
        'batch_size': cfg.training.batch_size,
        'shuffle': False,
        'num_workers': multiprocessing.cpu_count(),
        'pin_memory':True,
        'drop_last': False
    }

    train_loader = DataLoader(train_dataset, **train_params)
    val_loader = DataLoader(valid_dataset, **val_params)

    model = CommonlitModel(
        cfg=cfg,
        model_name=cfg.architecture.model_name,
        num_train_steps=num_train_steps,
        learning_rate=cfg.training.lr,
        num_labels=len(utils.target_cols),
        steps_per_epoch=len(train_dataset) / cfg.training.batch_size,
        tokenizer=tokenizer,
    )
    model.to(cfg.device)
    
    if cfg.architecture.pretrained_weights != "":
        if hasattr(cfg, "swa_model") and cfg.swa_model:
            model.load_state_dict(torch.load(cfg.architecture.pretrained_weights, map_location='cpu')['model'])
        else:
            model.load_state_dict(torch.load(cfg.architecture.pretrained_weights, map_location='cpu'))
        print(f"Weights loaded from path: {cfg.architecture.pretrained_weights}")
        
        
    # exit()
    swa_model = AveragedModel(model, device=cfg.device, use_buffers=True)
    swa_lr = 1.0e-7
    swa_start = cfg.training.swa_start #int(cfg.training.epochs / 2)
    optimizer, scheduler = model.optimizer_scheduler()
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)

    es = EarlyStopping(patience=cfg.es.patience, mode="min", model_pth=os.path.join(output_pth, f"model_{cfg.fold}.pth"),)
    
    for epoch in range(cfg.training.epochs):
        cfg.epoch = epoch
        start_time = time.time()
        model.train()
        losses = utils.AverageMeter()
        start = end = time.time()
        global_step = 0
        bar=tqdm(enumerate(train_loader),total=len(train_loader))
        for step, data_dict in bar:
            ids = data_dict['ids'].to(cfg.device, dtype = torch.long)
            mask = data_dict['mask'].to(cfg.device, dtype = torch.long)
            labels = data_dict['targets'].to(cfg.device, dtype = torch.float)
            batch_size = data_dict['targets'].size(0)
            
            # # use cutmix but don't change labels
            # if np.random.uniform()<0.25:
            #     cut=0.25
            #     perm=torch.randperm(ids.shape[0]).cuda()
            #     rand_len=int(ids.shape[1]*cut)
            #     start=np.random.randint(ids.shape[1]-int(ids.shape[1]*cut))
            #     ids[:,start:start+rand_len]=ids[perm,start:start+rand_len]
            #     mask[:,start:start+rand_len]=mask[perm,start:start+rand_len]
            #     # labels[:,start:start+rand_len]=labels[perm,start:start+rand_len]
                
                
            
            with torch.cuda.amp.autocast(enabled=cfg.environment.mixed_precision):
                logits, loss, rmse = model(ids, mask, targets=labels)
                
            if cfg.training.grad_accumulation > 1:
                loss = loss / cfg.training.grad_accumulation
                
            losses.update(loss.item(), train_loader.batch_size)   
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            if (step + 1) % cfg.training.grad_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                if cfg.training.batch_scheduler:
                    scheduler.step()
                    
            end = time.time()  
            
            if  step % cfg.training.print_freq == 0 or step == (len(train_loader)-1):
                print('Epoch: [{0}][{1}/{2}] '
                    'Elapsed {remain:s} '
                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                    'Grad: {grad_norm:.4f}  '
                    'LR: {lr:.8f}  '
                    .format(epoch+1, step, len(train_loader), 
                            remain=utils.timeSince(start, float(step+1)/len(train_loader)),
                            loss=losses,
                            grad_norm=grad_norm,
                            lr=scheduler.get_lr()[0]))
                
                if cfg.epoch >= 2:
                    val_score, val_scores, predictions = validate_epoch(cfg, model=model, test_df=df_valid, valid_samples=valid_samples, data_loader=val_loader)
                    cfg.logger.info(f"mid epoch score: {val_score}, scores: {val_scores}")  
                    es(cfg, val_score, model, predictions)
            bar.set_postfix({'train_loss': losses.avg})    
            
        
        # if epoch >= swa_start:
        #     swa_model.update_parameters(model)
        #     swa_scheduler.step()
        #     model_to_eval = swa_model
        # else:
        #     model_to_eval = model
        model_to_eval = model
            
        val_score, val_scores, predictions = validate_epoch(cfg, model=model_to_eval, test_df=df_valid, valid_samples=valid_samples, data_loader=val_loader)
        scheduler.step()
        
        cfg.logger.info(f"Fold {cfg.fold}, Epoch {epoch}, MCMRSE: {val_score}, Score: {val_scores}")  
        model_to_eval.train()
        ## saving checkpoint if it increased CV
        es(cfg, val_score, model_to_eval, predictions) 
        if es.early_stop:            
            df_valid = save_targets(cfg, df_valid)
            print('Early Stopping')
            break   
    df_valid = save_targets(cfg, df_valid)
    return df_valid
       



# setting up config
parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)
cfg = yaml.safe_load(open(parser_args.config).read())
for k, v in cfg.items():
    if type(v) == dict:
        cfg[k] = SimpleNamespace(**v)
cfg = SimpleNamespace(**cfg)

os.makedirs(f"{cfg.output_dir}/{cfg.experiment_name}", exist_ok=True)
shutil.copy(parser_args.config, f"{cfg.output_dir}/{cfg.experiment_name}")


CommonlitDataset = importlib.import_module(cfg.dataset_class).CommonlitDataset
CommonlitModel = importlib.import_module(cfg.model_class).CommonlitModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.device = device
output_pth = f"{cfg.output_dir}/{cfg.experiment_name}"


if __name__ == "__main__":
    
    LOGGER = utils.get_logger(cfg)
    cfg.logger = LOGGER
    
    if cfg.wandb.enable: utils.init_wandb(cfg)
    cfg.environment.seed=np.random.randint(1_000_000) if cfg.environment.seed < 0 else np.random.randint(1_000_000)
    utils.set_seed(cfg.environment.seed)
    
    
    df = pd.read_csv(f"{cfg.dataset.base_dir}/summaries_train.csv")
    # df['text'] = df['text'].apply(lambda x: utils.replace_newline(x)) 
    # create folds
    df = utils.create_folds(df, cfg=cfg)  
    # print(df.columns)
    
    # exit()
      
    tokenizer = AutoTokenizer.from_pretrained(cfg.architecture.model_name)
    # if hasattr(cfg.dataset, "replace_newline") and cfg.dataset.replace_newline:
    #     special_tokens_dict = {'additional_special_tokens': ["[BR]"]}
    #     num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    
    # tokenizer.save_pretrained(output_pth+'/tokenizer/')
    cfg.tokenizer = tokenizer   

    oof_df = pd.DataFrame()

    if cfg.training.fold != -1:
        folds = [cfg.training.fold]
    else:
        folds = [0,1,2,3]

    for fold in folds: # [0,1,2,3]
        cfg.fold = fold
        df_train = df.loc[df.fold != fold].reset_index(drop=True)
        df_valid = df.loc[df.fold == fold].reset_index(drop=True)
        
        if cfg.debug:
            df_train = df_train.head(50).reset_index(drop=True)
            df_valid = df_valid.head(100).reset_index(drop=True)
        
        print(df_train.shape, df_valid.shape)
        train_samples = utils.prepare_data(df_train, tokenizer, num_jobs=cfg.environment.num_jobs, datatype="train")
        valid_samples = utils.prepare_data(df_valid, tokenizer, num_jobs=cfg.environment.num_jobs, datatype="train")
        print(len(train_samples), len(valid_samples))
        _oof_df = train_loop(cfg, df_valid, train_samples, valid_samples)
        oof_df = pd.concat([oof_df, _oof_df])
        cfg.logger.info(f"========== fold: {cfg.fold} result ==========")
        get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        cfg.logger.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(output_pth+'/oof_df.pkl')

        

        
        
        
        

            
    
    # print(len(training_samples), len(valid_samples))
    # print(len(df_train['id'].unique()), len(df_test['id'].unique()), )
    # # start training
    # train_loop(cfg, df_test, valid_samples)