# =========================================================================================
# Libraries
# =========================================================================================
import os
import sys
import gc
import utils
import wandb
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
from dataset.datasets import get_train_dataloader, get_valid_dataloader
from dataset.collators import collate
from models.utils import get_model, freeze
from optimizer.optimizer import get_optimizer
from scheduler.scheduler import get_scheduler
from adversarial_learning.awp import AWP
from criterion.criterion import get_criterion

from models.pooling_layers import get_pooling_layer


import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import (
    AutoConfig, 
)
from transformers import AutoModel, AutoTokenizer, AdamW, DataCollatorWithPadding
from transformers import get_polynomial_decay_schedule_with_warmup,get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup


from datetime import datetime
torch.multiprocessing.set_sharing_strategy("file_system")




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str)
    parser.add_argument('--run_id', type=str)
    parser.add_argument('--debug', type=str_to_bool, default=False)
    parser.add_argument('--use_wandb', type=str_to_bool, default=True)
    parser.add_argument('--fold', type=int)
    arguments = parser.parse_args()
    return arguments



def check_arguments():
    all_folds = [i for i in range(config.dataset.n_folds)]
    assert args.fold in all_folds, \
        f'Invalid training fold, fold number must be in {all_folds}'

    if config.dataset.use_current_data_pseudo_labels and config.dataset.use_current_data_true_labels:
        logger.warning('Both use_current_data_pseudo_labels and use_current_data_true_labels are True. ')



# class MeanPooling(nn.Module):
#     def __init__(self):
#         super(MeanPooling, self).__init__()
        
#     def forward(self, last_hidden_state, attention_mask):
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
#         sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
#         sum_mask = input_mask_expanded.sum(1)
#         sum_mask = torch.clamp(sum_mask, min=1e-9)
#         mean_embeddings = sum_embeddings / sum_mask
#         return mean_embeddings


# class MeanPooling(nn.Module):
#     def __init__(self, backbone_config, pooling_config):
#         super(MeanPooling, self).__init__()
#         self.output_dim = backbone_config.hidden_size

#     def forward(self, inputs, backbone_outputs):
#         attention_mask = get_attention_mask(inputs)
#         last_hidden_state = get_last_hidden_state(backbone_outputs)

#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
#         sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
#         sum_mask = input_mask_expanded.sum(1)
#         sum_mask = torch.clamp(sum_mask, min=1e-9)
#         mean_embeddings = sum_embeddings / sum_mask
#         return mean_embeddings



class CommonlitModel(pl.LightningModule):
    def __init__(self,config):
        super().__init__()
        
        self.cfg = config
        
        self.backbone_config = AutoConfig.from_pretrained(
            config.architecture.model_name,
        )
        self.backbone_config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": 0.,
                "add_pooling_layer": False,
                "attention_probs_dropout_prob":0,
            }
        )
        
        self.backbone = AutoModel.from_pretrained(config.architecture.model_name,config=self.backbone_config)
        self.pooler = get_pooling_layer(config=self.cfg, backbone_config=self.backbone_config )

        # self.backbone.gradient_checkpointing_enable()
        
        if hasattr(self.cfg.training, "multi_dropout") and self.cfg.training.multi_dropout:
            self.dropout = nn.Dropout(self.backbone_config.hidden_dropout_prob)
            self.dropout1 = nn.Dropout(0.1)
            self.dropout2 = nn.Dropout(0.2)
            self.dropout3 = nn.Dropout(0.3)
            self.dropout4 = nn.Dropout(0.4)
            self.dropout5 = nn.Dropout(0.5)
        
        self.fc =  nn.Linear(self.pooler.output_dim, len(self.cfg.dataset.target_cols))
        if 'bart' in self.cfg.architecture.model_name:
            self.initializer_range = self.backbone_config.init_std
        else:
            self.initializer_range = self.backbone_config.initializer_range


        if config.architecture.gradient_checkpointing:
            if self.backbone.supports_gradient_checkpointing:
                self.backbone.gradient_checkpointing_enable()
            else:
                print(f'{config.model.backbone_type} does not support gradient checkpointing')


        if config.architecture.freeze_embeddings:
            freeze(model.backbone.embeddings)
        if config.architecture.freeze_n_layers > 0:
            freeze(model.backbone.encoder.layer[:config.model.freeze_n_layers])
        if config.architecture.reinitialize_n_layers > 0:
            for module in self.backbone.encoder.layer[-config.architecture.reinitialize_n_layers:]:
                self._init_weights(module)


        self._init_weights(self.fc)
        self.loss_function = nn.SmoothL1Loss(reduction='mean') 
        self.validation_step_outputs = []
        

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        

    def forward(self, input_ids, attention_mask, train):
        backbone_outputs = self.backbone(input_ids, attention_mask = attention_mask)
        feature = self.pooler({'input_ids': input_ids, 'attention_mask': attention_mask},  backbone_outputs)
        # x = self.classifier(last_layer_hidden_states)
        
        if hasattr(self.cfg.training, "multi_dropout") and self.cfg.training.multi_dropout:
            feature = self.dropout(feature)
            logits1 = self.fc(self.dropout1(feature))
            logits2 = self.fc(self.dropout2(feature))
            logits3 = self.fc(self.dropout3(feature))
            logits4 = self.fc(self.dropout4(feature))
            logits5 = self.fc(self.dropout5(feature))
            output = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        else:
            output = self.fc(feature)
        
        return output
    

    def training_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target = batch['labels'] 
        
        
        if self.cfg.training.cutmix_ratio > 0:
            if np.random.uniform()<0.4:
                cut=self.cfg.training.cutmix_ratio
                perm=torch.randperm(input_ids.shape[0]).cuda()
                rand_len=int(input_ids.shape[1]*cut)
                start=np.random.randint(input_ids.shape[1]-int(input_ids.shape[1]*cut))
                input_ids[:,start:start+rand_len]=input_ids[perm,start:start+rand_len]
                attention_mask[:,start:start+rand_len]=attention_mask[perm,start:start+rand_len]
                target[:] = (target[:] + target[perm]) / 2
                    # print(labels)
        
        
        output = self(input_ids,attention_mask,train=True)        
        loss = self.loss_function(output,target)
        self.log('train_loss', loss , prog_bar=True)
        return {'loss': loss}
    
    def train_epoch_end(self,outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print(f'epoch {self.trainer.current_epoch} training loss {avg_loss}')
        return {'train_loss': avg_loss} 
    
    
    def validation_step(self,batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target = batch['labels'] 
        output = self(input_ids,attention_mask,train=False)
        loss = self.loss_function(output, target)
        self.log('val_loss', loss , prog_bar=True)
        val_dict = {'val_loss': loss, 'logits': output,'targets':target} 
        self.validation_step_outputs.append(val_dict)
        return val_dict      

    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        output_val = torch.cat([x['logits'] for x in outputs],dim=0).cpu().detach().numpy()
        target_val = torch.cat([x['targets'] for x in outputs],dim=0).cpu().detach().numpy()
        # print(output_val.shape)
        mcrmse_score, scores = get_score(target_val, output_val) # competition_metrics(output_val, target_val)
        print(f'epoch {self.trainer.current_epoch} validation loss {avg_loss}')
        print(f'epoch {self.trainer.current_epoch} validation score {mcrmse_score}, {scores}')
        
        self.validation_step_outputs.clear()
        self.log("val_mcrmse", mcrmse_score, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': avg_loss, 'val_mcrmse':mcrmse_score, 'content': scores[0], 'wording': scores[1]}
    
        
    def train_dataloader(self):
        return self._train_dataloader 
    
    def validation_dataloader(self):
        return self._validation_dataloader

    # def get_optimizer_params(self, encoder_lr, decoder_lr, weight_decay=0.0):
    #     param_optimizer = list(model.named_parameters())
    #     no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    #     optimizer_parameters = [
    #         {'params': [p for n, p in self.transformers_model.named_parameters() if not any(nd in n for nd in no_decay)],
    #          'lr': encoder_lr, 'weight_decay': weight_decay},
    #         {'params': [p for n, p in self.transformers_model.named_parameters() if any(nd in n for nd in no_decay)],
    #          'lr': encoder_lr, 'weight_decay': 0.0},
    #         {'params': [p for n, p in self.named_parameters() if "transformers_model" not in n],
    #          'lr': decoder_lr, 'weight_decay': 0.0}
    #     ]
    #     return optimizer_parameters

    def configure_optimizers(self):
        
        optimizer = get_optimizer(model=self, config=self.cfg)
        # optimizer = AdamW(self.parameters(), lr = self.cfg.optimizer.encoder_lr)
        epoch_steps = self.cfg.data_length
        batch_size = self.cfg.training.train_batch_size
        warmup_steps = self.cfg.optimizer.warmup_ratio * epoch_steps // batch_size
        training_steps = self.cfg.training.epochs * epoch_steps // batch_size
        # scheduler = get_linear_schedule_with_warmup(optimizer,warmup_steps,training_steps,-1)
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, training_steps, lr_end=7e-7, power=3.0)

        lr_scheduler_config = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
    
      
    
        


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
    
    # seed_everything(seed=config.general.seed)
    seed_everything(config.environment.seed)
    check_arguments()

    train = pd.read_csv(filepaths['TRAIN_FOLDS_CSV_PATH'])
    train_prompt = pd.read_csv(filepaths['TRAIN_PROMPT_CSV_PATH'])
    
    ### new code to test model_performance  
    if not os.path.exists("../data/raw/train_folds_processed.csv"):
        preprocessor = Preprocessor(model_name=config.architecture.model_name)
        train = preprocessor.run(train_prompt, train, mode="train")
        train.to_csv("../data/raw/train_folds_processed.csv", index=False)
    else:
        train = pd.read_csv("../data/raw/train_folds_processed.csv")
    
    
    train['text'] = train['fixed_summary_text']
    special_tokens_replacement = get_additional_special_tokens()
    all_special_tokens = list(special_tokens_replacement.values())
    # print(all_special_tokens)
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.architecture.model_name,
        use_fast=True,
        additional_special_tokens=all_special_tokens
    )
    tokenizer.save_pretrained(filepaths['tokenizer_dir_path'])
    config.tokenizer = tokenizer
    
    train_df = pd.DataFrame(columns=train.columns)
    valid_df = train[train['fold'] == fold].reset_index(drop=True)
    if config.dataset.use_current_data_true_labels:
        train_df = pd.concat([train_df, train[train['fold'] != fold].reset_index(drop=True)], axis=0)
    
    if args.debug:
        logger.info('Debug mode: using only 50 samples')
        train_df = train_df.sample(n=50, random_state=config.general.seed).reset_index(drop=True)
        valid_df = valid_df.sample(n=50, random_state=config.general.seed).reset_index(drop=True)
        
    logger.info(f'Train shape: {train_df.shape}')
    logger.info(f'Valid shape: {valid_df.shape}')

    if config.dataset.set_max_length_from_data:
        logger.info('Setting max length from data')
        config.dataset.max_length = get_max_len_from_df(train_df, tokenizer)
        
    logger.info(f"Max tokenized sequence len: {config.dataset.max_length}")
    logger.info(f"==================== fold: {fold} training ====================")

    model_checkpoint_path = filepaths['model_checkpoint_fn_path'] if config.architecture.from_checkpoint else None
    logger.info(f'Using model checkpoint from: {model_checkpoint_path}')
    
    if args.debug:
        config.training.epochs = 1

    swa_callback = pl.callbacks.StochasticWeightAveraging(
        swa_epoch_start=config.training.swa.swa_epoch_start, swa_lrs= config.training.swa.swa_lrs,#0.05, 
        annealing_epochs=config.training.swa.annealing_epochs, annealing_strategy= config.training.swa.annealing_strategy, #'cos', 
        avg_fn=None, device="cuda"
    )
    
    
    train_dataloader = get_train_dataloader(config, train_df)
    valid_dataloader = get_valid_dataloader(config, valid_df)
    valid_labels = valid_df[config.dataset.target_cols].values
    
    
    config.data_length = len(train_df)
    # model = get_model(config, model_checkpoint_path=model_checkpoint_path)
    # torch.save(model.backbone_config, filepaths['backbone_config_fn_path'])
    # model.to(device)
    
    # optimizer = get_optimizer(model, config)
    # train_steps_per_epoch = int(len(train_df) / config.general.train_batch_size)
    # num_train_steps = train_steps_per_epoch * config.training.epochs
    # eval_steps = get_evaluation_steps(train_steps_per_epoch,
    #                                   config.training.evaluate_n_times_per_epoch)
    # scheduler = get_scheduler(optimizer, config, num_train_steps)    
    
    early_stop_callback = EarlyStopping(monitor="val_mcrmse", min_delta=config.es.min_delta, patience=config.es.patience, verbose= True, mode=config.es.mode)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mcrmse',
        dirpath=filepaths['run_dir_path'],
        save_top_k=1,
        save_last= False,
        save_weights_only=True,
        filename= f"{filepaths['model_fn_path'].split('/')[-1]}",
        verbose= True,
        mode='min'
    )
    
    print("Model Creation")
    model = CommonlitModel(config)
    trainer = Trainer(
        max_epochs= config.training.epochs,
        val_check_interval=config.training.val_check_interval,
        accumulate_grad_batches=config.training.gradient_accumulation_steps,
        devices=config.training.gpu_count,
        precision=config.training.precision,
        accelerator="gpu",
        callbacks=[swa_callback, checkpoint_callback, early_stop_callback],
        default_root_dir=filepaths['run_dir_path']
    )    
    # print("Trainer Starting")
    trainer.fit(model, train_dataloader, valid_dataloader)  
    print("prediction on validation data")

    del model,train_dataloader,valid_dataloader
    gc.collect()
    torch.cuda.empty_cache()  