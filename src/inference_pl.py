import os
import gc
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm.auto import tqdm

import torch

from transformers import AutoTokenizer
from dataset.datasets import get_test_dataloader
from data.preprocessing import Preprocessor, get_max_len_from_df, make_folds, preprocess_text

from utils import get_config, load_filepaths
from models.utils import get_model
from dataset.collators import collate
from utils import dictionary_to_namespace

import argparse
from utils import str_to_bool
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from transformers import AutoConfig, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir_path', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--debug', type=str_to_bool, default=False)
    arguments = parser.parse_args()
    return arguments


# def seed_everything(seed=42):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True







class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


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
        self.pooler = MeanPooling()
#         for parameter in self.transformers_model.encoder.layer[:8].parameters():
#           parameter.requires_grad = False
#         for parameter in self.transformers_model.embeddings.parameters():
#           parameter.requires_grad = False 

        self.backbone.gradient_checkpointing_enable()
        
        if hasattr(self.cfg.training, "multi_dropout") and self.cfg.training.multi_dropout:
            self.dropout = nn.Dropout(self.backbone_config.hidden_dropout_prob)
            self.dropout1 = nn.Dropout(0.1)
            self.dropout2 = nn.Dropout(0.2)
            self.dropout3 = nn.Dropout(0.3)
            self.dropout4 = nn.Dropout(0.4)
            self.dropout5 = nn.Dropout(0.5)
        
        
          
        self.fc =  nn.Linear(self.backbone.config.hidden_size,len(self.cfg.dataset.target_cols))
        if 'bart' in self.cfg.architecture.model_name:
            self.initializer_range = self.backbone_config.init_std
        else:
            self.initializer_range = self.backbone_config.initializer_range

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
        last_layer_hidden_states = self.backbone(input_ids,attention_mask = attention_mask)[0]
        feature = self.pooler(last_layer_hidden_states, attention_mask)
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
    
       
    def train_dataloader(self):
        return self._train_dataloader 
    
    def validation_dataloader(self):
        return self._validation_dataloader


    # def configure_optimizers(self):
        
    #     optimizer = get_optimizer(model=self, config=self.cfg)
    #     # optimizer = AdamW(self.parameters(), lr = self.cfg.optimizer.encoder_lr)
    #     epoch_steps = self.cfg.data_length
    #     batch_size = self.cfg.training.train_batch_size
    #     warmup_steps = self.cfg.optimizer.warmup_ratio * epoch_steps // batch_size
    #     training_steps = self.cfg.training.epochs * epoch_steps // batch_size
    #     # scheduler = get_linear_schedule_with_warmup(optimizer,warmup_steps,training_steps,-1)
    #     scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, training_steps, lr_end=7e-7, power=3.0)

    #     lr_scheduler_config = {
    #             'scheduler': scheduler,
    #             'interval': 'step',
    #             'frequency': 1,
    #         }

    #     return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}


def inference_fn(data_loader, model):
    model.to(config.device)
    model.eval()    
    preds = []
    for batch in tqdm(data_loader):
        with torch.no_grad():
            inputs = {key:val.reshape(val.shape[0], -1).to(config.device) for key,val in batch.items()}
            outputs = model(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'],train=False)
        preds.extend(outputs.detach().cpu().numpy())
    predictions = np.vstack(preds)
    return predictions



# def inference_fn(test_loader, model, device):
#     preds = []
#     model.eval()
#     model.to(device)
#     tk0 = tqdm(test_loader, total=len(test_loader))
#     for inputs in tk0:
#         inputs = collate(inputs)
#         for k, v in inputs.items():
#             inputs[k] = v.to(device)
#         with torch.no_grad():
#             y_preds = model(inputs)
#         preds.append(y_preds.to('cpu').numpy())
#     predictions = np.concatenate(preds)
#     return predictions




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()

    model_id = args.model_dir_path.split("/")[-1]
    filepaths = load_filepaths()

    config_path = os.path.join(filepaths['CONFIGS_DIR_PATH'], f'{model_id}_training_config.yaml')
    config = get_config(config_path)
    config['device'] = device

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_dir_path, 'tokenizer/'))

    config = dictionary_to_namespace(config)
    config.tokenizer = tokenizer
    config.architecture.pretrained = False
    

    assert args.mode in ['prev_pseudolabels', 'curr_pseudolabels', 'submission', 'oofs']

    seed_everything(seed=config.environment.seed)

    dataframe_path = None
    if args.mode == 'submission':
        dataframe_path = filepaths['TEST_CSV_PATH']
    elif args.mode == 'oofs':
        dataframe_path = filepaths['TRAIN_CSV_PATH']
    elif args.mode == 'prev_pseudolabels':
        dataframe_path = filepaths['PREVIOUS_DATA_CSV_PATH']
    elif args.mode == 'curr_pseudolabels':
        dataframe_path = filepaths['TRAIN_CSV_PATH']

    test_df = pd.read_csv(dataframe_path)

    if args.mode == 'oofs':        
        # test_df = make_folds(test_df,
        #     target_cols=config.general.target_columns,
        #     n_splits=config.general.n_folds,
        # )
        test_df = pd.read_csv(filepaths['TRAIN_FOLDS_CSV_PATH'])
        

    elif args.mode == 'prev_pseudolabels':
        df = pd.read_csv(filepaths['TRAIN_CSV_PATH'])
        test_df['in_fb3'] = test_df['text_id'].apply(lambda x: x in df.text_id.values)
        test_df = test_df[~test_df['in_fb3'].values]

    if args.debug:
        test_df = test_df.sample(50, random_state=42)


    ### new code to test model_performance  
    # if not os.path.exists("../data/raw/train_folds_processed.csv"):
    #     train_prompt = pd.read_csv(filepaths['TRAIN_PROMPT_CSV_PATH'])
    #     preprocessor = Preprocessor(model_name=config.architecture.model_name)
    #     train = preprocessor.run(train_prompt, train, mode="train")
    #     train.to_csv("../data/raw/train_folds_processed.csv", index=False)
    # else:
    test_df = pd.read_csv("../data/raw/train_folds_processed.csv")
    test_df['text'] = test_df['fixed_summary_text']
    # test_df['text'] = test_df['text'].apply(preprocess_text)
    
    

    test_df['tokenize_length'] = [len(config.tokenizer(text)['input_ids']) for text in test_df['text'].values]
    test_df = test_df.sort_values('tokenize_length', ascending=True).reset_index(drop=True)

    if config.dataset.set_max_length_from_data:
        config.dataset.max_length = get_max_len_from_df(test_df, config.tokenizer)

    if args.debug:
        test_df = test_df.sample(50, random_state=1)

    target_columns = ['content', 'wording']

    predictions = []
    oofs_list = []
    for fold in range(config.dataset.n_folds):

        subset = test_df.copy()
        if args.mode == 'oofs':
            subset = subset[subset['fold'] == fold].reset_index(drop=True)

        test_dataloader = get_test_dataloader(config, subset)

        backbone_type = config.architecture.model_name.replace('/', '-')
        model_checkpoint_path = os.path.join(args.model_dir_path, f"{backbone_type}_fold{fold}_best.pth")
        backbone_config_path = os.path.join(args.model_dir_path, 'config.pth')


        model = CommonlitModel.load_from_checkpoint(
            f'{model_checkpoint_path}.ckpt',
            train_dataloader=None,
            validation_dataloader=None,
            config=config
        )    
        # preds = predict(test_dataloader, model)   



        # model = get_model(config,
        #                   backbone_config_path=backbone_config_path,
        #                   model_checkpoint_path=model_checkpoint_path,
        #                   train=False)

        prediction = inference_fn(test_dataloader, model)
        predictions.append(prediction)

        gc.collect()
        torch.cuda.empty_cache()

        if args.mode in ['prev_pseudolabels', 'curr_pseudolabels', 'oofs']:
            out = pd.DataFrame(prediction, columns=['content', 'wording'])
            for col in ['student_id', 'prompt_id', 'text', 'tokenize_length', 'fold']:
                if col in subset.columns:
                    out[col] = subset[col]

            if args.mode in ['prev_pseudolabels', 'curr_pseudolabels']:
                pseudo_path = filepaths['PREVIOUS_DATA_PSEUDOLABELS_DIR_PATH'] \
                    if args.mode == 'prev_pseudolabels' \
                    else filepaths['CURRENT_DATA_PSEUDOLABELS_DIR_PATH']

                dir_path = os.path.join(pseudo_path, f'{model_id}_pseudolabels')
                if not os.path.isdir(dir_path):
                    os.mkdir(dir_path)

                out.to_csv(os.path.join(dir_path, f'pseudolabels_fold{fold}.csv'), index=False)

            if args.mode == 'oofs':
                oofs_list.append(out)

        del model, prediction

    if args.mode == 'submission':
        predictions = np.mean(predictions, axis=0)

        test_df[target_columns] = predictions

        submission = pd.read_csv(filepaths['SAMPLE_SUBMISSION_CSV_PATH'])
        submission = submission.drop(columns=target_columns) \
            .merge(test_df[['text_id'] + target_columns], on='text_id', how='left')
        submission[['text_id'] + target_columns].to_csv(os.path.join(filepaths['SUBMISSIONS_DIR_PATH'], f'{model_id}_submission.csv'), index=False)

    elif args.mode == 'oofs':
        oof_df = pd.concat(oofs_list)
        oof_df.to_csv(os.path.join(filepaths['OOFS_DIR_PATH'], f'{model_id}.csv'), index=False)