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
from data.preprocessing import  get_input_text, split_prompt_text
from criterion.score import get_score

from utils import get_config, load_filepaths, get_logger
from utils import get_config, dictionary_to_namespace, get_logger, save_config, update_filepaths

from models.utils import get_model
from dataset.collators import collate
from utils import dictionary_to_namespace
import argparse
from utils import str_to_bool


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir_path', type=str)
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



# def inference_fn(test_loader, model, config):
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

#             if config.architecture.pooling_type == "CLS":
#                 # sample_pred = y_preds
#                 for index, sample in enumerate(inputs["input_ids"]):
#                     pred_indexes_start = [i for i, k in enumerate(sample) if k == config.text_start_token][0]
#                     pred_indexes_end = [i for i, k in enumerate(sample) if k == config.text_end_token][0]
#                     x = y_preds[index, pred_indexes_start:pred_indexes_end+1, :]
#                     y_preds[index,:,:] =  torch.div(torch.sum(x, dim=0), x.shape[0]) # (y_preds[index][pred_indexes_start, :] + y_preds[index][pred_indexes_end+1, :]) / 2
#                 y_preds = torch.mean(y_preds, dim=1)


#         preds.append(y_preds.to('cpu').numpy())
#     predictions = np.concatenate(preds)
#     return predictions


def inference_fn(test_loader, model, config):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        with torch.no_grad():
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
            preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()

    model_id = args.model_dir_path.split("/")[-1]
    fold = args.fold
    filepaths = load_filepaths()
    filepaths['log_fn_path'] = os.path.join(args.model_dir_path, 'train.log')
    logger = get_logger(filename=filepaths['log_fn_path'])
    config_path = os.path.join(filepaths['CONFIGS_DIR_PATH'], f'{model_id}_training_config.yaml')
    config = get_config(config_path)
    config['device'] = device

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_dir_path, 'tokenizer/'))

    config = dictionary_to_namespace(config)
    config.tokenizer = tokenizer
    config.architecture.pretrained = False
    
    seed_everything(seed=config.environment.seed)
    test_df = pd.read_csv("../data/raw/train_folds_processed.csv")

    config.text_start_token = tokenizer.convert_tokens_to_ids("[SUMMARY_START]")
    config.text_end_token = tokenizer.convert_tokens_to_ids("[SUMMARY_END]")

    if hasattr(config.dataset, "prompt_text_sent_end_count"):
        test_df['prompt_text'] = test_df.apply(lambda x: split_prompt_text(config, x), axis=1)


    if hasattr(config.dataset, "preprocess_cols"):
        for col in config.dataset.preprocess_cols:
            test_df[col] = test_df[col].apply(lambda x: preprocess_text(x))


    if config.architecture.pooling_type == "CLS":
        test_df['text'] = test_df.text.apply(lambda x: f"[SUMMARY_START]{x}[SUMMARY_END]")


    test_df['input_text'] = test_df.progress_apply(lambda x: get_input_text(x, config), axis=1)
    if hasattr(config.dataset, "preprocess_all") and config.dataset.preprocess_all:
        test_df['text'] = test_df['text'].apply(lambda x: preprocess_text(x, config, type="summary"))
        test_df['prompt_question'] = test_df['prompt_question'].apply(lambda x: preprocess_text(x, config, type="prompt"))
        test_df['prompt_title'] = test_df['prompt_title'].apply(lambda x: preprocess_text(x, config, type="prompt"))
        test_df['prompt_text'] = test_df['prompt_text'].apply(lambda x: preprocess_text(x, config, type="prompt"))


    
    test_df['tokenize_length'] = [len(config.tokenizer(text)['input_ids']) for text in test_df['input_text'].values]
    test_df = test_df.sort_values('tokenize_length', ascending=True).reset_index(drop=True)

    if config.dataset.set_max_length_from_data:
        config.dataset.max_length = get_max_len_from_df(test_df, config.tokenizer, config)


    target_columns = ['content', 'wording']


    subset = test_df.copy()
    subset = subset[subset['fold'] == fold].reset_index(drop=True)
    test_dataloader = get_test_dataloader(config, subset)
    backbone_type = config.architecture.model_name.replace('/', '-')
    model_checkpoint_path = os.path.join(args.model_dir_path, f"{backbone_type}_fold{fold}_best.pth")
    backbone_config_path = os.path.join(args.model_dir_path, 'config.pth')

    model = get_model(config,
                        backbone_config_path=backbone_config_path,
                        model_checkpoint_path=model_checkpoint_path,
                        train=False)

    prediction = inference_fn(test_dataloader, model, config)
    score, scores = get_score(subset[['content', 'wording']].values, prediction)

    logger.info(f'\n Final fold {fold} Score SWA - Save Best Score: {score:.4f}, {scores} Model\n')
    logger.info(f"==================== fold: {fold} training Finished ====================")

