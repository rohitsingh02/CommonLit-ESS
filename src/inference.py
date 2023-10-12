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
from data.preprocessing import get_input_text, split_prompt_text, process_prompt_text
from joblib import Parallel, delayed

from utils import get_config, load_filepaths
from models.utils import get_model
from dataset.collators import collate
from utils import dictionary_to_namespace
import argparse
from utils import str_to_bool


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir_path', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--debug', type=str_to_bool, default=False)
    parser.add_argument('--avg_oof', type=str_to_bool, default=False)
    arguments = parser.parse_args()
    return arguments


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True




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
                # y_preds = masked_y_preds.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()

                y_preds_text = masked_y_preds.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
                if hasattr(config.dataset, "prompt_pooling"):
                    start_tokens = (input_ids == config.prompt_start_token).nonzero()[:, 1]
                    end_tokens = (input_ids == config.prompt_end_token).nonzero()[:, 1]
                    mask = torch.arange(input_ids.size(1)).expand(input_ids.size(0), -1).to(input_ids.device)
                    mask = (mask >= start_tokens.unsqueeze(1)) & (mask <= end_tokens.unsqueeze(1))
                    masked_y_preds = y_preds * mask.unsqueeze(2).float()
                    y_preds_prompt = masked_y_preds.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
                    y_preds_text = 0.8*y_preds_text + 0.2*y_preds_prompt

                y_preds = y_preds_text
            preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions



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
    

    config.text_start_token = tokenizer.convert_tokens_to_ids("[SUMMARY_START]")
    config.text_end_token = tokenizer.convert_tokens_to_ids("[SUMMARY_END]")
    if hasattr(config.dataset, "prompt_pooling"):
        config.prompt_start_token = tokenizer.convert_tokens_to_ids("[PROMPT_START]")
        config.prompt_end_token = tokenizer.convert_tokens_to_ids("[PROMPT_END]")


    assert args.mode in ['oofs']

    seed_everything(seed=config.environment.seed)

    test_df = pd.read_csv("../data/raw/train_folds_processed.csv")
    if args.debug:
        test_df = test_df.sample(50, random_state=42)

    if hasattr(config.dataset, "prompt_text_seq"):
        test_df['prompt_text'] = test_df.apply(lambda x: process_prompt_text(x, config, type=config.dataset.prompt_text_seq), axis=1)

    if hasattr(config.dataset, "prompt_text_sent_count") and config.dataset.prompt_text_sent_count > 1:
        test_df['prompt_text'] = test_df.apply(lambda x: split_prompt_text(config, x), axis=1)

    if hasattr(config.dataset, "preprocess_cols"):
        for col in config.dataset.preprocess_cols:
            test_df[col] = test_df[col].apply(lambda x: preprocess_text(x))


    test_df['text'] = test_df.text.apply(lambda x: f"[SUMMARY_START]{x}[SUMMARY_END]")
    if hasattr(config.dataset, "prompt_pooling"):
        test_df['prompt_text'] = test_df.prompt_text.apply(lambda x: f"[PROMPT_START]{x}[PROMPT_END]")


    test_df['input_text'] = test_df.progress_apply(lambda x: get_input_text(x, config), axis=1)
    
    test_df['tokenize_length'] = [len(config.tokenizer(text)['input_ids']) for text in test_df['input_text'].values]
    test_df = test_df.sort_values('tokenize_length', ascending=True).reset_index(drop=True)

    if config.dataset.set_max_length_from_data:
        config.dataset.max_length = get_max_len_from_df(test_df, config.tokenizer, config)

    if args.debug:
        test_df = test_df.sample(50, random_state=1)

    target_columns = ['content', 'wording']


    predictions = []
    oofs_list = []
    for fold in range(config.dataset.n_folds):
        subset = test_df.copy()
        if args.mode == 'oofs':
            subset = subset[subset['fold'] == fold].reset_index(drop=True)

        # fold_samples = prepare_data(config, subset ,num_jobs=config.environment.n_workers, datatype="test")
        test_dataloader = get_test_dataloader(config, subset)

        backbone_type = config.architecture.model_name.replace('/', '-')
        model_checkpoint_path = os.path.join(args.model_dir_path, f"{backbone_type}_fold{fold}_best_avg.pth")
        if args.avg_oof and os.path.exists(model_checkpoint_path):
            model_checkpoint_path = os.path.join(args.model_dir_path, f"{backbone_type}_fold{fold}_best_avg.pth")
        else:
            model_checkpoint_path = os.path.join(args.model_dir_path, f"{backbone_type}_fold{fold}_best.pth")


        backbone_config_path = os.path.join(args.model_dir_path, 'config.pth')

        model = get_model(config,
                          backbone_config_path=backbone_config_path,
                          model_checkpoint_path=model_checkpoint_path,
                          train=False)

        prediction = inference_fn(test_dataloader, model, config)
        predictions.append(prediction)

        gc.collect()
        torch.cuda.empty_cache()

        if args.mode in ['oofs']:
            out = pd.DataFrame(prediction, columns=['content', 'wording'])
            for col in ['student_id', 'prompt_id', 'text', 'tokenize_length', 'fold']:
                if col in subset.columns:
                    out[col] = subset[col]

            if args.mode == 'oofs':
                oofs_list.append(out)
        del model, prediction


    oof_df = pd.concat(oofs_list)
    if args.avg_oof:
        oof_df.to_csv(os.path.join(filepaths['OOFS_DIR_PATH'], f'{model_id}_avg.csv'), index=False)
    else:
        oof_df.to_csv(os.path.join(filepaths['OOFS_DIR_PATH'], f'{model_id}.csv'), index=False)