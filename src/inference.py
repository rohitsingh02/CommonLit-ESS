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
from data.preprocessing import get_input_text, split_prompt_text
from joblib import Parallel, delayed

from utils import get_config, load_filepaths
from models.utils import get_model
from dataset.collators import collate
from utils import dictionary_to_namespace
import argparse
from utils import str_to_bool
import time






os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir_path', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--debug', type=str_to_bool, default=False)
    arguments = parser.parse_args()
    return arguments


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def _prepare_data_helper(config, df, student_ids):
    samples = []
    for idx in tqdm(student_ids):
        temp_df = df[df["student_id"] == idx].reset_index(drop=True)
        text = temp_df.input_text.values[0]
        encoded_text = config.tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=config.dataset.max_length,
            padding="max_length",
            truncation=True,
        )
        # print(encoded_text.keys())
        samples.append(encoded_text) 

        # input_ids = encoded_text["input_ids"]
        # attention_mask = 
        # input_labels = copy.deepcopy(input_ids)
        # input_labels2 = copy.deepcopy(input_ids)

        # offset_mapping = encoded_text["offset_mapping"]
        
        # for k in range(len(input_labels)):
        #     input_labels[k] = "O"
        #     input_labels2[k] = "O"
            
        # sample = {
        #     "id": idx, # essay_id
        #     "input_ids": input_ids, # input_ids
        #     "text": text, # essay_text
        #     "offset_mapping": offset_mapping, # offset_mapping
        # }
        
        # for _, row in temp_df.iterrows():
        #     text_labels = [0] * len(text)
        #     text_labels2 = [0] * len(text)
        #     discourse_start = int(row["discourse_start"])
        #     discourse_end = int(row["discourse_end"])
        #     prediction_label = row["discourse_type"]
        #     discourse_effectiveness = row["discourse_effectiveness"]
            
        #     text_labels[discourse_start:discourse_end] = [1] * (discourse_end - discourse_start)
        #     text_labels2[discourse_start:discourse_end] = [1] * (discourse_end - discourse_start)
            
    
        #     target_idx = []
        #     for map_idx, (offset1, offset2) in enumerate(encoded_text["offset_mapping"]):
        #         if sum(text_labels[offset1:offset2]) > 0:
        #             if len(text[offset1:offset2].split()) > 0:
        #                 target_idx.append(map_idx)
        
        #     targets_start = target_idx[0]
        #     targets_end = target_idx[-1]
        #     pred_start = f"B-" + prediction_label
        #     pred_end = f"I-" + prediction_label
        #     input_labels[targets_start] = pred_start
        #     input_labels[targets_start + 1 : targets_end + 1] = [pred_end] * (targets_end - targets_start)
            
            
        #     input_labels2[targets_start] = discourse_effectiveness
        #     input_labels2[targets_start + 1 : targets_end + 1] = [discourse_effectiveness] * (targets_end - targets_start)

            
        # sample["input_ids"] = input_ids
        # sample["input_labels"] = input_labels
        # sample["input_labels2"] = input_labels2
        # samples.append(sample)
    return samples



def prepare_data(config, df, num_jobs, datatype):
    samples = []
    student_ids = df["student_id"].unique()
    student_ids_splits = np.array_split(student_ids, num_jobs)
    results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(_prepare_data_helper)(config, df, idx) for idx in student_ids_splits
    )
    for result in results:
        samples.extend(result)

    return samples


# def inference_fn(test_loader, model, config):
#     preds = []
#     model.eval()
#     model.to(device)
#     tk0 = tqdm(test_loader, total=len(test_loader))
#     for inputs in tk0:
#         inputs = collate(inputs)
#         # end = time.time()
#         # print(f"collate time {end-start}")
#         for k, v in inputs.items():
#             inputs[k] = v.to(device)

#         with torch.no_grad():
#             y_preds = model(inputs)
#             input_ids = inputs['input_ids']
#             # start = time.time()

#             if config.architecture.pooling_type == "CLS":
#                 # Find the positions of start and end tokens
#                 start_tokens = (input_ids == config.text_start_token).nonzero()[:, 1]
#                 end_tokens = (input_ids == config.text_end_token).nonzero()[:, 1]

#                 # print(start_tokens, end_tokens)
#                 # Calculate the mask for valid positions
#                 mask = torch.arange(input_ids.size(1)).expand(input_ids.size(0), -1).to(input_ids.device)
#                 mask = (mask >= start_tokens.unsqueeze(1)) & (mask <= end_tokens.unsqueeze(1))
#                 # Mask y_preds and calculate the mean along the time dimension
#                 masked_y_preds = y_preds * mask.unsqueeze(2).float()
#                 y_preds = masked_y_preds.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
#                 # end = time.time()
#                 # print(f"Input processing time: {end - start} seconds")
#                 # Print the resulting mean_y_preds tensor
#                 # print(y_preds)



#             # if config.architecture.pooling_type == "CLS":
#             #     for index, sample in enumerate(inputs["input_ids"]):
                    
#             #         pred_indexes_start = torch.where(sample == config.text_start_token)[0].item()
#             #         pred_indexes_end = torch.where(sample == config.text_end_token)[0].item()
#             #         print(pred_indexes_start, pred_indexes_end)
#             #     #     # pred_indexes_start = [i for i, k in enumerate(sample) if k == config.text_start_token][0]
#             #     #     # pred_indexes_end = [i for i, k in enumerate(sample) if k == config.text_end_token][0]
#             #     #     x = y_preds[index, pred_indexes_start:pred_indexes_end+1, :]
#             #     #     y_preds[index,:,:] =  torch.div(torch.sum(x, dim=0), x.shape[0])
#             #     # y_preds = torch.mean(y_preds, dim=1)
#             # end = time.time()
#             # print(f"input processing time {end-start}")
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
    train_prompt = pd.read_csv(filepaths['TRAIN_PROMPT_CSV_PATH'])
    test_df = test_df.merge(
        train_prompt, 
        on='prompt_id'
    ).reset_index(drop=True)
    
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

    if args.debug:
        test_df = test_df.sample(50, random_state=1)

    target_columns = ['content', 'wording']


    predictions = []
    oofs_list = []
    for fold in range(config.dataset.n_folds):

        # if fold > 1:
        #     break

        subset = test_df.copy()
        if args.mode == 'oofs':
            subset = subset[subset['fold'] == fold].reset_index(drop=True)

        # fold_samples = prepare_data(config, subset ,num_jobs=config.environment.n_workers, datatype="train")
        test_dataloader = get_test_dataloader(config, subset)
        # test_dataloader = get_test_dataloader(config, fold_samples, subset)

        backbone_type = config.architecture.model_name.replace('/', '-')
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


    oof_df = pd.concat(oofs_list)
    oof_df.to_csv(os.path.join(filepaths['OOFS_DIR_PATH'], f'{model_id}.csv'), index=False)