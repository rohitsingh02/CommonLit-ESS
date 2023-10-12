import glob
import torch
import argparse
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
from data.preprocessing import Preprocessor, get_max_len_from_df, preprocess_text
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



def average_checkpoints(input_folder, output_ckpt):
    input_ckpts = sorted(glob.glob(input_folder + '/*.pth'))
    # assert len(input_ckpts) >= 1
    if len(input_ckpts) >= 1:
        data = torch.load(input_ckpts[0], map_location='cpu')['state_dict']
        swa_n = 1
        for ckpt in input_ckpts[1:]:
            if "config" in ckpt:
                continue
            new_data = torch.load(ckpt, map_location='cpu')['state_dict']
            swa_n += 1
            for k, v in new_data.items():
                if v.dtype != torch.float32:
                    print(k)
                else:
                    data[k] += (new_data[k] - data[k]) / swa_n

        torch.save(dict(state_dict=data), output_ckpt)


def average_checkpoints_from_path(input_ckpts, output_ckpt):
    if len(input_ckpts) >= 1:
        data = torch.load(input_ckpts[0], map_location='cpu')['state_dict']
        swa_n = 1
        for ckpt in input_ckpts[1:]:
            if "config" in ckpt:
                continue
            new_data = torch.load(ckpt, map_location='cpu')['state_dict']
            swa_n += 1
            for k, v in new_data.items():
                if v.dtype != torch.float32:
                    print(k)
                else:
                    data[k] += (new_data[k] - data[k]) / swa_n
        torch.save(dict(state_dict=data), output_ckpt)







