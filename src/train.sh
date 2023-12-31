#!/bin/bash

set -ex


for model_id in model236
do
  CUDA_VISIBLE_DEVICES=0 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 0
  CUDA_VISIBLE_DEVICES=0 python inference_swa.py --model_dir_path "../models/${model_id}" --fold 0
  CUDA_VISIBLE_DEVICES=0 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 1
  CUDA_VISIBLE_DEVICES=0 python inference_swa.py --model_dir_path "../models/${model_id}" --fold 1
  CUDA_VISIBLE_DEVICES=0 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 2
  CUDA_VISIBLE_DEVICES=0 python inference_swa.py --model_dir_path "../models/${model_id}" --fold 2
  CUDA_VISIBLE_DEVICES=0 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 3
  CUDA_VISIBLE_DEVICES=0 python inference_swa.py --model_dir_path "../models/${model_id}" --fold 3
  CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir_path "../models/${model_id}" --mode oofs --debug False
  CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir_path "../models/${model_id}" --mode oofs --debug False --avg_oof True

done