#!/bin/bash

set -ex


# for model_id in model2
# do
#   # CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 0
#   # CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 1
#   # CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 2
#   # CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 3

#   CUDA_VISIBLE_DEVICES=1 python inference.py --model_dir_path "../models/${model_id}" --mode oofs --debug False
# done


# for model_id in model4
# do
#   # CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 0
#   # CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 1
#   # CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 2
#   # CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 3

#   CUDA_VISIBLE_DEVICES=1 python inference.py --model_dir_path "../models/${model_id}" --mode oofs --debug False
# done


# for model_id in model6
# do
#   CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 0
#   CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 1
#   CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 2
#   CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 3

#   CUDA_VISIBLE_DEVICES=1 python inference.py --model_dir_path "../models/${model_id}" --mode oofs --debug False
# done


# for model_id in model8
# do
#   CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 0
#   CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 1
#   CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 2
#   CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 3

#   CUDA_VISIBLE_DEVICES=1 python inference.py --model_dir_path "../models/${model_id}" --mode oofs --debug False
# done



for model_id in model10
do
  CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 0
  CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 1
  CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 2
  CUDA_VISIBLE_DEVICES=1 python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 3

  CUDA_VISIBLE_DEVICES=1 python inference.py --model_dir_path "../models/${model_id}" --mode oofs --debug False
done