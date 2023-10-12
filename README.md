# CommonLit - Evaluate Student Summaries - 49th Place Solution

It's 49th place solution to Kaggle competition: https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries

This repo contains the code i used to train the models, while the solution writeup is available here: https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446516

## HARDWARE: (The following specs were used to create the original solution)
* OS: Ubuntu 20.04.4 LTS
* CPU: Intel Xeon Gold 5315Y @3.2 GHz, 8 cores
* RAM: 44Gi
* GPU: 1 x NVIDIA RTX A6000 (49140MiB)

## SOFTWARE (python packages are detailed separately in requirements.txt):
* Python 3.9.13
* CUDA 11.6
* nvidia drivers v510.73.05

## Training 
All the data required for training is present at `data/raw` directory.

To train models:
* move to src directory and run: ./train.sh.


## Inference
Final inference kernel is available here: https://www.kaggle.com/code/rohitsingh9990/commonlit-ensemble-new-v2?scriptVersionId=145818122
