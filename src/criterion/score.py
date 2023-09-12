import numpy as np
from sklearn.metrics import mean_squared_error


def get_score(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False)
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores



def get_score_single(y_trues, y_preds):
    scores = []
    mcrmse_score = mean_squared_error(y_trues, y_preds, squared=False)
    scores.append(mcrmse_score)
    return mcrmse_score, scores