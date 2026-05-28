import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, average_precision_score
from scipy.stats import pearsonr


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def pearson_corr(y_true, y_pred):
    if np.std(y_true) == 0:
        return 0.0
    return pearsonr(y_true, y_pred)[0]


# Concordance Index（CI）
def ci(y_true, y_pred):
    n = 0
    h_sum = 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] != y_true[j]:
                n += 1
                if (y_pred[i] > y_pred[j] and y_true[i] > y_true[j]) or \
                   (y_pred[i] < y_pred[j] and y_true[i] < y_true[j]):
                    h_sum += 1
                elif y_pred[i] == y_pred[j]:
                    h_sum += 0.5
    return h_sum / n if n > 0 else 0.0


# AUPR（用于二分类）
def aupr(y_true, y_pred):
    return average_precision_score(y_true, y_pred)
def get_metric(metric_name):
    metric_dict = {
        'mse': mse,
        'r2': r2,
        'pearson': pearson_corr,
        'ci': ci,
        'aupr': aupr
    }
    return metric_dict[metric_name]
