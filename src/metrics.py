import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.optimize import linear_sum_assignment

def ari_score(y_true, y_pred):
    """计算调整兰德指数 (Adjusted Rand Index)"""
    return adjusted_rand_score(y_true, y_pred)

def nmi_score(y_true, y_pred):
    """计算归一化互信息 (Normalized Mutual Information)"""
    return normalized_mutual_info_score(y_true, y_pred)

def silhouette_score_custom(X, labels):
    """计算轮廓系数 (Silhouette Score)"""
    return silhouette_score(X, labels)

def cluster_acc(y_true, y_pred):
    """计算聚类准确率"""
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size

def calculate_metrics(X, y_true, y_pred):
    """计算所有评估指标"""
    metrics = {
        'ari': ari_score(y_true, y_pred),
        'nmi': nmi_score(y_true, y_pred),
        'silhouette': silhouette_score_custom(X, y_pred),
        'acc': cluster_acc(y_true, y_pred)
    }
    return metrics 