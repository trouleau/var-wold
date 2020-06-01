import numpy as np
import scipy


def edge_error(adj_excit_test, adj_excit_true, delta_t, n_nodes):
    diff_excit = (adj_excit_test - adj_excit_true).abs()
    return diff_excit.sum() * delta_t / n_nodes**2


def AUC(adj_test, adj_true):
    threshold_range = np.linspace(adj_test.max() + 0.1, 0.0, num=50)
    tps = np.array([utils.metrics.true_positive(adj_test, adj_true, threshold=x) for x in threshold_range])
    tns = np.array([utils.metrics.true_negative(adj_test, adj_true, threshold=x) for x in threshold_range])
    fns = np.array([utils.metrics.false_negative(adj_test, adj_true, threshold=x) for x in threshold_range])
    fps = np.array([utils.metrics.false_positive(adj_test, adj_true, threshold=x) for x in threshold_range])
    tpr = tps/(tps+fns)
    fpr = fps/(tns+fps)
    AUC_val = sum(np.diff(fpr) * tpr[1:])
    return AUC_val


def accuracy(adj_test, adj_true, threshold=0.05):
    assert (len(adj_test.shape) == 1) and (len(adj_true.shape) == 1), \
     "Parameters should be one-dimensional"
    shape = len(adj_test)
    n_err = np.sum((adj_test >= threshold) ^ (adj_true > 0))
    return (shape - n_err) / shape


def mean_kendall_rank_corr(adj_est, adj_true, return_vec=False):
    assert (len(adj_est.shape) == 1) and (len(adj_true.shape) == 1), \
         "Parameters should be one-dimensional"
    dim = int(np.sqrt(adj_est.shape[0]))
    adj_est = np.reshape(adj_est, (dim, dim))
    adj_true = np.reshape(adj_true, (dim, dim))
    arr = np.array([scipy.stats.kendalltau(adj_est[:, i], adj_true[:, i]).correlation
                    for i in range(dim)])
    val = np.mean(arr)
    if return_vec:
        return val, arr
    else:
        return val


def precision_at_n(adj_test, adj_true, n):
    assert (len(adj_test.shape) == 1) and (len(adj_true.shape) == 1), "Expect 1-dimensional arrays"
    sorted_args = np.argsort(adj_test)[::-1]
    return np.sum(adj_true[sorted_args][:n] > 0) / n


def precision_at_n_per_dim(A_pred, A_true, k=10):
    res = 0.0
    tmp = 0
    for i in range(A_true.shape[0]):
        x = A_true[i]
        x = x[x != 0]

        y = A_pred[i]
        y = y[y != 0]

        kx = min(len(x), k)
        ky = min(len(y), k)
        if ky == 0 or kx == 0:
            continue
        x = set(np.argpartition(x, -kx)[-kx:])
        y = set(np.argpartition(y, -ky)[-ky:])
        res += len(x.intersection(y)) / k
        tmp += 1
    if tmp == 0:
        return 0
    return res / tmp


def true_positive(adj_test, adj_true, threshold=0.05):
    assert (len(adj_test.shape) == 1) and (len(adj_true.shape) == 1), \
     "Parameters should be one-dimensional"
    return np.sum((adj_test > threshold) * (adj_true > 0))


def false_positive(adj_test, adj_true, threshold=0.05):
    assert (len(adj_test.shape) == 1) and (len(adj_true.shape) == 1), \
     "Parameters should be one-dimensional"
    return np.sum((adj_test > threshold) * (adj_true == 0))


def false_negative(adj_test, adj_true, threshold=0.05):
    assert (len(adj_test.shape) == 1) and (len(adj_true.shape) == 1), \
     "Parameters should be one-dimensional"
    return np.sum((adj_test <= threshold) * (adj_true > 0))


def true_negative(adj_test, adj_true, threshold=0.05):
    assert (len(adj_test.shape) == 1) and (len(adj_true.shape) == 1), \
     "Parameters should be one-dimensional"
    return np.sum((adj_test <= threshold) * (adj_true == 0))


def tp(adj_test, adj_true, threshold=0.05):
    return true_positive(adj_test, adj_true, threshold)


def fp(adj_test, adj_true, threshold=0.05):
    return false_positive(adj_test, adj_true, threshold)


def tn(adj_test, adj_true, threshold=0.05):
    return true_negative(adj_test, adj_true, threshold)


def fn(adj_test, adj_true, threshold=0.05):
    return false_negative(adj_test, adj_true, threshold)


def recall(adj_test, adj_true, threshold=0.05):
    tp_val = tp(adj_test, adj_true, threshold)
    if tp_val == 0:
        return tp_val
    else:
        tot = np.sum(adj_true > 0)
        return tp_val / tot


def precision(adj_test, adj_true, threshold=0.05):
    tp_val = tp(adj_test, adj_true, threshold)
    if tp_val == 0:
        return tp_val
    else:
        tot = np.sum(adj_test > threshold)
        return tp_val / tot


def tpr(adj_test, adj_true, threshold=0.05):
    return recall(adj_test, adj_true, threshold)


def fpr(adj_test, adj_true, threshold=0.05):
    fp_val = fp(adj_test, adj_true, threshold)
    if fp_val == 0:
        return fp_val
    else:
        tot = np.sum(adj_true == 0)
        return fp_val / tot


def tnr(adj_test, adj_true, threshold=0.05):
    tn_val = tn(adj_test, adj_true, threshold)
    if tn_val == 0:
        return tn_val
    else:
        tot = np.sum(adj_true == 0)
        return tn_val / tot


def fnr(adj_test, adj_true, threshold=0.05):
    fn_val = fn(adj_test, adj_true, threshold)
    if fn_val == 0:
        return fn_val
    else:
        tot = np.sum(adj_true > 0)
        return fn_val / tot


def fscore(adj_test, adj_true, threshold=0.05, beta=1.0):
    rec_val = recall(adj_test, adj_true, threshold)
    prec_val = precision(adj_test, adj_true, threshold)
    if (rec_val == 0) and (prec_val == 0):
        return 0.0
    return (1 + beta ** 2) * prec_val * rec_val / (beta ** 2 * prec_val + rec_val)


def tpr(adj_test, adj_true, threshold=0.05):
    return recall(adj_test, adj_true, threshold)


def fpr(adj_test, adj_true, threshold=0.05):
    return fp(adj_test, adj_true, threshold) / np.sum(adj_true == 0)


def nrmse(adj_test, adj_true):
    return np.sqrt(np.sum((adj_test - adj_true) ** 2)) / np.sum(adj_true ** 2)


def relerr(adj_test, adj_true, norm=True, null_norm='min'):
    mask = adj_true > 0
    n_nodes = adj_true.shape[0]
    try:
        if norm:
            if null_norm == 'min':
                rel_err = np.sum(np.abs(adj_test - adj_true)[mask] / adj_true[mask]) + np.abs(adj_test - adj_true)[~mask].sum() / adj_true[mask].min()
            elif null_norm == 'none':
                rel_err = np.sum(np.abs(adj_test - adj_true)[mask] / adj_true[mask]) + np.abs(adj_test - adj_true)[~mask].sum()
            elif null_norm == 'ignore':
                rel_err = np.sum(np.abs(adj_test - adj_true)[mask] / adj_true[mask])
            elif isinstance(null_norm, float):
                rel_err = np.sum(np.abs(adj_test - adj_true)[mask] / adj_true[mask]) + np.abs(adj_test - adj_true)[~mask].sum() / null_norm
            else:
                raise ValueError('Invalid norm')
        else:
            rel_err = np.sum(np.abs(adj_test - adj_true))
    except Exception:
        rel_err = np.nan
    return rel_err / n_nodes
