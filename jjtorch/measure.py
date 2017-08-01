import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score as ap


def _score_to_rank(score_list):
    rank_array = np.zeros([len(score_list)])
    score_array = np.array(score_list)
    idx_sorted = (-score_array).argsort()
    rank_array[idx_sorted] = np.arange(len(score_list))+1

    rank_list = rank_array.tolist()
    return rank_list


def _prob2idx(prob_list):
    '''
    each row is a probability distribution of one instance
    '''
    idx = np.argmax(prob_list, axis=1)
    return idx


"""
each row is an instance
each column is the prediction of a class
"""


def auc(Y_target, Y_score):
    """
    Y_target: list of lists. {0, 1}
        real labels

    Y_score: list of lists. real values
        prediction values
    """
    # Y_target = np.squeeze(np.array(Y_target))
    # Y_score = np.squeeze(np.array(Y_score))
    Y_target = np.array(Y_target)
    Y_score = np.array(Y_score)
    auc_list = []
    for i in range(Y_score.shape[1]):
        # try:
        try:
            auc = roc_auc_score(Y_target[:, i], Y_score[:, i])
        except:
            continue
        # except:
        #     auc = -1
        auc_list.append(auc)
    # print("AUC: %f" % (auc))
    return auc_list


def auc_y_classwise(Y_target, Y_score):
    """
    Y_target: list of lists. {0, 1}
        real labels

    Y_score: list of lists. real values
        prediction values
    """
    # Y_target = np.squeeze(np.array(Y_target))
    # Y_score = np.squeeze(np.array(Y_score))
    Y_target = np.array(Y_target)
    Y_score = np.array(Y_score)
    auc_list = roc_auc_score(Y_target, Y_score, average=None)
    return auc_list


def ap_y_classwise(Y_target, Y_score):
    """
    Y_target: list of lists. {0, 1}
        real labels

    Y_score: list of lists. real values
        prediction values
    """
    # Y_target = np.squeeze(np.array(Y_target))
    # Y_score = np.squeeze(np.array(Y_score))
    Y_target = np.array(Y_target)
    Y_score = np.array(Y_score)
    ap_list = ap(Y_target, Y_score, average=None)
    return ap_list


def mean_auc(Y_target, Y_score):
    auc_list = auc(Y_target, Y_score)
    mean_auc = np.mean(auc_list)
    return mean_auc


def mean_auc_y(Y_target, Y_score):
    '''
    along y-axis
    '''
    return mean_auc(Y_target, Y_score)


def mean_auc_x(Y_target, Y_score):
    '''
    along x-axis
    '''
    return mean_auc(np.array(Y_target).T, np.array(Y_score).T)


def hamming_loss(Y_target, Y_predicted):
    """
    Y_target: list of lists. {0, 1}
        real labels

    Y_predicted: list of lists. {0, 1}
        prediction labels
    """
    p = len(Y_target)
    q = len(Y_target[0])
    temp_sum = 0
    for y_target, y_predicted in zip(Y_target, Y_predicted):
        y_target = np.array(y_target)
        y_predicted = np.array(y_predicted)
        idx_target = set((y_target > 0).nonzero()[0].tolist())
        idx_predicted = set((y_predicted > 0).nonzero()[0].tolist())
        n_diff = len(set.symmetric_difference(idx_target, idx_predicted))
        temp_sum += n_diff/float(q)

    measure = temp_sum/p
    return measure


def one_error(Y_target, Y_score):
    """
    Y_target: list of lists. {0, 1}
        real labels

    Y_score: list of lists. real values
        prediction values
    """
    p = float(len(Y_target))
    temp_sum = 0
    for y_target, y_score in zip(Y_target, Y_score):
        y_target = np.array(y_target)
        y_score = np.array(y_score)
        if (y_target == 0).all() or (y_target == 1).all():
            continue
        idx_target = np.nonzero(y_target > 0)[0]
        idx_best = np.argmax(y_score)
        if idx_best not in idx_target:
            temp_sum += 1

    measure = temp_sum/p
    return measure


def coverage(Y_target, Y_score):
    """
    Y_target: list of lists. {0, 1}
        real labels

    Y_score: list of lists. real values
        prediction values
    """
    p = float(len(Y_target))
    temp_sum = 0
    for y_target, y_score in zip(Y_target, Y_score):
        y_target = np.array(y_target)
        y_score = np.array(y_score)
        if (y_target == 0).all() or (y_target == 1).all():
            continue
        idx_target = np.nonzero(y_target > 0)[0]
        rank_list = np.array(_score_to_rank(y_score))
        target_rank_list = rank_list[idx_target]
        temp_sum += max(target_rank_list)-1

    measure = temp_sum/p
    return measure


def ranking_loss(Y_target, Y_score):
    """
    Y_target: list of lists. {0, 1}
        real labels

    Y_score: list of lists. real values
        prediction values
    """
    p = float(len(Y_target))
    temp_sum = 0
    for y_target, y_score in zip(Y_target, Y_score):
        y_target = np.array(y_target)
        y_score = np.array(y_score)
        if (y_target == 0).all() or (y_target == 1).all():
            continue
        idx_target = np.nonzero(y_target > 0)[0]
        idx_nontarget = np.nonzero(y_target <= 0)[0]
        n_target = float(len(idx_target))
        n_nontarget = float(len(idx_nontarget))

        loss = sum([1 for ii in idx_target for jj in idx_nontarget
                    if y_score[ii] <= y_score[jj]])
        temp_sum += (1/(n_target*n_nontarget))*loss

    measure = temp_sum/p
    return measure


def average_precision(y_target, y_score):
    y_target = np.array(y_target)
    y_score = np.array(y_score)

    idx_target = np.nonzero(y_target > 0)[0]
    # idx_nontarget = np.nonzero(y_target <= 0)[0]
    n_target = float(len(idx_target))
    rank_list = np.array(_score_to_rank(y_score))
    target_rank_list = rank_list[idx_target]

    temp_sum_2 = 0
    for target_rank in target_rank_list:
        mm = sum([1 for ii in idx_target
                  if rank_list[ii] <= target_rank])/float(target_rank)
        temp_sum_2 += mm
    score = temp_sum_2/n_target
    return score


def mean_average_precision(Y_target, Y_score):
    """
    mean average precision
    raw-based operation

    Y_target: list of lists. {0, 1}
        real labels

    Y_score: list of lists. real values
        prediction values
    """
    p = float(len(Y_target))
    temp_sum = 0
    for y_target, y_score in zip(Y_target, Y_score):
        y_target = np.array(y_target)
        y_score = np.array(y_score)
        if (y_target == 0).all() or (y_target == 1).all():
            p -= 1
            continue
        idx_target = np.nonzero(y_target > 0)[0]
        # idx_nontarget = np.nonzero(y_target <= 0)[0]
        n_target = float(len(idx_target))
        rank_list = np.array(_score_to_rank(y_score))
        target_rank_list = rank_list[idx_target]

        temp_sum_2 = 0
        for target_rank in target_rank_list:
            mm = sum([1 for ii in idx_target
                      if rank_list[ii] <= target_rank])/float(target_rank)
            temp_sum_2 += mm
        temp_sum += temp_sum_2/n_target

    measure = temp_sum/p
    return measure


def map(Y_target, Y_score):
    return mean_average_precision(Y_target, Y_score)


def map_x(Y_target, Y_score):
    return mean_average_precision(Y_target, Y_score)


def map_y(Y_target, Y_score):
    return mean_average_precision(np.array(Y_target).T,
                                  np.array(Y_score).T)


def precision_at_k_y_axis(Y_target, Y_predicted, k):
    """
    Y_target: list of lists. {0, 1}
        real labels

    Y_predicted: list of lists. {0, 1}
        prediction labels
    """
    Y_tgt = Y_target.T
    Y_pr = Y_predicted.T

    score_list = []
    for y_tgt, y_pr in zip(Y_tgt, Y_pr):
        idx_list = sorted(range(len(y_tgt)),
                          key=lambda ii: y_pr[ii], reverse=True)
        idx_list = idx_list[:k]
        y_tgt_temp = np.array(y_tgt)[idx_list]
        score_ = sum(y_tgt_temp)/float(k)
        score_list.append(score_)
    score = np.mean(score_list)

    return score


def precision_at_10_y_axis(Y_target, Y_predicted):
    """
    Y_target: list of lists. {0, 1}
        real labels

    Y_predicted: list of lists. {0, 1}
        prediction labels
    """
    k = 10
    score = precision_at_k_y_axis(Y_target, Y_predicted, k)

    return score


def f1_score_one(y_target, y_predicted, average=None):
    return f1_score(y_target, y_predicted, average=None)


def precision_score_one(y_target, y_predicted, average=None):
    return precision_score(y_target, y_predicted, average=None)


def recall_score_one(y_target, y_predicted, average=None):
    return recall_score(y_target, y_predicted, average=None)


def f1(Y_target, Y_predicted, average='binary'):
    """
    Y_target: list of lists. {0, 1}
        real labels

    Y_predicted: list of lists. {0, 1}
        prediction labels
    """
    out_list = []
    for y_target, y_predicted in zip(Y_target, Y_predicted):
        out_list.append(f1_score(y_target, y_predicted, average=average))

    return out_list


def recall(Y_target, Y_predicted, average='binary'):
    """
    Y_target: list of lists. {0, 1}
        real labels

    Y_predicted: list of lists. {0, 1}
        prediction labels
    """
    out_list = []
    for y_target, y_predicted in zip(Y_target, Y_predicted):
        out_list.append(recall_score(y_target, y_predicted,
                                     average=average))

    return out_list


def precision(Y_target, Y_predicted, average='binary'):
    """
    Y_target: list of lists. {0, 1}
        real labels

    Y_predicted: list of lists. {0, 1}
        prediction labels
    """
    out_list = []
    for y_target, y_predicted in zip(Y_target, Y_predicted):
        out_list.append(precision_score(y_target, y_predicted,
                                        average=average))

    return out_list


def mean_precision(Y_target, Y_predicted):
    """
    Y_target: list of lists. {0, 1}
        real labels

    Y_predicted: list of lists. {0, 1}
        prediction labels
    """
    out_list = precision(Y_target, Y_predicted)
    return np.mean(out_list)


def mean_recall(Y_target, Y_predicted):
    """
    Y_target: list of lists. {0, 1}
        real labels

    Y_predicted: list of lists. {0, 1}
        prediction labels
    """
    out_list = recall(Y_target, Y_predicted)
    return np.mean(out_list)


def mean_f1(Y_target, Y_predicted):
    """
    Y_target: list of lists. {0, 1}
        real labels

    Y_predicted: list of lists. {0, 1}
        prediction labels
    """
    out_list = f1(Y_target, Y_predicted)
    return np.mean(out_list)


def mcd(Y_target, Y_score):
    """
    measure conversion quality
    """
    mcd_list = []
    for y_target, y_score in zip(Y_target, Y_score):
        mcd = np.mean((10/np.log(10))*(2*np.sum((y_score-y_target)**2,
                                                axis=1))**0.5)
        mcd_list.append(mcd)
    return np.mean(mcd_list)


def mcd_one(Y_target, Y_score):
    """
    measure conversion quality
    """
    mcd_list = []
    for y_target, y_score in zip(Y_target, Y_score):
        mcd = np.mean((10/np.log(10))*(2*np.sum((y_score-y_target)**2
                                                ))**0.5)
        mcd_list.append(mcd)
    return np.mean(mcd_list)


def rmse(Y_target, Y_score):
    """
    root mean squared error
    """
    rmse_list = []
    for y_target, y_score in zip(Y_target, Y_score):
        rmse = rmse_one(y_target, y_score)
        rmse_list.append(rmse)
    return np.mean(rmse_list)


def rmse_one(y_target, y_score):
    rmse = np.mean(np.mean((y_target-y_score)**2, axis=1)**0.5)
    return rmse


def confusion_mat(Y_target, Y_score):
    vec_target = _prob2idx(Y_target)
    vec_pr = _prob2idx(Y_score)

    mat = confusion_matrix(vec_target, vec_pr)
    return mat


def LSD(Y_target, Y_score):
    '''
    log spectral distance
    Y_score:
        predicted power spectrum
    Y_target:
        target power spectrum
    '''
    lsd_list = np.mean((10*np.log10(Y_target/Y_score))**2, axis=1)**0.5

    lsd = np.mean(lsd_list)

    return lsd


def _LSD(Y_target, Y_score):
    '''
    log spectral distance
    Y_score:
        predicted power spectrum
    Y_target:
        target power spectrum
    '''
    temp = 10*np.log10(Y_target/Y_score)**2
    notnan = [[not np.isnan(t) for t in term]for term in temp]
    temp = [term[nnn] for term, nnn in zip(temp, notnan)]

    lsd_list = [np.mean(term)**0.5 for term in temp]

    lsd = np.mean(lsd_list)

    return lsd


# For frame evaluation
def f1_micro(y_target, y_predicted):
    """
    y_target: m x n 2D array. {0, 1}
        real labels

    y_predicted: m x n 2D array {0, 1}
        prediction labels

    m (y-axis): # of instances
    n (x-axis): # of classes
    """
    average = 'micro'
    score = f1_score(y_target, y_predicted, average=average)
    return score


def f1_macro(y_target, y_predicted):
    """
    y_target: m x n 2D array. {0, 1}
        real labels

    y_predicted: m x n 2D array {0, 1}
        prediction labels

    m (y-axis): # of instances
    n (x-axis): # of classes
    """
    average = 'macro'
    score = f1_score(y_target, y_predicted, average=average)
    return score


def precision_micro(y_target, y_predicted):
    """
    y_target: m x n 2D array. {0, 1}
        real labels

    y_predicted: m x n 2D array {0, 1}
        prediction labels

    m (y-axis): # of instances
    n (x-axis): # of classes
    """
    average = 'micro'
    score = precision_score(y_target, y_predicted, average=average)
    return score


def precision_macro(y_target, y_predicted):
    """
    y_target: m x n 2D array. {0, 1}
        real labels

    y_predicted: m x n 2D array {0, 1}
        prediction labels

    m (y-axis): # of instances
    n (x-axis): # of classes
    """
    average = 'macro'
    score = precision_score(y_target, y_predicted, average=average)
    return score


def recall_micro(y_target, y_predicted):
    """
    y_target: m x n 2D array. {0, 1}
        real labels

    y_predicted: m x n 2D array {0, 1}
        prediction labels

    m (y-axis): # of instances
    n (x-axis): # of classes
    """
    average = 'micro'
    score = recall_score(y_target, y_predicted, average=average)
    return score


def recall_macro(y_target, y_predicted):
    """
    y_target: m x n 2D array. {0, 1}
        real labels

    y_predicted: m x n 2D array {0, 1}
        prediction labels

    m (y-axis): # of instances
    n (x-axis): # of classes
    """
    average = 'macro'
    score = recall_score(y_target, y_predicted, average=average)
    return score


# For multi-class
def accuracy(y_target, y_predicted):
    return accuracy_score(y_target, y_predicted)


def accuracy_array2idx(y_target, y_predicted):
    '''
    y_predicted: m x n 2D array
    m: the number of instances
    n: the dimension of output
    each row contains exactly one 1

    '''
    y_target_idx = np.argmax(y_target, axis=1)
    y_predicted_idx = np.argmax(y_predicted, axis=1)
    return accuracy(y_target_idx, y_predicted_idx)
