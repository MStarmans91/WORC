#!/usr/bin/env python

# Copyright 2016-2020 Biomedical Imaging Group Rotterdam, Departments of
# Medical Informatics and Radiology, Erasmus MC, Rotterdam, The Netherlands
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.metrics import check_scoring as check_scoring_sklearn
from scipy.linalg import pinv
from imblearn.metrics import geometric_mean_score


def performance_singlelabel(y_truth, y_prediction, y_score, regression=False):
    '''
    Singleclass performance metrics
    '''
    if regression:
        y_truth = np.array(y_truth).flatten()
        r2score = metrics.r2_score(y_truth, y_prediction)
        MSE = metrics.mean_squared_error(y_truth, y_prediction)
        coefICC = ICC(np.column_stack((y_prediction, y_truth)))
        C = pearsonr(y_prediction, y_truth)
        PearsonC = C[0]
        PearsonP = C[1]
        C = spearmanr(y_prediction, y_truth)
        SpearmanC = C.correlation
        SpearmanP = C.pvalue

        return r2score, MSE, coefICC, PearsonC, PearsonP, SpearmanC, SpearmanP

    else:
        # Compute confuction matrics and extract measures
        c_mat = confusion_matrix(y_truth, y_prediction)
        if c_mat.shape[0] == 0:
            print('[WORC Warning] No samples in y_truth and y_prediction.')
            TN = 0
            FN = 0
            TP = 0
            FP = 0
        elif c_mat.shape[0] == 1:
            print('[WORC Warning] Only a single class represented in y_truth and y_prediction.')
            if 0 in c_mat:
                TN = c_mat[0, 0]
                FN = 0
                TP = 0
                FP = 0
            else:
                TN = 0
                FN = 0
                TP = c_mat[0, 0]
                FP = 0
        else:
            TN = c_mat[0, 0]
            FN = c_mat[1, 0]
            TP = c_mat[1, 1]
            FP = c_mat[0, 1]

        if FN == 0 and TP == 0:
            if c_mat.shape[0] != 2:
                sensitivity = np.NaN
            else:
                sensitivity = 0
        else:
            sensitivity = float(TP)/(TP+FN)

        if FP == 0 and TN == 0:
            if c_mat.shape[0] != 2:
                specificity = np.NaN
            else:
                specificity = 0
        else:
            specificity = float(TN)/(FP+TN)

        if TP == 0 and FP == 0:
            if c_mat.shape[0] != 2:
                precision = np.NaN
            else:
                precision = 0
        else:
            precision = float(TP)/(TP+FP)

        if TN == 0 and FN == 0:
            if c_mat.shape[0] != 2:
                npv = np.NaN
            else:
                npv = 0
        else:
            npv = float(TN) / (TN + FN)

        # Additionally, compute accuracy, AUC and f1-score
        accuracy = accuracy_score(y_truth, y_prediction)
        BCA = balanced_accuracy_score(y_truth, y_prediction)
        try:
            auc = roc_auc_score(y_truth, y_score)
        except ValueError as e:
            print('[WORC Warning] ' + str(e) + '. Setting AUC to NaN.')
            auc = np.NaN

        f1_score_out = f1_score(y_truth, y_prediction, average='weighted')

        return accuracy, BCA, sensitivity, specificity, precision, npv, f1_score_out, auc


def performance_multilabel(y_truth, y_prediction, y_score=None, beta=1):
    '''
    Multiclass performance metrics.

    y_truth and y_prediction should both be lists with the multiclass label of each
    object, e.g.

    y_truth = [0, 0,	0,	0,	0,	0,	2,	2,	1,	1,	2]    ### Groundtruth
    y_prediction = [0, 0,	0,	0,	0,	0,	1,	2,	1,	2,	2]    ### Predicted labels
    y_score = [[0.3, 0.3, 0.4], [0.2, 0.6, 0.2], ... ] # Normalized score per patient for all labels (three in this example)


    Calculation of accuracy accorading to formula suggested in CAD Dementia Grand Challege http://caddementia.grand-challenge.org
    and the TADPOLE challenge https://tadpole.grand-challenge.org/Performance_Metrics/
    Calculation of Multi Class AUC according to classpy: https://bitbucket.org/bigr_erasmusmc/classpy/src/master/classpy/multi_class_auc.py

    '''
    cm = confusion_matrix(y_truth, y_prediction)

    # Determine no. of classes
    labels_class = np.unique(y_truth)
    n_class = len(labels_class)

    # Splits confusion matrix in true and false positives and negatives
    TP = np.zeros(shape=(1, n_class), dtype=int)
    FN = np.zeros(shape=(1, n_class), dtype=int)
    FP = np.zeros(shape=(1, n_class), dtype=int)
    TN = np.zeros(shape=(1, n_class), dtype=int)
    n = np.zeros(shape=(1, n_class), dtype=int)
    for i in range(n_class):
        TP[:, i] = cm[i, i]
        FN[:, i] = np.sum(cm[i, :])-cm[i, i]
        FP[:, i] = np.sum(cm[:, i])-cm[i, i]
        TN[:, i] = np.sum(cm[:])-TP[:, i]-FP[:, i]-FN[:, i]

    n = np.sum(cm)

    # Determine Accuracy
    Accuracy = (np.sum(TP))/n

    # BCA: Balanced Class Accuracy
    BCA = list()
    for i in range(n_class):
        BCAi = 1/2*(TP[:, i]/(TP[:, i] + FN[:, i]) + TN[:, i]/(TN[:, i] + FP[:, i]))
        BCA.append(BCAi)

    AverageAccuracy = np.mean(BCA)

    # Determine total positives and negatives
    P = TP + FN
    N = FP + TN

    # Calculation of sensitivity
    Sensitivity = TP/P
    Sensitivity = np.mean(Sensitivity)

    # Calculation of specifitity
    Specificity = TN/N
    Specificity = np.mean(Specificity)

    # Calculation of precision
    Precision = TP/(TP+FP)
    Precision = np.nan_to_num(Precision)
    Precision = np.mean(Precision)

    # Calculation of NPV
    NPV = TN/(TN+FN)
    NPV = np.nan_to_num(NPV)
    NPV = np.mean(NPV)

    # Calculation  of F1_Score
    F1_score = ((1+(beta**2))*(Sensitivity*Precision))/((beta**2)*(Precision + Sensitivity))
    F1_score = np.nan_to_num(F1_score)
    F1_score = np.mean(F1_score)

    # Calculation of Multi Class AUC according to classpy: https://bitbucket.org/bigr_erasmusmc/classpy/src/master/classpy/multi_class_auc.py
    if y_score is not None:
        AUC = multi_class_auc(y_truth, y_score)
    else:
        AUC = None

    return Accuracy, Sensitivity, Specificity, Precision, NPV, F1_score, AUC, AverageAccuracy


def pairwise_auc(y_truth, y_score, class_i, class_j):
    # Filter out the probabilities for class_i and class_j
    y_score = [est[class_i] for ref, est in zip(y_truth, y_score) if ref in (class_i, class_j)]
    y_truth = [ref for ref in y_truth if ref in (class_i, class_j)]

    # Sort the y_truth by the estimated probabilities
    sorted_y_truth = [y for x, y in sorted(zip(y_score, y_truth), key=lambda p: p[0])]

    # Calculated the sum of ranks for class_i
    sum_rank = 0
    for index, element in enumerate(sorted_y_truth):
        if element == class_i:
            sum_rank += index + 1
    sum_rank = float(sum_rank)

    # Get the counts for class_i and class_j
    n_class_i = float(y_truth.count(class_i))
    n_class_j = float(y_truth.count(class_j))

    # If a class in empty, AUC is 0.0
    if n_class_i == 0 or n_class_j == 0:
        return 0.0

    # Calculate the pairwise AUC
    return (sum_rank - (0.5 * n_class_i * (n_class_i + 1))) / (n_class_i * n_class_j)


def multi_class_auc(y_truth, y_score):
    classes = np.unique(y_truth)

    # if any(t == 0.0 for t in np.sum(y_score, axis=1)):
    #     raise ValueError('No AUC is calculated, output probabilities are missing')

    pairwise_auc_list = [0.5 * (pairwise_auc(y_truth, y_score, i, j) +
                                pairwise_auc(y_truth, y_score, j, i)) for i in classes for j in classes if i < j]

    c = len(classes)
    return (2.0 * sum(pairwise_auc_list)) / (c * (c - 1))


def multi_class_auc_score(y_truth, y_score):
    return make_scorer(multi_class_auc, needs_proba=True)


def f1_weighted_predictproba(y_truth, y_score):
    '''Calculate f1-score, but based on predict_proba instead of predict.

    Probabilities are thresholded at 0.5.
    '''
    # Convert predictions to binary by thresholding at 0.5
    y_pred = np.zeros(y_score.shape)
    y_pred[y_score >= 0.5] = 1
    y_pred[y_score < 0.5] = 0

    # Compute and return score
    return f1_score(y_truth, y_pred, average='weighted')


def check_scoring(estimator, scoring=None, allow_none=False):
    '''
    Surrogate for sklearn's check_scoring to enable use of some other
    scoring metrics.
    '''
    if scoring == 'average_precision_weighted':
        scorer = make_scorer(average_precision_score, average='weighted', needs_proba=True)
    elif scoring == 'gmean':
        scorer = make_scorer(geometric_mean_score, needs_proba=True)
    elif scoring == 'f1_weighted_predictproba':
        scorer = make_scorer(f1_weighted_predictproba, needs_proba=True)
    else:
        scorer = check_scoring_sklearn(estimator, scoring=scoring)
    return scorer


def check_multimetric_scoring(estimator, scoring=None):
    """Wrapper around sklearn function to enable more scoring options.


    Check the scoring parameter in cases when multiple metrics are allowed

    Parameters
    ----------
    estimator : sklearn estimator instance
        The estimator for which the scoring will be applied.
    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.
        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.
        See :ref:`multimetric_grid_search` for an example.
        If None the estimator's score method is used.
        The return value in that case will be ``{'score': <default_scorer>}``.
        If the estimator's score method is not available, a ``TypeError``
        is raised.
    Returns
    -------
    scorers_dict : dict
        A dict mapping each scorer name to its validated scorer.
    is_multimetric : bool
        True if scorer is a list/tuple or dict of callables
        False if scorer is None/str/callable
    """
    if callable(scoring) or scoring is None or isinstance(scoring,
                                                          str):
        scorers = {"score": check_scoring(estimator, scoring=scoring)}
        return scorers, False
    else:
        err_msg_generic = ("scoring should either be a single string or "
                           "callable for single metric evaluation or a "
                           "list/tuple of strings or a dict of scorer name "
                           "mapped to the callable for multiple metric "
                           "evaluation. Got %s of type %s"
                           % (repr(scoring), type(scoring)))

        if isinstance(scoring, (list, tuple, set)):
            err_msg = ("The list/tuple elements must be unique "
                       "strings of predefined scorers. ")
            invalid = False
            try:
                keys = set(scoring)
            except TypeError:
                invalid = True
            if invalid:
                raise ValueError(err_msg)

            if len(keys) != len(scoring):
                raise ValueError(err_msg + "Duplicate elements were found in"
                                 " the given list. %r" % repr(scoring))
            elif len(keys) > 0:
                if not all(isinstance(k, str) for k in keys):
                    if any(callable(k) for k in keys):
                        raise ValueError(err_msg +
                                         "One or more of the elements were "
                                         "callables. Use a dict of score name "
                                         "mapped to the scorer callable. "
                                         "Got %r" % repr(scoring))
                    else:
                        raise ValueError(err_msg +
                                         "Non-string types were found in "
                                         "the given list. Got %r"
                                         % repr(scoring))
                scorers = {scorer: check_scoring(estimator, scoring=scorer)
                           for scorer in scoring}
            else:
                raise ValueError(err_msg +
                                 "Empty list was given. %r" % repr(scoring))

        elif isinstance(scoring, dict):
            keys = set(scoring)
            if not all(isinstance(k, str) for k in keys):
                raise ValueError("Non-string types were found in the keys of "
                                 "the given dict. scoring=%r" % repr(scoring))
            if len(keys) == 0:
                raise ValueError("An empty dict was passed. %r"
                                 % repr(scoring))
            scorers = {key: check_scoring(estimator, scoring=scorer)
                       for key, scorer in scoring.items()}
        else:
            raise ValueError(err_msg_generic)
        return scorers, True


def ICC(M, ICCtype='inter'):
    '''
    Input:
        M is matrix of observations. Rows: patients, columns: observers.
        type: ICC type, currently "inter" or "intra".
    '''

    n, k = M.shape

    SStotal = np.var(M, ddof=1) * (n*k - 1)
    MSR = np.var(np.mean(M, 1), ddof=1) * k
    MSW = np.sum(np.var(M, 1, ddof=1)) / n
    MSC = np.var(np.mean(M, 0), ddof=1) * n
    MSE = (SStotal - MSR * (n - 1) - MSC * (k -1)) / ((n - 1) * (k - 1))

    if ICCtype == 'intra':
        r = (MSR - MSW) / (MSR + (k-1)*MSW)
    elif ICCtype == 'inter':
        r = (MSR - MSE) / (MSR + (k-1)*MSE + k*(MSC-MSE)/n)
    else:
        raise ValueError('No valid ICC type given.')

    return r


def ICC_anova(Y, ICCtype='inter', more=False):
    '''
    Adopted from Nipype with a slight alteration to distinguish inter and intra.
    the data Y are entered as a 'table' ie subjects are in rows and repeated
    measures in columns
    One Sample Repeated measure ANOVA
    Y = XB + E with X = [FaTor / Subjects]
    '''

    [nb_subjects, nb_conditions] = Y.shape
    dfc = nb_conditions - 1
    dfe = (nb_subjects - 1) * dfc
    dfr = nb_subjects - 1

    # Compute the repeated measure effect
    # ------------------------------------

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(nb_conditions), np.ones((nb_subjects, 1)))  # sessions
    x0 = np.tile(np.eye(nb_subjects), (nb_conditions, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(np.dot(np.dot(X, pinv(np.dot(X.T, X))), X.T), Y.flatten('F'))
    residuals = Y.flatten('F') - predicted_Y
    SSE = (residuals ** 2).sum()

    residuals.shape = Y.shape

    MSE = SSE / dfe

    # Sum square session effect - between colums/sessions
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * nb_subjects
    MSC = SSC / dfc / nb_subjects

    session_effect_F = MSC / MSE

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    # ICC(3,1) = (mean square subjeT - mean square error) / (mean square subjeT + (k-1)*-mean square error)
    if ICCtype == 'intra':
        ICC = (MSR - MSE) / (MSR + dfc*MSE)
    elif ICCtype == 'inter':
        ICC = (MSR - MSE) / (MSR + dfc*MSE + nb_conditions*(MSC-MSE)/nb_subjects)
    else:
        raise ValueError('No valid ICC type given.')

    e_var = MSE  # variance of error
    r_var = (MSR - MSE) / nb_conditions  # variance between subjects

    if more:
        return ICC, r_var, e_var, session_effect_F, dfc, dfe
    else:
        return ICC
