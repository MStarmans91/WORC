#!/usr/bin/env python

# Copyright 2016-2021 Biomedical Imaging Group Rotterdam, Departments of
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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import tikzplotlib
import pandas as pd
import argparse
from WORC.plotting.compute_CI import compute_confidence as CI
import numpy as np
from sklearn.metrics import roc_auc_score, auc
import csv
from WORC.plotting.plot_estimator_performance import plot_estimator_performance


def plot_single_ROC(y_truth, y_score, verbose=False, returnplot=False):
    '''
    Get the False Positive Ratio (FPR) and True Positive Ratio (TPR)
    for the ground truth and score of a single estimator. These ratios
    can be used to plot a Receiver Operator Characteristic (ROC) curve.
    '''
    # Sort both lists based on the scores
    y_truth = np.asarray(y_truth)
    y_truth = np.int_(y_truth)
    y_score = np.asarray(y_score)
    inds = y_score.argsort()
    y_truth_sorted = y_truth[inds]
    y_score = y_score[inds]

    # Compute the TPR and FPR for all possible thresholds
    FP = 0
    TP = 0
    fpr = list()
    tpr = list()
    thresholds = list()
    fprev = -np.inf
    i = 0
    N = float(np.bincount(y_truth)[0])
    if len(np.bincount(y_truth)) == 1:
        # No class = 1 present.
        P = 0
    else:
        P = float(np.bincount(y_truth)[1])

    if N == 0:
        print('[WORC Warning] No negative class samples found, cannot determine ROC. Skipping iteration.')
        return fpr, tpr, thresholds
    elif P == 0:
        print('[WORC Warning] No positive class samples found, cannot determine ROC. Skipping iteration.')
        return fpr, tpr, thresholds

    while i < len(y_truth_sorted):
        if y_score[i] != fprev:
            fpr.append(1 - FP/N)
            tpr.append(1 - TP/P)
            thresholds.append(y_score[i])
            fprev = y_score[i]

        if y_truth_sorted[i] == 1:
            TP += 1
        else:
            FP += 1

        i += 1

    if verbose or returnplot:
        roc_auc = auc(fpr, tpr)
        f = plt.figure()
        ax = plt.subplot(111)
        lw = 2
        ax.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

    if not returnplot:
        return fpr[::-1], tpr[::-1], thresholds[::-1]
    else:
        return fpr[::-1], tpr[::-1], thresholds[::-1], f


def plot_single_PRC(y_truth, y_score, verbose=False, returnplot=False):
    '''
    Get the precision and recall (=true positive rate)
    for the ground truth and score of a single estimator. These ratios
    can be used to plot a Precision Recall Curve (ROC).
    '''
    # Sort both lists based on the scores
    y_truth = np.asarray(y_truth)
    y_truth = np.int_(y_truth)
    y_score = np.asarray(y_score)
    inds = y_score.argsort()
    y_truth_sorted = y_truth[inds]
    y_score = y_score[inds]

    # Compute the TPR and FPR for all possible thresholds
    FP = 0
    TP = 0
    precision = list()
    tpr = list()
    thresholds = list()
    fprev = -np.inf
    i = 0
    N = float(np.bincount(y_truth)[0])
    if len(np.bincount(y_truth)) == 1:
        # No class = 1 present.
        P = 0
    else:
        P = float(np.bincount(y_truth)[1])

    if N == 0:
        print('[WORC Warning] No negative class samples found, cannot determine PRC. Skipping iteration.')
        return precision, tpr, thresholds
    elif P == 0:
        print('[WORC Warning] No positive class samples found, cannot determine PRC. Skipping iteration.')
        return precision, tpr, thresholds

    while i < len(y_truth_sorted):
        if y_score[i] != fprev:
            if TP == 0:
                # Prevent division by zero
                precision.append(0)
            else:
                precision.append(TP/(TP + FP))

            tpr.append(1 - TP/P)
            thresholds.append(y_score[i])
            fprev = y_score[i]

        if y_truth_sorted[i] == 1:
            TP += 1
        else:
            FP += 1

        i += 1

    if verbose or returnplot:
        prc_auc = auc(tpr, precision)
        f = plt.figure()
        ax = plt.subplot(111)
        lw = 2
        ax.plot(tpr, precision, color='darkorange',
                lw=lw, label='PR curve (area = %0.2f)' % prc_auc)
        ax.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend(loc="lower right")

    if not returnplot:
        return tpr[::-1], precision[::-1], thresholds[::-1]
    else:
        return tpr[::-1], precision[::-1], thresholds[::-1], f


def curve_thresholding(metric1t, metric2t, thresholds, nsamples=20):
    '''
    Construct metric1 and metric2 (either FPR and TPR, or TPR and Precision)
    ratios at different thresholds for the scores of an estimator.
    '''
    # Combine all found thresholds in a list and create samples
    T = list()
    for t in thresholds:
        T.extend(t)
    T = sorted(T)
    tsamples = np.linspace(0, len(T) - 1, nsamples)

    # Compute the metric1s and metric2s at the sample points
    nrocs = len(metric1t)
    metric1 = np.zeros((nsamples, nrocs))
    metric2 = np.zeros((nsamples, nrocs))

    th = list()
    for n_sample, tidx in enumerate(tsamples):
        tidx = int(tidx)
        th.append(T[tidx])
        for i_roc in range(0, nrocs):
            idx = 0
            while float(thresholds[i_roc][idx]) > float(T[tidx]) and idx < (len(thresholds[i_roc]) - 1):
                idx += 1
            metric1[n_sample, i_roc] = metric1t[i_roc][idx]
            metric2[n_sample, i_roc] = metric2t[i_roc][idx]

    return metric1, metric2, th


def plot_ROC_CIc(y_truth, y_score, N_1, N_2, plot='default', alpha=0.95,
                 verbose=False, DEBUG=False, tsamples=20):
    '''
    Plot a Receiver Operator Characteristic (ROC) curve with confidence intervals.

    tsamples: number of sample points on which to determine the confidence intervals.
              The sample pointsare used on the thresholds for y_score.
    '''
    # Compute ROC curve and ROC area for each class
    fprt = list()
    tprt = list()
    roc_auc = list()
    thresholds = list()
    for yt, ys in zip(y_truth, y_score):
        fpr_temp, tpr_temp, thresholds_temp = plot_single_ROC(yt, ys)
        if fpr_temp:
            roc_auc.append(roc_auc_score(yt, ys))
            fprt.append(fpr_temp)
            tprt.append(tpr_temp)
            thresholds.append(thresholds_temp)

    # Sample FPR and TPR at numerous points
    fpr, tpr, th = curve_thresholding(fprt, tprt, thresholds, tsamples)

    # Compute the confidence intervals for the ROC
    CIs_tpr = list()
    CIs_fpr = list()
    for i in range(0, tsamples):
        if i == 0:
            # Point (1, 1) is always in there, but shows as (nan, nan)
            CIs_fpr.append([1, 1])
            CIs_tpr.append([1, 1])
        else:
            cit_fpr = CI(fpr[i, :], N_1, N_2, alpha)
            CIs_fpr.append([cit_fpr[0], cit_fpr[1]])
            cit_tpr = CI(tpr[i, :], N_1, N_2, alpha)
            CIs_tpr.append([cit_tpr[0], cit_tpr[1]])

    # The point (0, 0) is also always there but not computed
    CIs_fpr.append([0, 0])
    CIs_tpr.append([0, 0])

    # Calculate also means of CIs after converting to array
    CIs_tpr = np.asarray(CIs_tpr)
    CIs_fpr = np.asarray(CIs_fpr)
    CIs_tpr_means = np.mean(CIs_tpr, axis=1).tolist()
    CIs_fpr_means = np.mean(CIs_fpr, axis=1).tolist()

    # compute AUC CI
    roc_auc = CI(roc_auc, N_1, N_2, alpha)

    f = plt.figure()
    lw = 2
    subplot = f.add_subplot(111)
    subplot.plot(CIs_fpr_means, CIs_tpr_means, color='orange',
                 lw=lw, label='ROC curve (AUC = (%0.2f, %0.2f))' % (roc_auc[0], roc_auc[1]))

    for i in range(0, len(CIs_fpr_means)):
        if CIs_tpr[i, 1] <= 1:
            ymax = CIs_tpr[i, 1]
        else:
            ymax = 1

        if CIs_tpr[i, 0] <= 0:
            ymin = 0
        else:
            ymin = CIs_tpr[i, 0]

        if CIs_tpr_means[i] <= 1:
            ymean = CIs_tpr_means[i]
        else:
            ymean = 1

        if CIs_fpr[i, 1] <= 1:
            xmax = CIs_fpr[i, 1]
        else:
            xmax = 1

        if CIs_fpr[i, 0] <= 0:
            xmin = 0
        else:
            xmin = CIs_fpr[i, 0]

        if CIs_fpr_means[i] <= 1:
            xmean = CIs_fpr_means[i]
        else:
            xmean = 1

        if DEBUG:
            print(xmin, xmax, ymean)
            print(ymin, ymax, xmean)

        subplot.plot([xmin, xmax],
                     [ymean, ymean],
                     color='black', alpha=0.15)
        subplot.plot([xmean, xmean],
                     [ymin, ymax],
                     color='black', alpha=0.15)

    subplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    if verbose:
        plt.show()

        f = plt.figure()
        lw = 2
        subplot = f.add_subplot(111)
        subplot.plot(CIs_fpr_means, CIs_tpr_means, color='darkorange',
                     lw=lw, label='ROC curve (AUC = (%0.2f, %0.2f))' % (roc_auc[0], roc_auc[1]))

        for i in range(0, len(CIs_fpr_means)):
            if CIs_tpr[i, 1] <= 1:
                ymax = CIs_tpr[i, 1]
            else:
                ymax = 1

            if CIs_tpr[i, 0] <= 0:
                ymin = 0
            else:
                ymin = CIs_tpr[i, 0]

            if CIs_tpr_means[i] <= 1:
                ymean = CIs_tpr_means[i]
            else:
                ymean = 1

            if CIs_fpr[i, 1] <= 1:
                xmax = CIs_fpr[i, 1]
            else:
                xmax = 1

            if CIs_fpr[i, 0] <= 0:
                xmin = 0
            else:
                xmin = CIs_fpr[i, 0]

            if CIs_fpr_means[i] <= 1:
                xmean = CIs_fpr_means[i]
            else:
                xmean = 1

            subplot.plot([xmin, xmax],
                         [ymean, ymean],
                         color='black', alpha=0.15)
            subplot.plot([xmean, xmean],
                         [ymin, ymax],
                         color='black', alpha=0.15)

        subplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

    return f, CIs_fpr, CIs_tpr


def plot_PRC_CIc(y_truth, y_score, N_1, N_2, plot='default', alpha=0.95,
                 verbose=False, DEBUG=False, tsamples=20):
    '''
    Plot a Precision-Recall curve with confidence intervals.

    tsamples: number of sample points on which to determine the confidence intervals.
              The sample pointsare used on the thresholds for y_score.
    '''
    # Compute PR curve and PR area for each class
    tprt = list()
    precisiont = list()
    prc_auc = list()
    thresholds = list()
    for yt, ys in zip(y_truth, y_score):
        tpr_temp, precision_temp, thresholds_temp = plot_single_PRC(yt, ys)
        if tpr_temp:
            prc_auc.append(auc(tpr_temp, precision_temp))
            tprt.append(tpr_temp)
            precisiont.append(precision_temp)
            thresholds.append(thresholds_temp)

    # Sample TPR and precision at numerous points
    tpr, precisionr, th = curve_thresholding(tprt, precisiont, thresholds, tsamples)

    # Compute the confidence intervals for the ROC
    CIs_precisionr = list()
    CIs_tpr = list()
    for i in range(0, tsamples):
        if i == 0:
            # Point (0, 0) is always in there, but shows as (nan, nan)
            CIs_tpr.append([1, 1])
            CIs_precisionr.append([0, 0])
        else:
            cit_tpr = CI(tpr[i, :], N_1, N_2, alpha)
            CIs_tpr.append([cit_tpr[0], cit_tpr[1]])

            cit_precisionr = CI(precisionr[i, :], N_1, N_2, alpha)
            CIs_precisionr.append([cit_precisionr[0], cit_precisionr[1]])

    # The point (0, 1) is also always there but not computed
    CIs_tpr.append([0, 0])
    CIs_precisionr.append([1, 1])

    # Calculate also means of CIs after converting to array
    CIs_precisionr = np.asarray(CIs_precisionr)
    CIs_tpr = np.asarray(CIs_tpr)
    CIs_precisionr_means = np.mean(CIs_precisionr, axis=1).tolist()
    CIs_tpr_means = np.mean(CIs_tpr, axis=1).tolist()

    # compute AUC CI
    prc_auc = CI(prc_auc, N_1, N_2, alpha)

    f = plt.figure()
    lw = 2
    subplot = f.add_subplot(111)
    subplot.plot(CIs_tpr_means, CIs_precisionr_means, color='orange',
                 lw=lw, label='PR curve (AUC = (%0.2f, %0.2f))' % (prc_auc[0], prc_auc[1]))

    for i in range(0, len(CIs_tpr_means)):
        if CIs_precisionr[i, 1] <= 1:
            ymax = CIs_precisionr[i, 1]
        else:
            ymax = 1

        if CIs_precisionr[i, 0] <= 0:
            ymin = 0
        else:
            ymin = CIs_precisionr[i, 0]

        if CIs_precisionr_means[i] <= 1:
            ymean = CIs_precisionr_means[i]
        else:
            ymean = 1

        if CIs_tpr[i, 1] <= 1:
            xmax = CIs_tpr[i, 1]
        else:
            xmax = 1

        if CIs_tpr[i, 0] <= 0:
            xmin = 0
        else:
            xmin = CIs_tpr[i, 0]

        if CIs_tpr_means[i] <= 1:
            xmean = CIs_tpr_means[i]
        else:
            xmean = 1

        if DEBUG:
            print(xmin, xmax, ymean)
            print(ymin, ymax, xmean)

        subplot.plot([xmin, xmax],
                     [ymean, ymean],
                     color='black', alpha=0.15)
        subplot.plot([xmean, xmean],
                     [ymin, ymax],
                     color='black', alpha=0.15)

    subplot.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")

    if verbose:
        plt.show()

        f = plt.figure()
        lw = 2
        subplot = f.add_subplot(111)
        subplot.plot(CIs_tpr_means, CIs_precisionr_means, color='darkorange',
                     lw=lw, label='PRC curve (AUC = (%0.2f, %0.2f))' % (prc_auc[0], prc_auc[1]))

        for i in range(0, len(CIs_tpr_means)):
            if CIs_precisionr[i, 1] <= 1:
                ymax = CIs_precisionr[i, 1]
            else:
                ymax = 1

            if CIs_precisionr[i, 0] <= 0:
                ymin = 0
            else:
                ymin = CIs_precisionr[i, 0]

            if CIs_precisionr[i] <= 1:
                ymean = CIs_precisionr[i]
            else:
                ymean = 1

            if CIs_tpr[i, 1] <= 1:
                xmax = CIs_tpr[i, 1]
            else:
                xmax = 1

            if CIs_tpr[i, 0] <= 0:
                xmin = 0
            else:
                xmin = CIs_tpr[i, 0]

            if CIs_tpr_means[i] <= 1:
                xmean = CIs_tpr_means[i]
            else:
                xmean = 1

            if DEBUG:
                print(xmin, xmax, ymean)
                print(ymin, ymax, xmean)

            subplot.plot([xmin, xmax],
                         [ymean, ymean],
                         color='black', alpha=0.15)
            subplot.plot([xmean, xmean],
                         [ymin, ymax],
                         color='black', alpha=0.15)

        subplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend(loc="lower right")

    return f, CIs_tpr, CIs_precisionr


def main():
    parser = argparse.ArgumentParser(description='Plot the ROC Curve of an estimator')
    parser.add_argument('-prediction', '--prediction', metavar='prediction',
                        nargs='+', dest='prediction', type=str, required=True,
                        help='Prediction file (HDF)')
    parser.add_argument('-pinfo', '--pinfo', metavar='pinfo',
                        nargs='+', dest='pinfo', type=str, required=True,
                        help='Patient Info File (txt)')
    parser.add_argument('-ensemble', '--ensemble', metavar='ensemble',
                        nargs='+', dest='ensemble', type=str, required=True,
                        help='Length of ensemble (int)')
    parser.add_argument('-label_type', '--label_type', metavar='label_type',
                        nargs='+', dest='label_type', type=str, required=True,
                        help='Label name that is predicted (string)')
    parser.add_argument('-ROC_png', '--ROC_png', metavar='ROC_png',
                        nargs='+', dest='ROC_png', type=str, required=False,
                        help='File to write ROC to (PNG)')
    parser.add_argument('-ROC_csv', '--ROC_csv', metavar='ROC_csv',
                        nargs='+', dest='ROC_csv', type=str, required=False,
                        help='File to write ROC to (csv)')
    parser.add_argument('-ROC_tex', '--ROC_tex', metavar='ROC_tex',
                        nargs='+', dest='ROC_tex', type=str, required=False,
                        help='File to write ROC to (tex)')
    parser.add_argument('-PRC_png', '--PRC_png', metavar='PRC_png',
                        nargs='+', dest='PRC_png', type=str, required=False,
                        help='File to write PR to (PNG)')
    parser.add_argument('-PRC_csv', '--PRC_csv', metavar='PRC_csv',
                        nargs='+', dest='PRC_csv', type=str, required=False,
                        help='File to write PR to (csv)')
    parser.add_argument('-PRC_tex', '--PRC_tex', metavar='PRC_tex',
                        nargs='+', dest='PRC_tex', type=str, required=False,
                        help='File to write PR to (tex)')
    args = parser.parse_args()

    plot_ROC(prediction=args.prediction,
             pinfo=args.pinfo,
             ensemble=args.ensemble,
             label_type=args.label_type,
             ROC_png=args.ROC_png,
             ROC_tex=args.ROC_tex,
             ROC_csv=args.ROC_csv,
             PRC_png=args.PRC_png,
             PRC_tex=args.PRC_tex,
             PRC_csv=args.PRC_csv)


def plot_ROC(prediction, pinfo, ensemble=1, label_type=None,
             ROC_png=None, ROC_tex=None, ROC_csv=None,
             PRC_png=None, PRC_tex=None, PRC_csv=None):
    # Convert the inputs to the correct format
    if type(prediction) is list:
        prediction = ''.join(prediction)

    if type(pinfo) is list:
        pinfo = ''.join(pinfo)

    if type(ensemble) is list:
        ensemble = int(ensemble[0])

    if type(ROC_png) is list:
        ROC_png = ''.join(ROC_png)

    if type(ROC_csv) is list:
        ROC_csv = ''.join(ROC_csv)

    if type(ROC_tex) is list:
        ROC_tex = ''.join(ROC_tex)

    if type(PRC_png) is list:
        PRC_png = ''.join(PRC_png)

    if type(PRC_csv) is list:
        PRC_csv = ''.join(PRC_csv)

    if type(PRC_tex) is list:
        PRC_tex = ''.join(PRC_tex)

    if type(label_type) is list:
        label_type = ''.join(label_type)

    # Read the inputs
    prediction = pd.read_hdf(prediction)
    if label_type is None:
        # Assume we want to have the first key
        label_type = prediction.keys()[0]
    N_1 = len(prediction[label_type].Y_train[0])
    N_2 = len(prediction[label_type].Y_test[0])

    # Determine the predicted score per patient
    print('Determining score per patient.')
    y_truths, y_scores, _, _ =\
        plot_estimator_performance(prediction, pinfo, [label_type],
                                   alpha=0.95, ensemble=ensemble,
                                   output='decision')

    # Check if we can compute confidence intervals
    config = prediction[label_type].config
    crossval_type = config['CrossValidation']['Type']

    # --------------------------------------------------------------
    # ROC Curve
    if crossval_type == 'LOO':
        print("LOO: Plotting the ROC without confidence intervals.")
        y_truths = [i[0] for i in y_truths]
        y_scores = [i[0] for i in y_scores]
        fpr, tpr, _, f = plot_single_ROC(y_truths, y_scores, returnplot=True)
    else:
        # Plot the ROC with confidence intervals
        print("Plotting the ROC with confidence intervals.")
        f, fpr, tpr = plot_ROC_CIc(y_truths, y_scores, N_1, N_2)

    # Save the outputs
    if ROC_png is not None:
        f.savefig(ROC_png)
        print(("ROC saved as {} !").format(ROC_png))

    if ROC_tex is not None:
        tikzplotlib.save(ROC_tex)
        print(("ROC saved as {} !").format(ROC_tex))

    if ROC_csv is not None:
        with open(ROC_csv, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['FPR', 'TPR'])
            for i in range(0, len(fpr)):
                data = [str(fpr[i]), str(tpr[i])]
                writer.writerow(data)

        print(("ROC saved as {} !").format(ROC_csv))

    # --------------------------------------------------------------
    # PR Curve
    if crossval_type == 'LOO':
        tpr, precisionr, _, f = plot_single_PRC(y_truths, y_scores, returnplot=True)
    else:
        f, tpr, precisionr = plot_PRC_CIc(y_truths, y_scores, N_1, N_2)

    if PRC_png is not None:
        f.savefig(PRC_png)
        print(("PRC saved as {} !").format(PRC_png))

    if PRC_tex is not None:
        tikzplotlib.save(PRC_tex)
        print(("PRC saved as {} !").format(PRC_tex))

    if PRC_csv is not None:
        with open(PRC_csv, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Recall', 'Precision'])
            for i in range(0, len(tpr)):
                data = [str(tpr[i]), str(precisionr[i])]
                writer.writerow(data)

        print(("PRC saved as {} !").format(PRC_csv))

    return f, fpr, tpr


if __name__ == '__main__':
    main()
