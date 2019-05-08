#!/usr/bin/env python

# Copyright 2016-2019 Biomedical Imaging Group Rotterdam, Departments of
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


import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import sys
import WORC.plotting.compute_CI
import pandas as pd
import os
import lifelines as ll
import WORC.processing.label_processing as lp
from scipy.stats import pearsonr, spearmanr
from WORC.classification.metrics import ICC


def plot_single_SVR(prediction, label_data, label_type, survival=False,
                    show_plots=False, alpha=0.95):
    if type(prediction) is not pd.core.frame.DataFrame:
        if os.path.isfile(prediction):
            prediction = pd.read_hdf(prediction)

    keys = prediction.keys()
    SVRs = list()
    label = keys[0]
    SVRs = prediction[label]['classifiers']

    Y_test = prediction[label]['Y_test']
    X_test = prediction[label]['X_test']
    Y_train = prediction[label]['X_train']

    if survival:
        # Also extract time to event and if event occurs from label data
        labels = [[label_type], ['E'], ['T']]
    else:
        labels = [[label_type]]

    if type(label_data) is not dict:
        if os.path.isfile(label_data):
            label_data = lp.load_labels(label_data, labels)

    patient_IDs = label_data['patient_IDs']
    labels = label_data['label']

    # Initialize scoring metrics
    r2score = list()
    MSE = list()
    coefICC = list()
    PearsonC = list()
    PearsonP = list()
    SpearmanC = list()
    SpearmanP = list()

    if survival:
        cindex = list()
        coxp = list()
        coxcoef = list()

    patient_MSE = dict()

    for i in range(0, len(Y_test)):
        test_patient_IDs = prediction[label]['patient_ID_test'][i]

        # FIXME: Put some wrong patient IDs in test files
        for num in range(0, len(test_patient_IDs)):
            if 'features_' in test_patient_IDs[num]:
                test_patient_IDs[num] = test_patient_IDs[num][9::]

            if '__tpl.hdf5' in test_patient_IDs[num]:
                test_patient_IDs[num] = test_patient_IDs[num][0:-10]

        test_patient_IDs = np.asarray(test_patient_IDs)

        X_temp = X_test[i]

        test_indices = list()
        for i_ID in test_patient_IDs:
            # FIXME: Error in specific study
            if i_ID == '112_recurrence-preop':
                i_ID = '112_recurrence_preop'
            test_indices.append(np.where(patient_IDs == i_ID)[0][0])

        y_truth = [labels[0][k][0] for k in test_indices]

        if type(SVRs) == list or type(SVRs) == tuple:
            estimator = SVRs[i]
        else:
            estimator = SVRs

        scaler = estimator.best_scaler
        try:
            y_prediction = estimator.predict(scaler.transform(X_temp))
        except ValueError:
            y_prediction = estimator.predict(X_temp)

        y_truth = np.asarray(y_truth)

        # if survival:
        #     # Normalize the scores
        #     y_prediction = np.subtract(1.01, np.divide(y_prediction, np.max(y_prediction)))

        print("Truth: " + y_truth)
        print("Prediction: " + y_prediction)

        # Compute error per patient
        for i_truth, i_predict, i_test_ID in zip(y_truth, y_prediction, test_patient_IDs):
            if i_test_ID not in patient_MSE.keys():
                patient_MSE[i_test_ID] = list()
            patient_MSE[i_test_ID].append((i_truth - i_predict)**2)

        # Compute evaluation metrics
        r2score.append(r2_score(y_truth, y_prediction))
        MSE.append(mean_squared_error(y_truth, y_prediction))
        coefICC.append(ICC(np.column_stack((y_prediction, y_truth))))
        C = pearsonr(y_prediction, y_truth)
        PearsonC.append(C[0])
        PearsonP.append(C[1])
        C = spearmanr(y_prediction, y_truth)
        SpearmanC.append(C.correlation)
        SpearmanP.append(C.pvalue)

        if survival:
            # Extract time to event and event from label data
            E_truth = np.asarray([labels[1][k][0] for k in test_indices])
            T_truth = np.asarray([labels[2][k][0] for k in test_indices])

            # Concordance index
            cindex.append(1 - ll.utils.concordance_index(T_truth, y_prediction, E_truth))

            # Fit Cox model using SVR output, time to event and event
            data = {'predict': y_prediction, 'E': E_truth, 'T': T_truth}
            data = pd.DataFrame(data=data, index=test_patient_IDs)

            cph = ll.CoxPHFitter()
            cph.fit(data, duration_col='T', event_col='E')

            coxcoef.append(cph.summary['coef']['predict'])
            coxp.append(cph.summary['p']['predict'])

    # Compute confidence intervals for given metrics
    N_1 = float(len(Y_train[0]))
    N_2 = float(len(Y_test[0]))

    if len(r2score) == 1:
        # No confidence intevals, just take the scores
        stats = dict()
        stats["r2_score:"] = str(r2score[0])
        stats["MSE:"] = str(MSE[0])
        stats["ICC:"] = str(coefICC[0])
        stats["PearsonC:"] = str(PearsonC[0])
        stats["SpearmanC: "] = str(SpearmanC[0])
        stats["PearsonP:"] = str(PearsonP[0])
        stats["SpearmanP: "] = str(SpearmanP[0])

        if survival:
            stats["Concordance:"] = str(cindex[0])
            stats["Cox coef.:"] = str(coxcoef[0])
            stats["Cox p:"] = str(coxp[0])
    else:
        # Compute confidence intervals from cross validations
        stats = dict()
        stats["r2_score 95%:"] = str(compute_CI.compute_confidence(r2score, N_1, N_2, alpha))
        stats["MSE 95%:"] = str(compute_CI.compute_confidence(MSE, N_1, N_2, alpha))
        stats["ICC 95%:"] = str(compute_CI.compute_confidence(coefICC, N_1, N_2, alpha))
        stats["PearsonC 95%:"] = str(compute_CI.compute_confidence(PearsonC, N_1, N_2, alpha))
        stats["SpearmanC 95%: "] = str(compute_CI.compute_confidence(SpearmanC, N_1, N_2, alpha))
        stats["PearsonP 95%:"] = str(compute_CI.compute_confidence(PearsonP, N_1, N_2, alpha))
        stats["SpearmanP 95%: "] = str(compute_CI.compute_confidence(SpearmanP, N_1, N_2, alpha))

        if survival:
            stats["Concordance 95%:"] = str(compute_CI.compute_confidence(cindex, N_1, N_2, alpha))
            stats["Cox coef. 95%:"] = str(compute_CI.compute_confidence(coxcoef, N_1, N_2, alpha))
            stats["Cox p 95%:"] = str(compute_CI.compute_confidence(coxp, N_1, N_2, alpha))

    for k, v in stats.iteritems():
        print(k, v)

    # Calculate and sort individual patient MSE
    patient_MSE = {k: np.mean(v) for k, v in patient_MSE.iteritems()}
    order = np.argsort(patient_MSE.values())
    sortedkeys = np.asarray(patient_MSE.keys())[order].tolist()
    sortedvalues = np.asarray(patient_MSE.values())[order].tolist()
    patient_MSE = [(k, v) for k, v in zip(sortedkeys, sortedvalues)]

    for p in patient_MSE:
        print(p[0], p[1])

    stats["Patient_MSE"] = patient_MSE

    if show_plots:
        # TODO: Plot metrics, see also plot_SVM
        pass

    return stats


def main():
    if len(sys.argv) == 1:
        prediction = '/media/martijn/DATA/CLM_MICCAI/Results/classification_all_SVR.hdf5'
        label_data = '/home/martijn/git/RadTools/CLM_MICCAI/pinfo_CLM_MICCAI_test_months.txt'
        label_type = 'KM'
        survival = True
    elif len(sys.argv) != 3:
        raise IOError("This function accepts two arguments")
    else:
        prediction = sys.argv[1]
        label_data = sys.argv[2]
    plot_single_SVR(prediction, label_data, label_type, survival)


if __name__ == '__main__':
    main()
