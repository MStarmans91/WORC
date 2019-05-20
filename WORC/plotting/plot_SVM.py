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
import sys
import WORC.plotting.compute_CI as compute_CI
import pandas as pd
import os
import lifelines as ll
import WORC.processing.label_processing as lp
from WORC.classification import metrics
import WORC.addexceptions as ae
from sklearn.base import is_regressor


def plot_SVM(prediction, label_data, label_type, show_plots=False,
             alpha=0.95, ensemble=False, verbose=True,
             ensemble_scoring=None, output='stats',
             modus='singlelabel',
             thresholds=None, survival=False):
    '''
    Plot the output of a single binary estimator, e.g. a SVM.

    Parameters
    ----------
    prediction: pandas dataframe or string, mandatory
        output of trainclassifier function, either a pandas dataframe
        or a HDF5 file

    label_data: string, mandatory
        Contains the path referring to a .txt file containing the
        patient label(s) and value(s) to be used for learning. See
        the Github Wiki for the format.

    label_type: string, mandatory
        Name of the label to extract from the label data to test the
        estimator on.

    show_plots: Boolean, default False
        Determine whether matplotlib performance plots are made.

    alpha: float, default 0.95
        Significance of confidence intervals.

    ensemble: False, integer or 'Caruana'
        Determine whether an ensemble will be created. If so,
        either provide an integer to determine how many of the
        top performing classifiers should be in the ensemble, or use
        the string "Caruana" to use smart ensembling based on
        Caruana et al. 2004.

    verbose: boolean, default True
        Plot intermedate messages.

    ensemble_scoring: string, default None
        Metric to be used for evaluating the ensemble. If None,
        the option set in the prediction object will be used.

    output: string, default stats
        Determine which results are put out. If stats, the statistics of the
        estimator will be returned. If scores, the scores will be returned.

    thresholds: list of integer(s), default None
        If None, use default threshold of sklearn (0.5) on posteriors to
        converge to a binary prediction. If one integer is provided, use that one.
        If two integers are provided, posterior < thresh[0] = 0, posterior > thresh[1] = 1.

    Returns
    ----------
    Depending on the output parameters, the following outputs are returned:

    If output == 'stats':
    stats: dictionary
        Contains the confidence intervals of the performance metrics
        and the number of times each patient was classifier correctly
        or incorrectly.

    If output == 'scores':
    y_truths: list
        Contains the true label for each object.

    y_scores: list
        Contains the score (e.g. posterior) for each object.

    y_predictions: list
        Contains the predicted label for each object.

    PIDs: list
        Contains the patient ID/name for each object.
    '''

    # Load the prediction object if it's a hdf5 file
    if type(prediction) is not pd.core.frame.DataFrame:
        if os.path.isfile(prediction):
            prediction = pd.read_hdf(prediction)
        else:
            raise ae.WORCIOError(('{} is not an existing file!').format(str(prediction)))

    # Select the estimator from the pandas dataframe to use
    keys = prediction.keys()
    SVMs = list()
    if label_type is None:
        label_type = keys[0]

    # Load the label data
    if type(label_data) is not dict:
        if os.path.isfile(label_data):
            if type(label_type) is not list:
                # Singlelabel: convert to list
                label_type = [[label_type]]
            label_data = lp.load_labels(label_data, label_type)

    patient_IDs = label_data['patient_IDs']
    labels = label_data['label']

    if type(label_type) is list:
        # FIXME: Support for multiple label types not supported yet.
        print('[WORC Warning] Support for multiple label types not supported yet. Taking first label for plot_SVM.')
        label_type = keys[0]

    # Extract the estimators, features and labels
    SVMs = prediction[label_type]['classifiers']
    regression = is_regressor(SVMs[0].best_estimator_)
    Y_test = prediction[label_type]['Y_test']
    X_test = prediction[label_type]['X_test']
    X_train = prediction[label_type]['X_train']
    Y_train = prediction[label_type]['Y_train']
    feature_labels = prediction[label_type]['feature_labels']

    # Create lists for performance measures
    if not regression:
        sensitivity = list()
        specificity = list()
        precision = list()
        npv = list()
        accuracy = list()
        auc = list()
        f1_score_list = list()
    else:
        r2score = list()
        MSE = list()
        coefICC = list()
        PearsonC = list()
        PearsonP = list()
        SpearmanC = list()
        SpearmanP = list()
        cindex = list()
        coxcoef = list()
        coxp = list()

    patient_classification_list = dict()
    percentages_selected = list()

    if output in ['scores', 'decision']:
        # Keep track of all groundth truths and scores
        y_truths = list()
        y_scores = list()
        y_predictions = list()
        PIDs = list()

    # Loop over the test sets, which probably correspond with cross validation
    # iterations
    for i in range(0, len(Y_test)):
        print("\n")
        print(("Cross validation {} / {}.").format(str(i + 1), str(len(Y_test))))
        test_patient_IDs = prediction[label_type]['patient_ID_test'][i]
        train_patient_IDs = prediction[label_type]['patient_ID_train'][i]
        X_test_temp = X_test[i]
        X_train_temp = X_train[i]
        Y_train_temp = Y_train[i]
        Y_test_temp = Y_test[i]
        test_indices = list()

        # Check which patients are in the test set.
        for i_ID in test_patient_IDs:
            test_indices.append(np.where(patient_IDs == i_ID)[0][0])

            # Initiate counting how many times a patient is classified correctly
            if i_ID not in patient_classification_list:
                patient_classification_list[i_ID] = dict()
                patient_classification_list[i_ID]['N_test'] = 0
                patient_classification_list[i_ID]['N_correct'] = 0
                patient_classification_list[i_ID]['N_wrong'] = 0

            patient_classification_list[i_ID]['N_test'] += 1

        # Extract ground truth
        y_truth = Y_test_temp

        # If requested, first let the SearchCV object create an ensemble
        if ensemble:
            # NOTE: Added for backwards compatability
            if not hasattr(SVMs[i], 'cv_iter'):
                cv_iter = list(SVMs[i].cv.split(X_train_temp, Y_train_temp))
                SVMs[i].cv_iter = cv_iter

            # Create the ensemble
            X_train_temp = [(x, feature_labels) for x in X_train_temp]
            SVMs[i].create_ensemble(X_train_temp, Y_train_temp,
                                    method=ensemble, verbose=verbose,
                                    scoring=ensemble_scoring)

        # Create prediction
        y_prediction = SVMs[i].predict(X_test_temp)

        if regression:
            y_score = y_prediction
        else:
            y_score = SVMs[i].predict_proba(X_test_temp)[:, 1]

        # Create a new binary score based on the thresholds if given
        if thresholds is not None:
            if len(thresholds) == 1:
                y_prediction = y_score >= thresholds[0]
            elif len(thresholds) == 2:
                y_score_temp = list()
                y_prediction_temp = list()
                y_truth_temp = list()
                test_patient_IDs_temp = list()
                for pnum in range(len(y_score)):
                    if y_score[pnum] <= thresholds[0] or y_score[pnum] > thresholds[1]:
                        y_score_temp.append(y_score[pnum])
                        y_prediction_temp.append(y_prediction[pnum])
                        y_truth_temp.append(y_truth[pnum])
                        test_patient_IDs_temp.append(test_patient_IDs[pnum])

                perc = float(len(y_prediction_temp))/float(len(y_prediction))
                percentages_selected.append(perc)
                print(f"Selected {len(y_prediction_temp)} from {len(y_prediction)} ({perc}%) patients using two thresholds.")
                y_score = y_score_temp
                y_prediction = y_prediction_temp
                y_truth = y_truth_temp
                test_patient_IDs = test_patient_IDs_temp
            else:
                raise ae.WORCValueError(f"Need None, one or two thresholds on the posterior; got {len(thresholds)}.")

        print("Truth: " + str(y_truth))
        print("Prediction: " + str(y_prediction))

        # Add if patient was classified correctly or not to counting
        for i_truth, i_predict, i_test_ID in zip(y_truth, y_prediction, test_patient_IDs):
            if modus == 'multilabel':
                success = (i_truth == i_predict).all()
            else:
                success = i_truth == i_predict

            if success:
                patient_classification_list[i_test_ID]['N_correct'] += 1
            else:
                patient_classification_list[i_test_ID]['N_wrong'] += 1

        if output == 'decision':
            # Output the posteriors
            y_scores.append(y_score)
            y_truths.append(y_truth)
            y_predictions.append(y_prediction)
            PIDs.append(test_patient_IDs)

        elif output == 'scores':
            # Output the posteriors
            y_scores.append(y_score)
            y_truths.append(y_truth)
            y_predictions.append(y_prediction)
            PIDs.append(test_patient_IDs)

        elif output == 'stats':
            # Compute statistics
            # Compute confusion matrix and use for sensitivity/specificity
            if modus == 'singlelabel':
                # Compute singlelabel performance metrics
                if not regression:
                    accuracy_temp, sensitivity_temp, specificity_temp,\
                        precision_temp, npv_temp, f1_score_temp, auc_temp =\
                        metrics.performance_singlelabel(y_truth,
                                                        y_prediction,
                                                        y_score,
                                                        regression)
                else:
                    r2score_temp, MSE_temp, coefICC_temp, PearsonC_temp,\
                        PearsonP_temp, SpearmanC_temp,\
                        SpearmanP_temp =\
                        metrics.performance_singlelabel(y_truth,
                                                        y_prediction,
                                                        y_score,
                                                        regression)

            elif modus == 'multilabel':
                # Convert class objects to single label per patient
                y_truth_temp = list()
                y_prediction_temp = list()
                for yt, yp in zip(y_truth, y_prediction):
                    label = np.where(yt == 1)
                    if len(label) > 1:
                        raise ae.WORCNotImplementedError('Multiclass classification evaluation is not supported in WORC.')

                    y_truth_temp.append(label[0][0])
                    label = np.where(yp == 1)
                    y_prediction_temp.append(label[0][0])

                y_truth = y_truth_temp
                y_prediction = y_prediction_temp

                # Compute multilabel performance metrics
                accuracy_temp, sensitivity_temp, specificity_temp,\
                    precision_temp, npv_temp, f1_score_temp, auc_temp =\
                    metrics.performance_multilabel(y_truth,
                                                   y_prediction,
                                                   y_score)

            else:
                raise ae.WORCKeyError('{} is not a valid modus!').format(modus)

            # Print AUC to keep you up to date
            if not regression:
                print('AUC: ' + str(auc_temp))

                # Append performance to lists for all cross validations
                accuracy.append(accuracy_temp)
                sensitivity.append(sensitivity_temp)
                specificity.append(specificity_temp)
                auc.append(auc_temp)
                f1_score_list.append(f1_score_temp)
                precision.append(precision_temp)
                npv.append(npv_temp)
            else:
                print('R2 Score: ' + str(r2score_temp))

                r2score.append(r2score_temp)
                MSE.append(MSE_temp)
                coefICC.append(coefICC_temp)
                PearsonC.append(PearsonC_temp)
                PearsonP.append(PearsonP_temp)
                SpearmanC.append(SpearmanC_temp)
                SpearmanP.append(SpearmanP_temp)

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

    if output in ['scores', 'decision']:
        # Return the scores and true values of all patients
        return y_truths, y_scores, y_predictions, PIDs
    elif output == 'stats':
        # Compute statistics
        # Extract sample size
        N_1 = float(len(train_patient_IDs))
        N_2 = float(len(test_patient_IDs))

        # Compute alpha confidence intervals (CIs)
        stats = dict()
        if not regression:
            stats["Accuracy 95%:"] = str(compute_CI.compute_confidence(accuracy, N_1, N_2, alpha))
            stats["AUC 95%:"] = str(compute_CI.compute_confidence(auc, N_1, N_2, alpha))
            stats["F1-score 95%:"] = str(compute_CI.compute_confidence(f1_score_list, N_1, N_2, alpha))
            stats["Precision 95%:"] = str(compute_CI.compute_confidence(precision, N_1, N_2, alpha))
            stats["NPV 95%:"] = str(compute_CI.compute_confidence(npv, N_1, N_2, alpha))
            stats["Sensitivity 95%: "] = str(compute_CI.compute_confidence(sensitivity, N_1, N_2, alpha))
            stats["Specificity 95%:"] = str(compute_CI.compute_confidence(specificity, N_1, N_2, alpha))

            if thresholds is not None:
                if len(thresholds) == 2:
                    # Compute percentage of patients that was selected
                    stats["Percentage Selected 95%:"] = str(compute_CI.compute_confidence(percentages_selected, N_1, N_2, alpha))

            # Extract statistics on how often patients got classified correctly
            alwaysright = dict()
            alwayswrong = dict()
            percentages = dict()
            for i_ID in patient_classification_list:
                percentage_right = patient_classification_list[i_ID]['N_correct'] / float(patient_classification_list[i_ID]['N_test'])

                if i_ID in patient_IDs:
                    label = labels[0][np.where(i_ID == patient_IDs)]
                else:
                    # Multiple instance of one patient
                    label = labels[0][np.where(i_ID.split('_')[0] == patient_IDs)]

                label = label[0][0]
                percentages[i_ID] = str(label) + ': ' + str(round(percentage_right, 2) * 100) + '%'
                if percentage_right == 1.0:
                    alwaysright[i_ID] = label
                    print(("Always Right: {}, label {}").format(i_ID, label))

                elif percentage_right == 0:
                    alwayswrong[i_ID] = label
                    print(("Always Wrong: {}, label {}").format(i_ID, label))

            stats["Always right"] = alwaysright
            stats["Always wrong"] = alwayswrong
            stats['Percentages'] = percentages
        else:
            # Regression
            stats['R2-score 95%: '] = str(compute_CI.compute_confidence(r2score, N_1, N_2, alpha))
            stats['MSE 95%: '] = str(compute_CI.compute_confidence(MSE, N_1, N_2, alpha))
            stats['ICC 95%: '] = str(compute_CI.compute_confidence(coefICC, N_1, N_2, alpha))
            stats['PearsonC 95%: '] = str(compute_CI.compute_confidence(PearsonC, N_1, N_2, alpha))
            stats['PearsonP 95%: '] = str(compute_CI.compute_confidence(PearsonP, N_1, N_2, alpha))
            stats['SpearmanC 95%: '] = str(compute_CI.compute_confidence(SpearmanC, N_1, N_2, alpha))
            stats['SpearmanP 95%: '] = str(compute_CI.compute_confidence(SpearmanP, N_1, N_2, alpha))

            if survival:
                stats["Concordance 95%:"] = str(compute_CI.compute_confidence(cindex, N_1, N_2, alpha))
                stats["Cox coef. 95%:"] = str(compute_CI.compute_confidence(coxcoef, N_1, N_2, alpha))
                stats["Cox p 95%:"] = str(compute_CI.compute_confidence(coxp, N_1, N_2, alpha))

        # Print all CI's
        for k, v in stats.items():
            print(f"{k} : {v}.")

        if show_plots:
            # Plot some characteristics in boxplots
            import matplotlib.pyplot as plt

            plt.figure()
            plt.boxplot(accuracy)
            plt.ylim([-0.05, 1.05])
            plt.ylabel('Accuracy')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')  # labels along the bottom edge are off
            plt.tight_layout()
            plt.show()

            plt.figure()
            plt.boxplot(auc)
            plt.ylim([-0.05, 1.05])
            plt.ylabel('AUC')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')  # labels along the bottom edge are off
            plt.tight_layout()
            plt.show()

            plt.figure()
            plt.boxplot(precision)
            plt.ylim([-0.05, 1.05])
            plt.ylabel('Precision')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')  # labels along the bottom edge are off
            plt.tight_layout()
            plt.show()

            plt.figure()
            plt.boxplot(npv)
            plt.ylim([-0.05, 1.05])
            plt.ylabel('NPV')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')  # labels along the bottom edge are off
            plt.tight_layout()
            plt.show()

            plt.figure()
            plt.boxplot(sensitivity)
            plt.ylim([-0.05, 1.05])
            plt.ylabel('Sensitivity')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')  # labels along the bottom edge are off
            plt.tight_layout()
            plt.show()

            plt.figure()
            plt.boxplot(specificity)
            plt.ylim([-0.05, 1.05])
            plt.ylabel('Specificity')
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')  # labels along the bottom edge are off
            plt.tight_layout()
            plt.show()

        return stats


def main():
    if len(sys.argv) == 1:
        print("TODO: Put in an example")
    elif len(sys.argv) != 4:
        raise IOError("This function accepts three arguments!")
    else:
        prediction = sys.argv[1]
        patientinfo = sys.argv[2]
        label_type = sys.argv[3]
    plot_SVM(prediction, patientinfo, label_type)


if __name__ == '__main__':
    main()
