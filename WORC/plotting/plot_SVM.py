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


import numpy as np
import sys
from WORC.plotting.compute_CI import compute_confidence
from WORC.plotting.compute_CI import compute_confidence_bootstrap
import pandas as pd
import os
import lifelines as ll
import WORC.processing.label_processing as lp
from WORC.classification import metrics
import WORC.addexceptions as ae
from sklearn.base import is_regressor
from collections import OrderedDict
from sklearn.utils import resample


def fit_thresholds(thresholds, estimator, X_train, Y_train, ensemble, ensemble_scoring):
    print('Fitting thresholds on validation set')
    if not hasattr(estimator, 'cv_iter'):
        cv_iter = list(estimator.cv.split(X_train, Y_train))
        estimator.cv_iter = cv_iter

    p_est = estimator.cv_results_['params'][0]
    p_all = estimator.cv_results_['params_all'][0]
    n_iter = len(estimator.cv_iter)

    thresholds_low = list()
    thresholds_high = list()
    for it, (train, valid) in enumerate(estimator.cv_iter):
        print(' - iteration {it + 1} / {n_iter}.')
        # NOTE: Explicitly exclude validation set, elso refit and score
        # somehow still seems to use it.
        X_train_temp = [prediction[label_type]['X_train'][i] for i in train]
        Y_train_temp = [prediction[label_type]['Y_train'][i] for i in train]
        train_temp = range(0, len(train))

        # Refit a SearchCV object with the provided parameters
        if ensemble:
            estimator.create_ensemble(X_train_temp, Y_train_temp,
                                      method=ensemble, verbose=False,
                                      scoring=ensemble_scoring)
        else:
            estimator.refit_and_score(X_train_temp, Y_train_temp, p_all,
                                      p_est, train_temp, train_temp,
                                      verbose=False)

        # Predict and save scores
        X_train_values = [x[0] for x in X_train] # Throw away labels
        X_train_values_valid = [X_train_values[i] for i in valid]
        Y_valid_score_temp = estimator.predict_proba(X_train_values_valid)

        # Only take the probabilities for the second class
        Y_valid_score_temp = Y_valid_score_temp[:, 1]

        # Select thresholds
        thresholds_low.append(np.percentile(Y_valid_score_temp, thresholds[0]*100.0))
        thresholds_high.append(np.percentile(Y_valid_score_temp, thresholds[1]*100.0))

    thresholds_val = [np.mean(thresholds_low), np.mean(thresholds_high)]
    print(f'Thresholds {thresholds} converted to {thresholds_val}.')
    return thresholds_val


def plot_SVM(prediction, label_data, label_type, show_plots=False,
             alpha=0.95, ensemble=False, verbose=True,
             ensemble_scoring=None, output='stats',
             modus='singlelabel',
             thresholds=None, survival=False,
             generalization=False, shuffle_estimators=False,
             bootstrap=False, bootstrap_N=1000,
             overfit_scaler=False):
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

    pids: list
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
    if label_type is None:
        label_type = keys[0]

    # Load the label data
    if type(label_data) is not dict:
        if os.path.isfile(label_data):
            if type(label_type) is not list:
                # Singlelabel: convert to list
                label_type = [[label_type]]
            label_data = lp.load_labels(label_data, label_type)
        else:
            raise ae.WORCValueError(f"Label data {label_data} incorrect: not a dictionary, or file does not exist.")

    n_labels = len(label_type)
    patient_IDs = label_data['patient_IDs']
    labels = label_data['label']

    if type(label_type) is list:
        # FIXME: Support for multiple label types not supported yet.
        print('[WORC Warning] Support for multiple label types not supported yet. Taking first label for plot_SVM.')
        original_label_type = label_type[:]
        label_type = keys[0]

    # Extract the estimators, features and labels
    regression = is_regressor(prediction[label_type]['classifiers'][0].best_estimator_)
    feature_labels = prediction[label_type]['feature_labels']

    # Create lists for performance measures
    if not regression:
        sensitivity = list()
        specificity = list()
        precision = list()
        npv = list()
        accuracy = list()
        bca = list()
        auc = list()
        f1_score_list = list()

        if modus == 'multilabel':
            acc_av = list()

            # Also add scoring measures for all single label scores
            sensitivity_single = list()
            specificity_single = list()
            precision_single = list()
            npv_single = list()
            accuracy_single = list()
            bca_single = list()
            auc_single = list()
            f1_score_list_single = list()

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
        pids = list()

    # Extract sample size
    N_1 = float(len(prediction[label_type]['patient_ID_train'][0]))
    N_2 = float(len(prediction[label_type]['patient_ID_test'][0]))

    # Convert tuples to lists if required
    if type(prediction[label_type]['X_test']) is tuple:
        prediction[label_type]['X_test'] = list(prediction[label_type]['X_test'])
        prediction[label_type]['X_train'] = list(prediction[label_type]['X_train'])
        prediction[label_type]['Y_train'] = list(prediction[label_type]['Y_train'])
        prediction[label_type]['Y_test'] = list(prediction[label_type]['Y_test'])
        prediction[label_type]['patient_ID_test'] = list(prediction[label_type]['patient_ID_test'])
        prediction[label_type]['patient_ID_train'] = list(prediction[label_type]['patient_ID_train'])
        prediction[label_type]['classifiers'] = list(prediction[label_type]['classifiers'])

    # Loop over the test sets, which correspond to cross-validation
    # or bootstrapping iterations
    n_iter = len(prediction[label_type]['Y_test'])
    if bootstrap:
        iterobject = range(0, bootstrap_N)
    else:
        iterobject = range(0, n_iter)

    for i in iterobject:
        print("\n")
        if bootstrap:
            print(f"Bootstrap {i + 1} / {bootstrap_N}.")
        else:
            print(f"Cross-validation {i + 1} / {n_iter}.")

        test_indices = list()

        # When bootstrapping, there is only a single train/test set.
        if bootstrap:
            X_test_temp = prediction[label_type]['X_test'][0]
            X_train_temp = prediction[label_type]['X_train'][0]
            Y_train_temp = prediction[label_type]['Y_train'][0]
            Y_test_temp = prediction[label_type]['Y_test'][0]
            test_patient_IDs = prediction[label_type]['patient_ID_test'][0]
            train_patient_IDs = prediction[label_type]['patient_ID_train'][0]
            fitted_model = prediction[label_type]['classifiers'][0]
        else:
            X_test_temp = prediction[label_type]['X_test'][i]
            X_train_temp = prediction[label_type]['X_train'][i]
            Y_train_temp = prediction[label_type]['Y_train'][i]
            Y_test_temp = prediction[label_type]['Y_test'][i]
            test_patient_IDs = prediction[label_type]['patient_ID_test'][i]
            train_patient_IDs = prediction[label_type]['patient_ID_train'][i]
            fitted_model = prediction[label_type]['classifiers'][i]

        # If bootstrap, generate a bootstrapped sample
        if bootstrap:
            X_test_temp, Y_test_temp, test_patient_IDs = resample(X_test_temp, Y_test_temp, test_patient_IDs)

        # Check which patients are in the test set.
        for i_ID in test_patient_IDs:
            # Initiate counting how many times a patient is classified correctly
            if i_ID not in patient_classification_list:
                patient_classification_list[i_ID] = dict()
                patient_classification_list[i_ID]['N_test'] = 0
                patient_classification_list[i_ID]['N_correct'] = 0
                patient_classification_list[i_ID]['N_wrong'] = 0

            patient_classification_list[i_ID]['N_test'] += 1

            # Check if this is exactly the label of the patient within the label file
            if i_ID not in patient_IDs:
                print(f'[WORC WARNING] Patient {i_ID} is not found the patient labels, removing underscore.')
                i_ID = i_ID.split("_")[0]
                if i_ID not in patient_IDs:
                    print(f'[WORC WARNING] Did not help, excluding patient {i_ID}.')
                    continue

            test_indices.append(np.where(patient_IDs == i_ID)[0][0])

        # Extract ground truth
        y_truth = Y_test_temp

        # If required, shuffle estimators for "Random" ensembling
        if shuffle_estimators:
            # Compute generalization score
            print('Shuffling estimators for random ensembling.')
            shuffle(fitted_model.cv_results_['params'])
            shuffle(fitted_model.cv_results_['params_all'])

        # If required, rank according to generalization score instead of mean_validation_score
        if generalization:
            # Compute generalization score
            print('Using generalization score for estimator ranking.')
            difference_score = abs(fitted_model.cv_results_['mean_train_score'] - fitted_model.cv_results_['mean_test_score'])
            generalization_score = fitted_model.cv_results_['mean_test_score'] - difference_score

            # Rerank based on score
            indices = np.argsort(generalization_score)
            fitted_model.cv_results_['params'] = [fitted_model.cv_results_['params'][i] for i in indices[::-1]]
            fitted_model.cv_results_['params_all'] = [fitted_model.cv_results_['params_all'][i] for i in indices[::-1]]

        # If requested, first let the SearchCV object create an ensemble
        if bootstrap and i > 0:
            # For bootstrapping, only do this at the first iteration
            pass
        elif not fitted_model.ensemble:
            # NOTE: Added for backwards compatability
            if not hasattr(fitted_model, 'cv_iter'):
                cv_iter = list(fitted_model.cv.split(X_train_temp, Y_train_temp))
                fitted_model.cv_iter = cv_iter

            # Create the ensemble
            X_train_temp = [(x, feature_labels) for x in X_train_temp]
            fitted_model.create_ensemble(X_train_temp, Y_train_temp,
                                         method=ensemble, verbose=verbose,
                                         scoring=ensemble_scoring,
                                         overfit_scaler=overfit_scaler)

        # Create prediction
        y_prediction = fitted_model.predict(X_test_temp)

        if regression:
            y_score = y_prediction
        elif modus == 'multilabel':
            y_score = fitted_model.predict_proba(X_test_temp)
        else:
            y_score = fitted_model.predict_proba(X_test_temp)[:, 1]

        # Create a new binary score based on the thresholds if given
        if thresholds is not None:
            if len(thresholds) == 1:
                y_prediction = y_score >= thresholds[0]
            elif len(thresholds) == 2:
                # X_train_temp = [x[0] for x in X_train_temp]

                y_score_temp = list()
                y_prediction_temp = list()
                y_truth_temp = list()
                test_patient_IDs_temp = list()

                thresholds_val = fit_thresholds(thresholds, fitted_model, X_train_temp, Y_train_temp, ensemble,
                                                ensemble_scoring)
                for pnum in range(len(y_score)):
                    if y_score[pnum] <= thresholds_val[0] or y_score[pnum] > thresholds_val[1]:
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
            pids.append(test_patient_IDs)

        elif output == 'scores':
            # Output the posteriors
            y_scores.append(y_score)
            y_truths.append(y_truth)
            y_predictions.append(y_prediction)
            pids.append(test_patient_IDs)

        elif output == 'stats':
            # Compute statistics
            # Compute confusion matrix and use for sensitivity/specificity
            if modus == 'singlelabel':
                # Compute singlelabel performance metrics
                if not regression:
                    accuracy_temp, bca_temp, sensitivity_temp,\
                        specificity_temp,\
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
                    precision_temp, npv_temp, f1_score_temp, auc_temp, acc_av_temp =\
                    metrics.performance_multilabel(y_truth,
                                                   y_prediction,
                                                   y_score)

                # Compute all single label performance metrics as well
                for i_label in range(n_labels):
                    y_truth_single = [i == i_label for i in y_truth]
                    y_prediction_single = [i == i_label for i in y_prediction]
                    y_score_single = y_score[:, i_label]

                    accuracy_temp_single, bca_temp_single, sensitivity_temp_single, specificity_temp_single,\
                        precision_temp_single, npv_temp_single, f1_score_temp_single, auc_temp_single =\
                        metrics.performance_singlelabel(y_truth_single,
                                                        y_prediction_single,
                                                        y_score_single,
                                                        regression)

            else:
                raise ae.WORCKeyError('{modus} is not a valid modus!')

            # Print AUC to keep you up to date
            if not regression:
                print('AUC: ' + str(auc_temp))

                # Append performance to lists for all cross validations
                accuracy.append(accuracy_temp)
                bca.append(bca_temp)
                sensitivity.append(sensitivity_temp)
                specificity.append(specificity_temp)
                auc.append(auc_temp)
                f1_score_list.append(f1_score_temp)
                precision.append(precision_temp)
                npv.append(npv_temp)

                if modus == 'multilabel':
                    acc_av.append(acc_av_temp)

                    accuracy_single.append(accuracy_temp_single)
                    bca_single.append(bca_temp_single)
                    sensitivity_single.append(sensitivity_temp_single)
                    specificity_single.append(specificity_temp_single)
                    auc_single.append(auc_temp_single)
                    f1_score_list_single.append(f1_score_temp_single)
                    precision_single.append(precision_temp_single)
                    npv_single.append(npv_temp_single)

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

        # Delete some objects to save memory in cross-validtion
        if not bootstrap:
            del fitted_model, X_test_temp, X_train_temp, Y_train_temp
            del Y_test_temp, test_patient_IDs, train_patient_IDs
            prediction[label_type]['X_test'][i] = None
            prediction[label_type]['X_train'][i] = None
            prediction[label_type]['Y_train'][i] = None
            prediction[label_type]['Y_test'][i] = None
            prediction[label_type]['patient_ID_test'][i] = None
            prediction[label_type]['patient_ID_train'][i] = None
            prediction[label_type]['classifiers'][i] = None

    if output in ['scores', 'decision']:
        # Return the scores and true values of all patients
        return y_truths, y_scores, y_predictions, pids
    elif output == 'stats':
        # Compute statistics

        # Compute alpha confidence intervals (CIs)
        stats = dict()
        if not regression:
            if bootstrap:
                # Compute once for the real test set the performance
                X_test_temp = prediction[label_type]['X_test'][0]
                y_truth = prediction[label_type]['Y_test'][0]
                y_prediction = fitted_model.predict(X_test_temp)

                if regression:
                    y_score = y_prediction
                else:
                    y_score = fitted_model.predict_proba(X_test_temp)[:, 1]

                accuracy_test, bca_test, sensitivity_test, specificity_test,\
                    precision_test, npv_test, f1_score_test, auc_test =\
                    metrics.performance_singlelabel(y_truth,
                                                    y_prediction,
                                                    y_score,
                                                    regression)

                stats["Accuracy 95%:"] = f"{accuracy_test} {str(compute_confidence_bootstrap(accuracy, accuracy_test, N_1, alpha))}"
                stats["BCA 95%:"] = f"{bca_test} {str(compute_confidence_bootstrap(bca, bca_test, N_1, alpha))}"
                stats["AUC 95%:"] = f"{auc_test} {str(compute_confidence_bootstrap(auc, auc_test, N_1, alpha))}"
                stats["F1-score 95%:"] = f"{f1_score_test} {str(compute_confidence_bootstrap(f1_score_list, f1_score_test, N_1, alpha))}"
                stats["Precision 95%:"] = f"{precision_test} {str(compute_confidence_bootstrap(precision, precision_test, N_1, alpha))}"
                stats["NPV 95%:"] = f"{npv_test} {str(compute_confidence_bootstrap(npv, npv_test, N_1, alpha))}"
                stats["Sensitivity 95%: "] = f"{sensitivity_test} {str(compute_confidence_bootstrap(sensitivity, sensitivity_test, N_1, alpha))}"
                stats["Specificity 95%:"] = f"{specificity_test} {str(compute_confidence_bootstrap(specificity, specificity_test, N_1, alpha))}"
            else:
                stats["Accuracy 95%:"] = f"{np.nanmean(accuracy)} {str(compute_confidence(accuracy, N_1, N_2, alpha))}"
                stats["BCA 95%:"] = f"{np.nanmean(bca)} {str(compute_confidence(bca, N_1, N_2, alpha))}"
                stats["AUC 95%:"] = f"{np.nanmean(auc)} {str(compute_confidence(auc, N_1, N_2, alpha))}"
                stats["F1-score 95%:"] = f"{np.nanmean(f1_score_list)} {str(compute_confidence(f1_score_list, N_1, N_2, alpha))}"
                stats["Precision 95%:"] = f"{np.nanmean(precision)} {str(compute_confidence(precision, N_1, N_2, alpha))}"
                stats["NPV 95%:"] = f"{np.nanmean(npv)} {str(compute_confidence(npv, N_1, N_2, alpha))}"
                stats["Sensitivity 95%: "] = f"{np.nanmean(sensitivity)} {str(compute_confidence(sensitivity, N_1, N_2, alpha))}"
                stats["Specificity 95%:"] = f"{np.nanmean(specificity)} {str(compute_confidence(specificity, N_1, N_2, alpha))}"

            if modus == 'multilabel':
                stats["Average Accuracy 95%:"] = f"{np.nanmean(acc_av)} {str(compute_confidence(acc_av, N_1, N_2, alpha))}"

            if thresholds is not None:
                if len(thresholds) == 2:
                    # Compute percentage of patients that was selected
                    stats["Percentage Selected 95%:"] = f"{np.nanmean(percentages_selected)} {str(compute_confidence(percentages_selected, N_1, N_2, alpha))}"

            # Extract statistics on how often patients got classified correctly
            rankings = dict()
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
                    print(f"Always Right: {i_ID}, label {label}.")

                elif percentage_right == 0:
                    alwayswrong[i_ID] = label
                    print(f"Always Wrong: {i_ID}, label {label}.")

            rankings["Always right"] = alwaysright
            rankings["Always wrong"] = alwayswrong
            rankings['Percentages'] = percentages
        else:
            # Regression
            stats['R2-score 95%: '] = f"{np.nanmean(r2_score)} {str(compute_confidence(r2score, N_1, N_2, alpha))}"
            stats['MSE 95%: '] = f"{np.nanmean(MSE)} {str(compute_confidence(MSE, N_1, N_2, alpha))}"
            stats['ICC 95%: '] = f"{np.nanmean(coefICC)} {str(compute_confidence(coefICC, N_1, N_2, alpha))}"
            stats['PearsonC 95%: '] = f"{np.nanmean(PearsonC)} {str(compute_confidence(PearsonC, N_1, N_2, alpha))}"
            stats['PearsonP 95%: '] = f"{np.nanmean(PearsonP)} {str(compute_confidence(PearsonP, N_1, N_2, alpha))}"
            stats['SpearmanC 95%: '] = f"{np.nanmean(SpearmanC)} {str(compute_confidence(SpearmanC, N_1, N_2, alpha))}"
            stats['SpearmanP 95%: '] = f"{np.nanmean(SpearmanP)} {str(compute_confidence(SpearmanP, N_1, N_2, alpha))}"

            if survival:
                stats["Concordance 95%:"] = f"{np.nanmean(cindex)} {str(compute_confidence(cindex, N_1, N_2, alpha))}"
                stats["Cox coef. 95%:"] = f"{np.nanmean(coxcoef)} {str(compute_confidence(coxcoef, N_1, N_2, alpha))}"
                stats["Cox p 95%:"] = f"{np.nanmean(coxp)} {str(compute_confidence(coxp, N_1, N_2, alpha))}"

        # Print all CI's
        stats = OrderedDict(sorted(stats.items()))
        for k, v in stats.items():
            print(f"{k} : {v}.")

        # Combine stats and rankings in one output
        output = dict()
        output['Statistics'] = stats
        output['Rankings'] = rankings

        return output


def combine_multiple_estimators(predictions, label_data, multilabel_type, label_types,
                                ensemble=1, strategy='argmax', alpha=0.95):
    '''
    Combine multiple estimators in a single model.

    Note: the multilabel_type labels should correspond to the ordering in label_types.
    Hence, if multilabel_type = 0, the prediction is label_type[0] etc.
    '''

    # Load the multilabel label data
    label_data = lp.load_labels(label_data, multilabel_type)
    patient_IDs = label_data['patient_IDs']
    labels = label_data['label']

    # Initialize some objects
    y_truths = list()
    y_scores = list()
    y_predictions = list()
    pids = list()

    y_truths_train = list()
    y_scores_train = list()
    y_predictions_train = list()
    pids_train = list()

    accuracy = list()
    sensitivity = list()
    specificity = list()
    auc = list()
    f1_score_list = list()
    precision = list()
    npv = list()
    acc_av = list()

    # Extract all the predictions from the estimators
    for prediction, label_type in zip(predictions, label_types):
        y_truth, y_score, y_prediction, pid,\
            y_truth_train, y_score_train, y_prediction_train, pid_train =\
            plot_SVM(prediction, label_data, label_type,
                     ensemble=ensemble, output='allscores')
        y_truths.append(y_truth)
        y_scores.append(y_score)
        y_predictions.append(y_prediction)
        pids.append(pid)

        y_truths_train.append(y_truth_train)
        y_scores_train.append(y_score_train)
        y_predictions_train.append(y_prediction_train)
        pids_train.append(pid_train)

    # Combine the predictions
    for i_crossval in range(0, len(y_truths[0])):
        # Extract all values for this cross validation iteration from all objects
        y_truth = [t[i_crossval] for t in y_truths]
        y_score = [t[i_crossval] for t in y_scores]
        pid = [t[i_crossval] for t in pids]

        if strategy == 'argmax':
            # For each patient, take the maximum posterior
            y_prediction = np.argmax(y_score, axis=0)
            y_score = np.max(y_score, axis=0)
        elif strategy == 'decisiontree':
            # Fit a decision tree on the training set
            a = 1
        else:
            raise ae.WORCValueError(f"{strategy} is not a valid estimation combining strategy! Should be one of [argmax].")

        # Compute multilabel performance metrics
        y_truth = np.argmax(y_truth, axis=0)
        accuracy_temp, sensitivity_temp, specificity_temp, \
            precision_temp, npv_temp, f1_score_temp, auc_temp, accav_temp = \
            metrics.performance_multilabel(y_truth,
                                           y_prediction,
                                           y_score)

        print("Truth: " + str(y_truth))
        print("Prediction: " + str(y_prediction))
        print('AUC: ' + str(auc_temp))

        # Append performance to lists for all cross validations
        accuracy.append(accuracy_temp)
        sensitivity.append(sensitivity_temp)
        specificity.append(specificity_temp)
        auc.append(auc_temp)
        f1_score_list.append(f1_score_temp)
        precision.append(precision_temp)
        npv.append(npv_temp)
        acc_av.append(acc_av_temp)

    # Extract sample size
    N_1 = float(len(train_patient_IDs))
    N_2 = float(len(test_patient_IDs))

    # Compute confidence intervals
    stats = dict()
    stats["Accuracy 95%:"] = f"{np.nanmean(accuracy)} {str(compute_confidence(accuracy, N_1, N_2, alpha))}"
    stats["Average Accuracy 95%:"] = f"{np.nanmean(acc_av)} {str(compute_confidence(accuracy, N_1, N_2, alpha))}"
    stats["AUC 95%:"] = f"{np.nanmean(auc)} {str(compute_confidence(auc, N_1, N_2, alpha))}"
    stats["F1-score 95%:"] = f"{np.nanmean(f1_score_list)} {str(compute_confidence(f1_score_list, N_1, N_2, alpha))}"
    stats["Precision 95%:"] = f"{np.nanmean(precision)} {str(compute_confidence(precision, N_1, N_2, alpha))}"
    stats["NPV 95%:"] = f"{np.nanmean(npv)} {str(compute_confidence(npv, N_1, N_2, alpha))}"
    stats["Sensitivity 95%: "] = f"{np.nanmean(sensitivity)} {str(compute_confidence(sensitivity, N_1, N_2, alpha))}"
    stats["Specificity 95%:"] = f"{np.nanmean(specificity)} {str(compute_confidence(specificity, N_1, N_2, alpha))}"

    # Print all CI's
    stats = OrderedDict(sorted(stats.items()))
    for k, v in stats.items():
        print(f"{k} : {v}.")

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
