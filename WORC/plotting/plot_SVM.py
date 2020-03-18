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
from abc import ABC, abstractmethod

import numpy as np
import sys

from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv

from WORC.plotting.compute_CI import compute_confidence
from WORC.plotting.compute_CI import compute_confidence_bootstrap
import pandas as pd
import os
import lifelines as ll
import WORC.processing.label_processing as lp
from WORC.classification import metrics, construct_classifier as cc
import WORC.addexceptions as ae
from sklearn.base import is_regressor
from collections import OrderedDict


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
        X_train_temp = [X_train[i] for i in train]
        Y_train_temp = [Y_train[i] for i in train]
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

def plot_from_pickled_locals():
    # import jsonpickle

    # kwargs = locals()
    #
    #
    # with open('/scratch/tphil/pickled_plot_svm.json', 'w') as fh:
    #     fh.write(jsonpickle.encode(kwargs))
    #
    # return
    #
    # with open('/scratch/tphil/pickled_plot_svm.json', 'r') as fh:
    #     scope = jsonpickle.decode(fh.read())

    import pickle
    with open('/scratch/tphil/pickled_plot_svm.pickle', 'rb') as fh:
        scope = pickle.load(fh)

    #print(scope)

    scope['modus'] = 'survival'
    del scope['survival']


    plot_SVM(**scope)


class BasePlotter(ABC):
    @abstractmethod
    def decision(self):
        pass

    @abstractmethod
    def _get_stats(self):
        pass

    @abstractmethod
    def _calc_y_score(self):
        pass

    def __init__(self, feature_labels, x_train, x_test, y_train, y_test, prediction, patient_ids, labels, verbose, ensemble, ensemble_scoring, shuffle_estimators, generalization, svms, label_type, bootstrap, bootstrap_n, *args, **kwargs):
        self._x_train = x_train
        self._x_test = x_test
        self._y_train = y_train
        self._y_test = y_test
        self._bootstrap_n = bootstrap_n
        self._prediction = prediction
        self._svms = svms
        self._label_type = label_type
        self._patient_ids = patient_ids
        self._do_shuffle_estimators = shuffle_estimators
        self._do_generalization = generalization
        self._ensemble = ensemble
        self._ensemble_scoring = ensemble_scoring
        self._verbose = verbose
        self._labels = labels
        self._feature_labels = feature_labels

        self._test_indices = []

        if bootstrap:
            self._bootstrappedinit()
        else:
            self._normalinit()

    def _bootstrappedinit(self, bootstrap_n):
        iterobject = range(0, bootstrap_n)

        for i in iterobject:
            print(f"Bootstrap {i + 1} / {bootstrap_n}.")
            # When bootstrapping, there is only a single train/test set.
            x_test_temp = self._x_test[0]
            x_train_temp = self._x_train[0]
            y_train_temp = self._y_train[0]
            y_test_temp = self._y_test[0]
            test_patient_ids = self._prediction[self._label_type]['patient_ID_test'][0]
            train_patient_ids = self._prediction[self._label_type]['patient_ID_train'][0]
            fitted_model = self._svms[0]

            # TODO: resample here is undefined, when bootstrap == True this code will fail
            # @Martijn please fix plx plx? :)
            x_test_temp, y_test_temp, test_patient_ids = resample(x_test_temp, y_test_temp, test_patient_ids)

            self._dowhatisinloop1(y_test_temp, test_patient_ids, fitted_model)

            # If requested, first let the SearchCV object create an ensemble
            # For bootstrapping, only do this at the first iteration
            if i == 0 and self._ensemble > 1 and not fitted_model.ensemble:
                x_train_temp = self._create_ensemble(fitted_model, x_train_temp, y_train_temp)

            self._dowhatisinloop2(x_test_temp, x_train_temp, y_train_temp, y_test_temp, test_patient_ids,
                                  train_patient_ids, fitted_model)

    def _normalinit(self):
        iterobject = range(0, len(self._y_test))

        for i in iterobject:
            print(f"Cross validation {i + 1} / {len(self._y_test)}.")

            x_test_temp = self._x_test[i]
            x_train_temp = self._x_train[i]
            y_train_temp = self._y_train[i]
            y_test_temp = self._y_test[i]
            test_patient_ids = self._prediction[self._label_type]['patient_ID_test'][i]
            train_patient_ids = self._prediction[self._label_type]['patient_ID_train'][i]
            fitted_model = self._svms[i]

            self._dowhatisinloop1(y_test_temp, test_patient_ids, fitted_model)

            if self._ensemble > 1 and not fitted_model.ensemble:
                x_train_temp = self._create_ensemble(fitted_model, x_train_temp, y_train_temp)

            self._dowhatisinloop2(x_test_temp, x_train_temp, y_train_temp, y_test_temp, test_patient_ids,
                                  train_patient_ids, fitted_model)

    def _dowhatisinloop2(self, x_test_temp, x_train_temp, y_train_temp, y_test_temp, test_patient_ids, train_patient_ids, fitted_model):
        # Create prediction
        y_prediction = fitted_model.predict(x_test_temp)

        self._y_score = self._calc_y_score(y_prediction)

        #Create a new binary score based on the thresholds if given
        # if thresholds is not None:
        #     if len(thresholds) == 1:
        #         y_prediction = y_score >= thresholds[0]
        #     elif len(thresholds) == 2:
        #         # X_train_temp = [x[0] for x in X_train_temp]
        #
        #         y_score_temp = list()
        #         y_prediction_temp = list()
        #         y_truth_temp = list()
        #         test_patient_IDs_temp = list()
        #
        #         thresholds_val = fit_thresholds(thresholds, fitted_model, X_train_temp, Y_train_temp, ensemble,
        #                                         ensemble_scoring)
        #         for pnum in range(len(y_score)):
        #             if y_score[pnum] <= thresholds_val[0] or y_score[pnum] > thresholds_val[1]:
        #                 y_score_temp.append(y_score[pnum])
        #                 y_prediction_temp.append(y_prediction[pnum])
        #                 y_truth_temp.append(y_truth[pnum])
        #                 test_patient_IDs_temp.append(test_patient_ids[pnum])
        #
        #         perc = float(len(y_prediction_temp)) / float(len(y_prediction))
        #         percentages_selected.append(perc)
        #         print(
        #             f"Selected {len(y_prediction_temp)} from {len(y_prediction)} ({perc}%) patients using two thresholds.")
        #         y_score = y_score_temp
        #         y_prediction = y_prediction_temp
        #         y_truth = y_truth_temp
        #         test_patient_ids = test_patient_IDs_temp
        #     else:
        #         raise ae.WORCValueError(
        #             f"Need None, one or two thresholds on the posterior; got {len(thresholds)}.")
        #
        # print("Truth: " + str(y_truth))
        # print("Prediction: " + str(y_prediction))

        # Add if patient was classified correctly or not to counting
        # for i_truth, i_predict, i_test_ID in zip(y_truth, y_prediction, test_patient_ids):
        #     if modus == 'multilabel':
        #         success = (i_truth == i_predict).all()
        #     else:
        #         success = i_truth == i_predict
        #
        #     if success:
        #         patient_classification_list[i_test_ID]['N_correct'] += 1
        #     else:
        #         patient_classification_list[i_test_ID]['N_wrong'] += 1

        self._calc_stats(test_patient_ids, y_prediction, x_test_temp, y_test_temp, x_train_temp, y_train_temp)


    def _create_ensemble(self, fitted_model, x_train_temp, y_train_temp):
        # Create the ensemble
        x_train_temp = [(x, self._feature_labels) for x in x_train_temp]
        fitted_model.create_ensemble(x_train_temp, y_train_temp,
                                     method=self._ensemble, verbose=self._verbose,
                                     scoring=self._ensemble_scoring)

        return x_train_temp

    def _check_patients_in_test_set(self, test_patient_ids):
        patient_classification_list = dict()

        # Check which patients are in the test set.
        for i_id in test_patient_ids:
            # Initiate counting how many times a patient is classified correctly
            if i_id not in patient_classification_list:
                patient_classification_list[i_id] = dict()
                patient_classification_list[i_id]['N_test'] = 0
                patient_classification_list[i_id]['N_correct'] = 0
                patient_classification_list[i_id]['N_wrong'] = 0

            patient_classification_list[i_id]['N_test'] += 1

            # Check if this is exactly the label of the patient within the label file
            if i_id not in self._patient_ids:
                print(f'[WORC WARNING] Patient {i_id} is not found the patient labels, removing underscore.')
                i_id = i_id.split("_")[0]
                if i_id not in self._patient_ids:
                    print(f'[WORC WARNING] Did not help, excluding patient {i_id}.')
                    continue

            self._test_indices.append(np.where(self._patient_ids == i_id)[0][0])

    def _shuffle_estimators(self, fitted_model):
        print('Shuffling estimators for random ensembling.')
        # TODO: shuffle seems to be undefined
        # @Martijn fix plxplx? :)
        shuffle(fitted_model.cv_results_['params'])
        shuffle(fitted_model.cv_results_['params_all'])

    def _generalization(self, fitted_model):
        # Compute generalization score
        print('Using generalization score for estimator ranking.')
        difference_score = abs(
            fitted_model.cv_results_['mean_train_score'] - fitted_model.cv_results_['mean_test_score'])
        generalization_score = fitted_model.cv_results_['mean_test_score'] - difference_score

        # Rerank based on score
        indices = np.argsort(generalization_score)
        fitted_model.cv_results_['params'] = [fitted_model.cv_results_['params'][i] for i in indices[::-1]]
        fitted_model.cv_results_['params_all'] = [fitted_model.cv_results_['params_all'][i] for i in
                                                  indices[::-1]]

    def _dowhatisinloop1(self, y_test_temp, test_patient_ids, fitted_model):
        self._check_patients_in_test_set(test_patient_ids)

        # Extract ground truth
        y_truth = y_test_temp

        # If required, shuffle estimators for "Random" ensembling
        if self._do_shuffle_estimators:
            self._shuffle_estimators(fitted_model)

        # If required, rank according to generalization score instead of mean_validation_score
        if self._do_generalization:
            self._generalization(fitted_model)

        self._y_truth = y_truth

    @abstractmethod
    def _calc_stats(self, *args, **kwargs):
        pass

    @abstractmethod
    def scores(self):
        pass

    def stats(self):
        s = self._get_stats()
        # for k, v in s.items():
        #     print(k)
        #     for i in v:
        #         print('\t', i)
        return s


class RegressionPlotter(BasePlotter):
    def __init__(self):
        pass


class SurvivalPlotter(BasePlotter):
    def _get_stats(self):
        return self._statsdict

    def __init__(self, *args, **kwargs):
        self._statsdict = {}
        super(SurvivalPlotter, self).__init__(*args, **kwargs)

    def decision(self):
        # Output the posteriors
        # y_scores.append(y_score)
        # y_truths.append(y_truth)
        # y_predictions.append(y_prediction)
        # pids.append(test_patient_ids)
        pass

    def _append_statsdict(self, key, value):
        if key not in self._statsdict:
            self._statsdict[key] = []

        self._statsdict[key].append(value)

    def _append_mappings(self, key, d):
        for k, v in d.items():
            self._append_statsdict(f'{key}_{k}', v)

    def _calc_stats(self, test_patient_ids, y_prediction, x_test, y_test, x_train, y_train, *args, **kwargs):
        # Compute statistics
        # Extract time to event and event from label data

        # print(len(y_train))
        # print(len(y_test))
        #
        # return

        e_test = np.asarray([bool(x[0]) for x in y_test])
        t_test = np.asarray([x[1] for x in y_test])

        e_train = np.asarray([bool(x[0]) for x in y_train])
        t_train = np.asarray([x[1] for x in y_train])

        #concordance_index_censored(y['status'], y['time'], prediction)

        # # Concordance index
        # self._cindex.append(1 - ll.utils.concordance_index(T_truth, y_prediction, E_truth))


        train = Surv().from_arrays(e_train, t_train)
        test = Surv().from_arrays(e_test, t_test)

        cic = concordance_index_censored(e_test, t_test, y_prediction)
        cii = concordance_index_ipcw(train, test, y_prediction, max(t_train))  # tau param specifies values to be truncated as max(t_test) <= max(t_train)
        cda = cumulative_dynamic_auc(train, test, y_prediction, range(int(min(t_test)), int(max(t_test))))

        cic_mappings = {
            'cindex': cic[0],
            'concordant': cic[1],
            'concordant': cic[2],
            'tied_risk': cic[3],
            'tied_time': cic[4]
        }

        cii_mappings = {
            'cindex': cii[0],
            'concordant': cii[1],
            'concordant': cii[2],
            'tied_risk': cii[3],
            'tied_time': cii[4]
        }

        cda_mappings = {
            'auc': cda[0],
            'mean_auc': cda[1]
        }

        # TODO: compute_confidence(..) uitrekenen

        self._append_mappings('concordance_index_censored', cic_mappings)
        self._append_mappings('concordance_index_ipcw', cii_mappings)
        self._append_mappings('cumulative_dynamic_auc', cda_mappings)

        # # Fit Cox model using SVR output, time to event and event
        # data = {'predict': y_prediction, 'E': E_truth, 'T': T_truth}
        # data = pd.DataFrame(data=data, index=test_patient_ids)
        #
        # cph = ll.CoxPHFitter()
        # cph.fit(data, duration_col='T', event_col='E')
        #
        # self._statsdict['coxcoef'].append(cph.summary['coef']['predict'])
        # self._statsdict['coxp'].append(cph.summary['p']['predict'])

    def scores(self):
        # if output in ['scores', 'decision']:
        #     # Keep track of all groundth truths and scores
        #     y_truths = list()
        #     y_scores = list()
        #     y_predictions = list()
        #     pids = list()
        #
        #     # Output the posteriors
        #     y_scores.append(y_score)
        #     y_truths.append(y_truth)
        #     y_predictions.append(y_prediction)
        #     pids.append(test_patient_ids)
        pass

    def _calc_y_score(self, y_prediction):
        return y_prediction

class MultilabelClassificationPlotter():
    def __init__(self):
        pass


class ClassificationPlotter(BasePlotter):
    def __init__(self):
        pass


def plot_SVM(prediction, label_data, label_type, show_plots=False,
             alpha=0.95, ensemble=False, verbose=True,
             ensemble_scoring=None, output='stats',
             modus='singlelabel',
             thresholds=None,
             generalization=False, shuffle_estimators=False,
             bootstrap=False, bootstrap_N=1000):
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

    # scope = locals()
    # import pickle
    # with open('/scratch/tphil/pickled_plot_svm.pickle', 'wb') as fh:
    #     pickle.dump(scope, fh)
    #
    # return

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

    n_labels = len(label_type)
    patient_IDs = label_data['patient_IDs']
    labels = label_data['label']

    if type(label_type) is list:
        # FIXME: Support for multiple label types not supported yet.
        print('[WORC Warning] Support for multiple label types not supported yet. Taking first label for plot_SVM.')
        original_label_type = label_type[:]
        label_type = keys[0]

    # Extract the estimators, features and labels
    SVMs = prediction[label_type]['classifiers']

    survival = (modus == 'survival')
    regression = cc.is_regression_classifier(SVMs[0].best_estimator_)
    multilabel = (modus == 'multilabel')

    Y_test = prediction[label_type]['Y_test']
    X_test = prediction[label_type]['X_test']
    X_train = prediction[label_type]['X_train']
    Y_train = prediction[label_type]['Y_train']
    feature_labels = prediction[label_type]['feature_labels']

    if survival:
        plotter = SurvivalPlotter(feature_labels, X_train, X_test, Y_train, Y_test, prediction, patient_IDs, labels, verbose, ensemble, ensemble_scoring, shuffle_estimators, generalization, SVMs, label_type, bootstrap, bootstrap_N)
    # elif regression:
    #     plotter = RegressionPlotter()
    # elif multilabel:
    #     plotter = MultilabelClassificationPlotter()
    # else:
    #     plotter = ClassificationPlotter()


    if plotter and output == 'stats':
        return plotter.stats()
    # elif plotter and output == 'scores':
    #     return plotter.scores()

    sensitivity = list()
    specificity = list()
    precision = list()
    npv = list()
    accuracy = list()
    bca = list()
    auc = list()
    f1_score_list = list()

    if multilabel:
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

    patient_classification_list = dict()
    percentages_selected = list()

    if output in ['scores', 'decision']:
        # Keep track of all groundth truths and scores
        y_truths = list()
        y_scores = list()
        y_predictions = list()
        pids = list()

    # Loop over the test sets, which correspond to cross-validation
    # or bootstrapping iterations
    if bootstrap:
        iterobject = range(0, bootstrap_N)
    else:
        iterobject = range(0, len(Y_test))

    for i in iterobject:
        print("\n")
        if bootstrap:
            print(f"Bootstrap {i + 1} / {bootstrap_N}.")
        else:
            print(f"Cross validation {i + 1} / {len(Y_test)}.")

        test_indices = list()

        # When bootstrapping, there is only a single train/test set.
        if bootstrap:
            X_test_temp = X_test[0]
            X_train_temp = X_train[0]
            Y_train_temp = Y_train[0]
            Y_test_temp = Y_test[0]
            test_patient_IDs = prediction[label_type]['patient_ID_test'][0]
            train_patient_IDs = prediction[label_type]['patient_ID_train'][0]
            fitted_model = SVMs[0]
        else:
            X_test_temp = X_test[i]
            X_train_temp = X_train[i]
            Y_train_temp = Y_train[i]
            Y_test_temp = Y_test[i]
            test_patient_IDs = prediction[label_type]['patient_ID_test'][i]
            train_patient_IDs = prediction[label_type]['patient_ID_train'][i]
            fitted_model = SVMs[i]

        # If bootstrap, generate a bootstrapped sample
        if bootstrap:
            # TODO: resample here is undefined, when bootstrap == True this code will fail
            # @Martijn please fix plx plx? :)
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

            # TODO: shuffle seems to be undefined
            # @Martijn fix plxplx?
            print('Shuffling estimators for random ensembling.')
            shuffle(fitted_model.cv_results_['params'])
            shuffle(fitted_model.cv_results_['params_all'])

        # If required, rank according to generalization score instead of mean_validation_score
        if generalization:
            # Compute generalization score
            print('Using generalization score for estimator ranking.')
            difference_score = abs(
                fitted_model.cv_results_['mean_train_score'] - fitted_model.cv_results_['mean_test_score'])
            generalization_score = fitted_model.cv_results_['mean_test_score'] - difference_score

            # Rerank based on score
            indices = np.argsort(generalization_score)
            fitted_model.cv_results_['params'] = [fitted_model.cv_results_['params'][i] for i in indices[::-1]]
            fitted_model.cv_results_['params_all'] = [fitted_model.cv_results_['params_all'][i] for i in
                                                      indices[::-1]]

        # If requested, first let the SearchCV object create an ensemble
        if bootstrap and i > 0:
            # For bootstrapping, only do this at the first iteration
            pass
        elif ensemble > 1 and not fitted_model.ensemble:
            # NOTE: Added for backwards compatability
            if not hasattr(fitted_model, 'cv_iter'):
                cv_iter = list(fitted_model.cv.split(X_train_temp, Y_train_temp))
                fitted_model.cv_iter = cv_iter

            # Create the ensemble
            X_train_temp = [(x, feature_labels) for x in X_train_temp]
            fitted_model.create_ensemble(X_train_temp, Y_train_temp,
                                         method=ensemble, verbose=verbose,
                                         scoring=ensemble_scoring)

        # Create prediction
        y_prediction = fitted_model.predict(X_test_temp)

        if regression or survival:
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

                perc = float(len(y_prediction_temp)) / float(len(y_prediction))
                percentages_selected.append(perc)
                print(
                    f"Selected {len(y_prediction_temp)} from {len(y_prediction)} ({perc}%) patients using two thresholds.")
                y_score = y_score_temp
                y_prediction = y_prediction_temp
                y_truth = y_truth_temp
                test_patient_IDs = test_patient_IDs_temp
            else:
                raise ae.WORCValueError(
                    f"Need None, one or two thresholds on the posterior; got {len(thresholds)}.")

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
                    accuracy_temp, bca_temp, sensitivity_temp, \
                    specificity_temp, \
                    precision_temp, npv_temp, f1_score_temp, auc_temp = \
                        metrics.performance_singlelabel(y_truth,
                                                        y_prediction,
                                                        y_score,
                                                        regression)
                else:
                    r2score_temp, MSE_temp, coefICC_temp, PearsonC_temp, \
                    PearsonP_temp, SpearmanC_temp, \
                    SpearmanP_temp = \
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
                        raise ae.WORCNotImplementedError(
                            'Multiclass classification evaluation is not supported in WORC.')

                    y_truth_temp.append(label[0][0])
                    label = np.where(yp == 1)
                    y_prediction_temp.append(label[0][0])

                y_truth = y_truth_temp
                y_prediction = y_prediction_temp

                # Compute multilabel performance metrics
                accuracy_temp, sensitivity_temp, specificity_temp, \
                precision_temp, npv_temp, f1_score_temp, auc_temp, acc_av_temp = \
                    metrics.performance_multilabel(y_truth,
                                                   y_prediction,
                                                   y_score)

                # Compute all single label performance metrics as well
                for i_label in range(n_labels):
                    y_truth_single = [i == i_label for i in y_truth]
                    y_prediction_single = [i == i_label for i in y_prediction]
                    y_score_single = y_score[:, i_label]

                    accuracy_temp_single, bca_temp_single, sensitivity_temp_single, specificity_temp_single, \
                    precision_temp_single, npv_temp_single, f1_score_temp_single, auc_temp_single = \
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

        # TODO: override with new survival code
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
        return y_truths, y_scores, y_predictions, pids
    elif output == 'stats':
        # Compute statistics
        # Extract sample si ze
        N_1 = float(len(train_patient_IDs))
        N_2 = float(len(test_patient_IDs))

        # Compute alpha confidence intervals (CIs)
        stats = dict()
        if not regression:
            if bootstrap:
                # Compute once for the real test set the performance
                X_test_temp = X_test[0]
                y_truth = Y_test[0]
                y_prediction = fitted_model.predict(X_test_temp)

                if regression:
                    y_score = y_prediction
                else:
                    y_score = fitted_model.predict_proba(X_test_temp)[:, 1]

                accuracy_test, bca_test, sensitivity_test, specificity_test, \
                precision_test, npv_test, f1_score_test, auc_test = \
                    metrics.performance_singlelabel(y_truth,
                                                    y_prediction,
                                                    y_score,
                                                    regression)

                stats[
                    "Accuracy 95%:"] = f"{accuracy_test} {str(compute_confidence_bootstrap(accuracy, accuracy_test, N_1, alpha))}"
                stats["BCA 95%:"] = f"{bca_test} {str(compute_confidence_bootstrap(bca, bca_test, N_1, alpha))}"
                stats["AUC 95%:"] = f"{auc_test} {str(compute_confidence_bootstrap(auc, auc_test, N_1, alpha))}"
                stats[
                    "F1-score 95%:"] = f"{f1_score_list_test} {str(compute_confidence_bootstrap(f1_score_list, f1_score_test, N_1, alpha))}"
                stats[
                    "Precision 95%:"] = f"{precision_test} {str(compute_confidence_bootstrap(precision, precision_test, N_1, alpha))}"
                stats["NPV 95%:"] = f"{npv_test} {str(compute_confidence_bootstrap(npv, npv_test, N_1, alpha))}"
                stats[
                    "Sensitivity 95%: "] = f"{sensitivity_test} {str(compute_confidence_bootstrap(sensitivity, sensitivity_test, N_1, alpha))}"
                stats[
                    "Specificity 95%:"] = f"{specificity_test} {str(compute_confidence_bootstrap(specificity, specificity_test, N_1, alpha))}"
            else:
                stats[
                    "Accuracy 95%:"] = f"{np.nanmean(accuracy)} {str(compute_confidence(accuracy, N_1, N_2, alpha))}"
                stats["BCA 95%:"] = f"{np.nanmean(bca)} {str(compute_confidence(bca, N_1, N_2, alpha))}"
                stats["AUC 95%:"] = f"{np.nanmean(auc)} {str(compute_confidence(auc, N_1, N_2, alpha))}"
                stats[
                    "F1-score 95%:"] = f"{np.nanmean(f1_score_list)} {str(compute_confidence(f1_score_list, N_1, N_2, alpha))}"
                stats[
                    "Precision 95%:"] = f"{np.nanmean(precision)} {str(compute_confidence(precision, N_1, N_2, alpha))}"
                stats["NPV 95%:"] = f"{np.nanmean(npv)} {str(compute_confidence(npv, N_1, N_2, alpha))}"
                stats[
                    "Sensitivity 95%: "] = f"{np.nanmean(sensitivity)} {str(compute_confidence(sensitivity, N_1, N_2, alpha))}"
                stats[
                    "Specificity 95%:"] = f"{np.nanmean(specificity)} {str(compute_confidence(specificity, N_1, N_2, alpha))}"

            if modus == 'multilabel':
                stats[
                    "Average Accuracy 95%:"] = f"{np.nanmean(acc_av)} {str(compute_confidence(acc_av, N_1, N_2, alpha))}"

            if thresholds is not None:
                if len(thresholds) == 2:
                    # Compute percentage of patients that was selected
                    stats[
                        "Percentage Selected 95%:"] = f"{np.nanmean(percentages_selected)} {str(compute_confidence(percentages_selected, N_1, N_2, alpha))}"

            # Extract statistics on how often patients got classified correctly
            alwaysright = dict()
            alwayswrong = dict()
            percentages = dict()
            for i_ID in patient_classification_list:
                percentage_right = patient_classification_list[i_ID]['N_correct'] / float(
                    patient_classification_list[i_ID]['N_test'])

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

            stats["Always right"] = alwaysright
            stats["Always wrong"] = alwayswrong
            stats['Percentages'] = percentages
        else:
            # Regression
            stats['R2-score 95%: '] = f"{np.nanmean(r2_score)} {str(compute_confidence(r2score, N_1, N_2, alpha))}"
            stats['MSE 95%: '] = f"{np.nanmean(MSE)} {str(compute_confidence(MSE, N_1, N_2, alpha))}"
            stats['ICC 95%: '] = f"{np.nanmean(coefICC)} {str(compute_confidence(coefICC, N_1, N_2, alpha))}"
            stats['PearsonC 95%: '] = f"{np.nanmean(PearsonC)} {str(compute_confidence(PearsonC, N_1, N_2, alpha))}"
            stats['PearsonP 95%: '] = f"{np.nanmean(PearsonP)} {str(compute_confidence(PearsonP, N_1, N_2, alpha))}"
            stats[
                'SpearmanC 95%: '] = f"{np.nanmean(SpearmanC)} {str(compute_confidence(SpearmanC, N_1, N_2, alpha))}"
            stats[
                'SpearmanP 95%: '] = f"{np.nanmean(SpearmanP)} {str(compute_confidence(SpearmanP, N_1, N_2, alpha))}"

            if survival:
                stats[
                    "Concordance 95%:"] = f"{np.nanmean(cindex)} {str(compute_confidence(cindex, N_1, N_2, alpha))}"
                stats[
                    "Cox coef. 95%:"] = f"{np.nanmean(coxcoef)} {str(compute_confidence(coxcoef, N_1, N_2, alpha))}"
                stats["Cox p 95%:"] = f"{np.nanmean(coxp)} {str(compute_confidence(coxp, N_1, N_2, alpha))}"

        # Print all CI's
        stats = OrderedDict(sorted(stats.items()))
        for k, v in stats.items():
            print(f"{k} : {v}.")

        return stats



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
