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

import os
from abc import ABCMeta, abstractmethod
from collections.abc import Sized
import numpy as np
import warnings
import numbers
import random
import string
import fastr
from fastr.api import ResourceLimit
from joblib import Parallel, delayed
from scipy.stats import rankdata
import six
import pandas as pd
import json
import glob
from itertools import islice
import shutil

from sklearn.model_selection._search import ParameterSampler
from sklearn.model_selection._search import ParameterGrid, _check_param_grid
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.base import MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import indexable, check_is_fitted
from sklearn.model_selection._split import check_cv
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.validation import _check_fit_params
from sklearn.model_selection._validation import _aggregate_score_dicts

from WORC.classification.fitandscore import fit_and_score, replacenan
from WORC.classification.metrics import check_multimetric_scoring
from WORC.classification import construct_classifier as cc
from WORC.featureprocessing.Preprocessor import Preprocessor
from WORC.detectors.detectors import DebugDetector
import WORC.addexceptions as WORCexceptions


def rms_score(truth, prediction):
    """Root-mean-square-error metric."""
    return np.sqrt(mean_squared_error(truth, prediction))


def sar_score(truth, prediction):
    """SAR metric from Caruana et al. 2004."""
    ROC = roc_auc_score(truth, prediction)
    # Convert score to binaries first
    for num in range(0, len(prediction)):
        if prediction[num] >= 0.5:
            prediction[num] = 1
        else:
            prediction[num] = 0

    ACC = accuracy_score(truth, prediction)
    RMS = rms_score(truth, prediction)
    SAR = (ACC + ROC + (1 - RMS))/3
    return SAR


def chunksdict(data, SIZE):
    """Split a dictionary in equal parts of certain slice."""
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class Ensemble(six.with_metaclass(ABCMeta, BaseEstimator,
                                  MetaEstimatorMixin)):
    """Ensemble of BaseSearchCV Estimators."""

    # @abstractmethod
    def __init__(self, estimators):
        """Initialize object with list of estimators."""
        if not estimators:
            message = 'You supplied an empty list of estimators: No ensemble creation possible.'
            raise WORCexceptions.WORCValueError(message)
        self.estimators = estimators
        self.n_estimators = len(estimators)

    def predict(self, X):
        """Call predict on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self.estimators[0]._check_is_fitted('predict')

        # Check if we are dealing with multilabel
        if len(self.estimators[0].predict(X).shape) == 1:
            nlabels = 1
        else:
            nlabels = self.estimators[0].predict(X).shape[1]

        if type(self.estimators[0].best_estimator_) == OneVsRestClassifier:
            multilabel = True
        elif nlabels > 1:
            multilabel = True
        else:
            multilabel = False

        if multilabel:
            # Multilabel
            outcome = np.zeros((self.n_estimators, len(X), nlabels))
            for num, est in enumerate(self.estimators):
                if hasattr(est, 'predict_proba'):
                    # BUG: SVM kernel can be wrong type
                    if hasattr(est.best_estimator_, 'kernel'):
                        est.best_estimator_.kernel = str(est.best_estimator_.kernel)
                    outcome[num, :, :] = est.predict_proba(X)
                else:
                    outcome[num, :, :] = est.predict(X)

            # Replace NAN if they are there
            if np.isnan(outcome).any():
                print('[WARNING] Predictions contain NaN, removing those rows.')
                outcome = outcome[~np.isnan(outcome).any(axis=1)]

            outcome = np.squeeze(np.mean(outcome, axis=0))

            # NOTE: Binarize specifically for multiclass
            for i in range(0, outcome.shape[0]):
                label = np.argmax(outcome[i, :])
                outcome[i, :] = np.zeros(outcome.shape[1])
                outcome[i, label] = 1

        else:
            # Singlelabel
            outcome = np.zeros((self.n_estimators, len(X)))
            for num, est in enumerate(self.estimators):
                if hasattr(est, 'predict_proba'):
                    # BUG: SVM kernel can be wrong type
                    if hasattr(est.best_estimator_, 'kernel'):
                        est.best_estimator_.kernel = str(est.best_estimator_.kernel)
                    outcome[num, :] = est.predict_proba(X)[:, 1]
                else:
                    outcome[num, :] = est.predict(X)

            # Replace NAN if they are there
            outcome = outcome[~np.isnan(outcome).any(axis=1)]

            outcome = np.squeeze(np.mean(outcome, axis=0))

            # Binarize
            isclassifier = is_classifier(est.best_estimator_)

            if isclassifier:
                outcome[outcome >= 0.5] = 1
                outcome[outcome < 0.5] = 0

        return outcome

    def predict_proba(self, X):
        """Call predict_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self.estimators[0]._check_is_fitted('predict_proba')

        # Check if we are dealing with multilabel
        if len(self.estimators[0].predict(X).shape) == 1:
            nlabels = 1
        else:
            nlabels = self.estimators[0].predict(X).shape[1]

        if type(self.estimators[0].best_estimator_) == OneVsRestClassifier:
            multilabel = True
        elif nlabels > 1:
            multilabel = True
        else:
            multilabel = False

        if multilabel:
            # Multilabel
            outcome = np.zeros((self.n_estimators, len(X), nlabels))
            for num, est in enumerate(self.estimators):
                if hasattr(est, 'predict_proba'):
                    # BUG: SVM kernel can be wrong type
                    if hasattr(est.best_estimator_, 'kernel'):
                        est.best_estimator_.kernel = str(est.best_estimator_.kernel)
                    outcome[num, :, :] = est.predict_proba(X)
                else:
                    outcome[num, :, :] = est.predict(X)

            # Replace NAN if they are there
            if np.isnan(outcome).any():
                print('[WARNING] Predictions contain NaN, removing those rows.')
                outcome = outcome[~np.isnan(outcome).any(axis=1)]

            outcome = np.squeeze(np.mean(outcome, axis=0))
        else:
            # Single label
            # For probabilities, we get both a class0 and a class1 score
            outcome = np.zeros((len(X), 2))
            outcome_class1 = np.zeros((self.n_estimators, len(X)))
            outcome_class2 = np.zeros((self.n_estimators, len(X)))
            for num, est in enumerate(self.estimators):
                # BUG: SVM kernel can be wrong type
                if hasattr(est.best_estimator_, 'kernel'):
                    est.best_estimator_.kernel = str(est.best_estimator_.kernel)
                outcome_class1[num, :] = est.predict_proba(X)[:, 0]
                outcome_class2[num, :] = est.predict_proba(X)[:, 1]

            outcome[:, 0] = np.squeeze(np.mean(outcome_class1, axis=0))
            outcome[:, 1] = np.squeeze(np.mean(outcome_class2, axis=0))

        return outcome

    def predict_log_proba(self, X):
        """Call predict_log_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self.estimators[0]._check_is_fitted('predict_log_proba')

        outcome = np.zeros((self.n_estimators, len(X)))
        for num, est in enumerate(self.estimators):
            outcome[num, :] = est.predict_log_proba(X)

        outcome = np.squeeze(np.mean(outcome, axis=0))
        return outcome

    def decision_function(self, X):
        """Call decision_function on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``decision_function``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self.estimators[0]._check_is_fitted('decision_function')

        # NOTE: Check if we are dealing with multilabel
        if type(self.estimators[0].best_estimator_) == OneVsRestClassifier:
            # Multilabel
            nlabels = self.estimators[0].decision_function(X).shape[1]
            outcome = np.zeros((self.n_estimators, len(X), nlabels))
            for num, est in enumerate(self.estimators):
                outcome[num, :, :] = est.decision_function(X)

            outcome = np.squeeze(np.mean(outcome, axis=0))

        else:
            # Singlelabel
            outcome = np.zeros((self.n_estimators, len(X)))
            for num, est in enumerate(self.estimators):
                outcome[num, :] = est.decision_function(X)

            outcome = np.squeeze(np.mean(outcome, axis=0))

        return outcome

    def transform(self, X):
        """Call transform on the estimator with the best found parameters.

        Only available if the underlying estimator supports ``transform`` and
        ``refit=True``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self.estimators[0]._check_is_fitted('transform')

        outcome = np.zeros((self.n_estimators, len(X)))
        for num, est in enumerate(self.estimators):
            outcome[num, :] = est.transform(X)

        outcome = np.squeeze(np.mean(outcome, axis=0))
        return outcome

    def inverse_transform(self, Xt):
        """Call inverse_transform on the estimator with the best found params.

        Only available if the underlying estimator implements
        ``inverse_transform`` and ``refit=True``.

        Parameters
        -----------
        Xt : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self.estimators[0]._check_is_fitted('inverse_transform')

        outcome = np.zeros((self.n_estimators, len(Xt)))
        for num, est in enumerate(self.estimators):
            outcome[num, :] = est.transform(Xt)

        outcome = np.squeeze(np.mean(outcome, axis=0))
        return outcome


class BaseSearchCV(six.with_metaclass(ABCMeta, BaseEstimator,
                                      MetaEstimatorMixin)):
    """Base class for hyper parameter search with cross-validation."""

    @abstractmethod
    def __init__(self, param_distributions={}, n_iter=10, scoring=None,
                 fit_params=None, n_jobs=1, iid=True,
                 refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs',
                 random_state=None, error_score='raise',
                 return_train_score=True,
                 n_jobspercore=100, maxlen=100, fastr_plugin=None, memory='2G',
                 ranking_score='test_score', refit_workflows=False):
        """Initialize SearchCV Object."""
        # Added for fastr and joblib executions
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.n_jobspercore = n_jobspercore
        self.random_state = random_state
        self.ensemble = list()
        self.fastr_plugin = fastr_plugin
        self.memory = memory

        # Below are the defaults from sklearn
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.fit_params = fit_params if fit_params is not None else {}
        self.iid = iid
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch

        # Manually added steps
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.maxlen = maxlen
        self.ranking_score = ranking_score
        self.refit_workflows = refit_workflows
        self.fitted_workflows = list()

        # Only for WORC Paper
        self.test_RS = True

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def score(self, X, y=None):
        """Compute the score (i.e. probability) on a given data.

        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float

        """
        if self.scorer_ is None:
            raise ValueError("No score function explicitly defined, "
                             "and the estimator doesn't provide one %s"
                             % self.best_estimator_)

        X, y = self.preprocess(X, y)

        return self.scorer_(self.best_estimator_, X, y)

    def _check_is_fitted(self, method_name):
        if not self.refit:
            raise NotFittedError(('This GridSearchCV instance was initialized '
                                  'with refit=False. %s is '
                                  'available only after refitting on the best '
                                  'parameters. ') % method_name)
        else:
            check_is_fitted(self, 'best_estimator_')

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict(self, X):
        """Call predict on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('predict')

        if self.ensemble:
            return self.ensemble.predict(X)
        else:
            X, _ = self.preprocess(X)
            return self.best_estimator_.predict(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_proba(self, X):
        """Call predict_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('predict_proba')

        # BUG: kernel sometimes saved as unicode
        # BUG: SVM kernel can be wrong type
        if hasattr(self.best_estimator_, 'kernel'):
            self.best_estimator_.kernel = str(self.best_estimator_.kernel)
        if self.ensemble:
            return self.ensemble.predict_proba(X)
        else:
            X, _ = self.preprocess(X)
            return self.best_estimator_.predict_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_log_proba(self, X):
        """Call predict_log_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('predict_log_proba')

        # BUG: SVM kernel can be wrong type
        if hasattr(self.est.best_estimator_, 'kernel'):
            self.best_estimator_.kernel = str(self.best_estimator_.kernel)

        if self.ensemble:
            return self.ensemble.predict_log_proba(X)
        else:
            X, _ = self.preprocess(X)
            return self.best_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def decision_function(self, X):
        """Call decision_function on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``decision_function``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('decision_function')

        if self.ensemble:
            return self.ensemble.decision_function(X)
        else:
            X, _ = self.preprocess(X)
            return self.best_estimator_.decision_function(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def transform(self, X):
        """Call transform on the estimator with the best found parameters.

        Only available if the underlying estimator supports ``transform`` and
        ``refit=True``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('transform')

        if self.ensemble:
            return self.ensemble.transform(X)
        else:
            X = self.preprocess(X)
            return self.best_estimator_.transform(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def inverse_transform(self, Xt):
        """Call inverse_transform on the estimator with the best found params.

        Only available if the underlying estimator implements
        ``inverse_transform`` and ``refit=True``.

        Parameters
        -----------
        Xt : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('inverse_transform')

        if self.ensemble:
            return self.ensemble.transform(Xt)
        else:
            Xt, _ = self.preprocess(Xt)
            return self.best_estimator_.transform(Xt)

    def preprocess(self, X, y=None, training=False):
        """Apply the available preprocssing methods to the features."""
        if self.best_preprocessor is not None:
            X = self.best_preprocessor.transform(X)

        if self.best_encoder is not None:
            X = self.best_encoder.transform(X)

        if self.best_imputer is not None:
            X = self.best_imputer.transform(X)

        # Replace nan if still left
        X = replacenan(np.asarray(X)).tolist()

        if self.best_groupsel is not None:
            X = self.best_groupsel.transform(X)

        if not training and hasattr(self, 'overfit_scaler') and self.overfit_scaler:
            # Overfit the feature scaling on the test set
            # NOTE: Never use this in an actual model, only to assess how
            # different your features are in your train and test sets
            m = '[WORC WARNING] You choose to overfit the feature scaling. ' +\
                'Never use this in an actual model, only to assess how ' +\
                'different your features are in your train and test sets.'
            print(m)
            scaler = StandardScaler().fit(X)

            if scaler is not None:
                X = scaler.transform(X)
        else:
            if self.best_scaler is not None:
                X = self.best_scaler.transform(X)

        if self.best_varsel is not None:
            X = self.best_varsel.transform(X)

        if self.best_reliefsel is not None:
            X = self.best_reliefsel.transform(X)

        if self.best_modelsel is not None:
            X = self.best_modelsel.transform(X)

        if self.best_pca is not None:
            X = self.best_pca.transform(X)

        if self.best_statisticalsel is not None:
            X = self.best_statisticalsel.transform(X)

        # Only resampling in training phase, i.e. if we have the labels
        if y is not None:
            if self.best_Sampler is not None:
                X, y = self.best_Sampler.transform(X, y)

        return X, y

    def process_fit(self, n_splits, parameters_all,
                    test_sample_counts, test_score_dicts,
                    train_score_dicts, fit_time, score_time, cv_iter,
                    X, y, fitted_workflows=None):
        """Process a fit.

        Process the outcomes of a SearchCV fit and find the best settings
        over all cross validations from all hyperparameters tested

        Very similar to the _format_results function or the original SearchCV.

        """
        # test_score_dicts and train_score dicts are lists of dictionaries and
        # we make them into dict of lists
        if self.verbose:
            print('Processing fits.')
        test_scores = _aggregate_score_dicts(test_score_dicts)
        if self.return_train_score:
            train_scores = _aggregate_score_dicts(train_score_dicts)

        # We take only one result per split, default by sklearn
        pipelines_per_split = int(len(parameters_all) / n_splits)
        candidate_params_all = list(parameters_all[:pipelines_per_split])
        n_candidates = len(candidate_params_all)

        # Store some of the resulting scores
        results = dict()

        # Computed the (weighted) mean and std for test scores alone
        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_."""
            array = np.transpose(np.array(array, dtype=np.float64).reshape(n_splits,
                                                                           n_candidates))

            if splits:
                for split_i in range(n_splits):
                    results["split%d_%s"
                            % (split_i, key_name)] = array[:, split_i]

            try:
                array_means = np.average(array, axis=1, weights=weights)
            except ZeroDivisionError as e:
                e = f'[WORC Warning] {e}. Setting {key_name} to unweighted.'
                print(e)
                array_means = np.average(array, axis=1)

            results['mean_%s' % key_name] = array_means

            array_mins = np.min(array, axis=1)
            results['min_%s' % key_name] = array_mins

            # Weighted std is not directly available in numpy
            try:
                array_stds = np.sqrt(np.average((array -
                                                 array_means[:, np.newaxis]) ** 2,
                                                axis=1, weights=weights))
            except ZeroDivisionError as e:
                e = f'[WORC Warning] {e}. Setting {key_name} to unweighted.'
                print(e)
                array_stds = np.sqrt(np.average((array -
                                                 array_means[:, np.newaxis]) ** 2,
                                                axis=1))

            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        _store('fit_time', fit_time)
        _store('score_time', score_time)

        # Store scores
        # Check whether to do multimetric scoring
        test_estimator = cc.construct_classifier(candidate_params_all[0])
        scorers, self.multimetric_ = check_multimetric_scoring(
            test_estimator, scoring=self.scoring)

        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
                                      dtype=np.int)

        if self.iid != 'deprecated':
            warnings.warn(
                "The parameter 'iid' is deprecated in 0.22 and will be "
                "removed in 0.24.", FutureWarning
            )
            iid = self.iid
        else:
            iid = False

        icheck = 0
        for scorer_name in scorers.keys():
            # Computed the (weighted) mean and std for test scores alone
            key_name = 'test_%s' % scorer_name
            _store('test_%s' % scorer_name, test_scores[scorer_name],
                   splits=True, rank=True,
                   weights=test_sample_counts if iid else None)

            if DebugDetector().do_detection() and icheck == 0:
                # Check the scores for some splits
                for i in range(10):
                    print('Iteration: ' + str(i))
                    print(test_scores[scorer_name][i])
                    print(results["split%d_%s" % (0, key_name)][i])
                    print(test_scores[scorer_name][i + 10])
                    print(results["split%d_%s" % (1, key_name)][i])
                    print(results['mean_%s' % key_name][i])
                    print('\n')
                    icheck += 1

            if self.return_train_score:
                _store('train_%s' % scorer_name, train_scores[scorer_name],
                       splits=True)

        # Compute the "Generalization" score
        difference_score = abs(results['mean_train_score'] - results['mean_test_score'])
        generalization_score = results['mean_test_score'] - difference_score
        results['generalization_score'] = generalization_score
        results['rank_generalization_score'] = np.asarray(
            rankdata(-results['generalization_score'], method='min'), dtype=np.int32)

        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, str) or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers) and not callable(self.refit):
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key or a "
                                 "callable to refit an estimator with the "
                                 "best parameter setting on the whole "
                                 "data and make the best_* attributes "
                                 "available for that metric. If this is "
                                 "not needed, refit should be set to "
                                 "False explicitly. %r was passed."
                                 % self.refit)
            else:
                refit_metric = self.refit
        else:
            refit_metric = 'score'

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index_ = self.refit(results)
                if not isinstance(self.best_index_, numbers.Integral):
                    raise TypeError('best_index_ returned is not an integer')
                if (self.best_index_ < 0 or
                   self.best_index_ >= len(results["params"])):
                    raise IndexError('best_index_ index out of range')
            else:
                self.best_index_ = results["rank_test_%s"
                                           % refit_metric].argmin()
                self.best_score_ = results["mean_test_%s" % refit_metric][
                                           self.best_index_]
            self.best_params_ = candidate_params_all[self.best_index_]

        # Rank the indices of scores from all parameter settings
        ranked_test_scores = results["rank_" + self.ranking_score]
        indices = range(0, len(ranked_test_scores))
        sortedindices = [x for _, x in sorted(zip(ranked_test_scores, indices))]

        # In order to reduce the memory used, we will only save
        # a maximum of results
        maxlen = min(self.maxlen, n_candidates)
        bestindices = sortedindices[0:maxlen]

        candidate_params_all = np.asarray(candidate_params_all)[bestindices].tolist()
        for k in results.keys():
            results[k] = results[k][bestindices]
        n_candidates = len(candidate_params_all)
        results['params'] = candidate_params_all

        # Store the atributes of the best performing estimator
        best_index = np.flatnonzero(results["rank_" + self.ranking_score] == 1)[0]
        best_parameters_all = candidate_params_all[best_index]

        # Store several objects
        self.cv_results_ = results
        self.n_splits_ = n_splits
        self.cv_iter = cv_iter
        self.best_index_ = best_index
        self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # We always refit on the full dataset
            indices = np.arange(0, len(y))
            self.refit_and_score(X, y, best_parameters_all,
                                 train=indices, test=indices)

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        # Refit the top performing workflows on the full training dataset
        if self.refit_workflows:
            # Select only from one train-val split, as they are identical
            fitted_workflows = fitted_workflows[:pipelines_per_split]

            # Sort according to best indices
            fitted_workflows = [fitted_workflows[i] for i in bestindices]

            # Remove None workflows
            fitted_workflows = [f for f in fitted_workflows if f is not None]

            self.fitted_workflows = fitted_workflows

        return self

    def refit_and_score(self, X, y, parameters_all,
                        train, test, verbose=None):
        """Refit the base estimator and attributes such as GroupSel.

        Parameters
        ----------
        X: array, mandatory
                Array containingfor each object (rows) the feature values
                (1st Column) and the associated feature label (2nd Column).

        y: list(?), mandatory
                List containing the labels of the objects.

        parameters_all: dictionary, mandatory
                Contains the settings used for the all preprocessing functions
                and the fitting. TODO: Create a default object and show the
                fields.

        train: list, mandatory
                Indices of the objects to be used as training set.

        test: list, mandatory
                Indices of the objects to be used as testing set.

        """
        if verbose is None:
            verbose = self.verbose

        # Preprocess features if required
        if 'FeatPreProcess' in parameters_all:
            if parameters_all['FeatPreProcess'] == 'True':
                print("Preprocessing features.")
                feature_values = np.asarray([x[0] for x in X])
                feature_labels = np.asarray([x[1] for x in X])
                preprocessor = Preprocessor(verbose=False)
                preprocessor.fit(feature_values, feature_labels=feature_labels[0, :])
                feature_values = preprocessor.transform(feature_values)
                feature_labels = preprocessor.transform(feature_labels)
                X_fit = [(values, labels) for values, labels in zip(feature_values, feature_labels)]
            else:
                X_fit = X
                preprocessor = None
        else:
            X_fit = X
            preprocessor = None

        # Refit all preprocessing functions
        fit_params = _check_fit_params(X_fit, self.fit_params)
        out = fit_and_score(X_fit, y, self.scoring,
                            train, test, parameters_all,
                            fit_params=fit_params,
                            return_train_score=self.return_train_score,
                            return_n_test_samples=True,
                            return_times=True, return_parameters=False,
                            return_estimator=True,
                            error_score=self.error_score,
                            verbose=verbose,
                            return_all=True)

        # Associate best options with new fits
        (save_data, GroupSel, VarSel, SelectModel, feature_labels, scalers,
            encoders, Imputers, PCAs, StatisticalSel, ReliefSel, Sampler) = out
        fitted_estimator = save_data[-2]
        self.best_groupsel = GroupSel
        self.best_scaler = scalers
        self.best_varsel = VarSel
        self.best_modelsel = SelectModel
        self.best_preprocessor = preprocessor
        self.best_imputer = Imputers
        self.best_encoder = encoders
        self.best_pca = PCAs
        self.best_featlab = feature_labels
        self.best_statisticalsel = StatisticalSel
        self.best_reliefsel = ReliefSel
        self.best_Sampler = Sampler
        self.best_estimator_ = fitted_estimator
        self.best_params_ = parameters_all

        return self

    def create_ensemble(self, X_train, Y_train, verbose=None, initialize=True,
                        scoring=None, method=50, overfit_scaler=False):
        """Create ensemble of multiple workflows.

        Create an (optimal) ensemble of a combination of hyperparameter settings
        and the associated groupsels, PCAs, estimators etc.

        Based on Caruana et al. 2004, but a little different:

        1. Recreate the training/validation splits for a n-fold cross validation.
        2. For each fold:
            a. Start with an empty ensemble
            b. Create starting ensemble by adding N individually best performing
               models on the validation set. N is tuned on the validation set.
            c. Add model that improves ensemble performance on validation set the most, with replacement.
            d. Repeat (c) untill performance does not increase

        The performance metric is the same as for the original hyperparameter
        search, i.e. probably the F1-score for classification and r2-score
        for regression. However, we recommend using the SAR score, as this is
        more universal.

        Method: top50 or Caruana

        """
        # Define a function for scoring the performance of a classifier
        def compute_performance(scoring, Y_valid_truth, Y_valid_score):
            if scoring == 'f1_weighted':
                # Convert score to binaries first
                for num in range(0, len(Y_valid_score)):
                    if Y_valid_score[num] >= 0.5:
                        Y_valid_score[num] = 1
                    else:
                        Y_valid_score[num] = 0

                perf = f1_score(Y_valid_truth, Y_valid_score, average='weighted')
            elif scoring == 'f1':
                # Convert score to binaries first
                for num in range(0, len(Y_valid_score)):
                    if Y_valid_score[num] >= 0.5:
                        Y_valid_score[num] = 1
                    else:
                        Y_valid_score[num] = 0

                perf = f1_score(Y_valid_truth, Y_valid_score, average='macro')
            elif scoring == 'auc':
                perf = roc_auc_score(Y_valid_truth, Y_valid_score)
            elif scoring == 'sar':
                perf = sar_score(Y_valid_truth, Y_valid_score)
            else:
                raise KeyError('[WORC Warning] No valid score method given in ensembling: ' + str(scoring))

            return perf

        if verbose is None:
            verbose = self.verbose

        if scoring is None:
            scoring = self.scoring

        # Get settings for best 100 estimators
        parameters_all = self.cv_results_['params']
        n_classifiers = len(parameters_all)
        n_iter = len(self.cv_iter)

        # Create a new base object for the ensemble components
        if type(self) == RandomizedSearchCVfastr:
            base_estimator = RandomizedSearchCVfastr()
        elif type(self) == RandomizedSearchCVJoblib:
            base_estimator = RandomizedSearchCVJoblib()

        if type(method) is int:
            # Simply take the top50 best hyperparameters
            if verbose:
                print(f'Creating ensemble using top {str(method)} individual classifiers.')
            if method == 1:
                # Next functions expect list
                ensemble = [0]
            else:
                ensemble = range(0, method)

        elif method == 'FitNumber':
            # Use optimum number of models

            # In order to speed up the process, we precompute all scores of the possible
            # classifiers in all cross validation estimatons

            # Create the training and validation set scores
            if verbose:
                print('Precomputing scores on training and validation set.')
            Y_valid_score = list()
            Y_valid_truth = list()
            performances = np.zeros((n_iter, n_classifiers))
            for it, (train, valid) in enumerate(self.cv_iter):
                if verbose:
                    print(f' - iteration {it + 1} / {n_iter}.')
                Y_valid_score_it = np.zeros((n_classifiers, len(valid)))

                # Loop over the 100 best estimators
                for num, p_all in enumerate(parameters_all):
                    # NOTE: Explicitly exclude validation set, elso refit and score
                    # somehow still seems to use it.
                    X_train_temp = [X_train[i] for i in train]
                    Y_train_temp = [Y_train[i] for i in train]
                    train_temp = np.arange(0, len(train))

                    # Refit a SearchCV object with the provided parameters
                    base_estimator.refit_and_score(X_train_temp, Y_train_temp, p_all,
                                                   train_temp, train_temp,
                                                   verbose=False)

                    # Predict and save scores
                    X_train_values = [x[0] for x in X_train] # Throw away labels
                    X_train_values_valid = [X_train_values[i] for i in valid]
                    Y_valid_score_temp = base_estimator.predict_proba(X_train_values_valid)

                    # Only take the probabilities for the second class
                    Y_valid_score_temp = Y_valid_score_temp[:, 1]

                    # Append to array for all classifiers on this validation set
                    Y_valid_score_it[num, :] = Y_valid_score_temp

                    if num == 0:
                        # Also store the validation ground truths
                        Y_valid_truth.append(Y_train[valid])

                    performances[it, num] = compute_performance(scoring,
                                                                Y_train[valid],
                                                                Y_valid_score_temp)

                Y_valid_score.append(Y_valid_score_it)

            # Sorted Ensemble Initialization -------------------------------------
            # Go on adding to the ensemble untill we find the optimal performance
            # Initialize variables

            # Note: doing this in a greedy way doesnt work. We compute the
            # performances for the ensembles of lengt [1, n_classifiers] and
            # select the optimum
            best_performance = 0
            new_performance = 0.001
            iteration = 0
            ensemble = list()
            y_score = [None]*n_iter
            best_index = 0
            single_estimator_performance = new_performance

            if initialize:
                # Rank the models based on scoring on the validation set
                performances = np.mean(performances, axis=0)
                sortedindices = np.argsort(performances)[::-1]
                performances_n_class = list()

                if verbose:
                    print("\n")
                    print('Sorted Ensemble Initialization.')
                # while new_performance > best_performance:
                for dummy in range(0, n_classifiers):
                    # Score is better, so expand ensemble and replace new best score
                    best_performance = new_performance

                    if iteration > 1:
                        # Stack scores: not needed for first iteration
                        ensemble.append(best_index)
                        # N_models += 1
                        for num in range(0, n_iter):
                            y_score[num] = np.vstack((y_score[num], Y_valid_score[num][ensemble[-1], :]))

                    elif iteration == 1:
                        # Create y_score object for second iteration
                        single_estimator_performance = new_performance
                        ensemble.append(best_index)
                        # N_models += 1
                        for num in range(0, n_iter):
                            y_score[num] = Y_valid_score[num][ensemble[-1], :]

                    # Perform n-fold cross validation to estimate performance of next best classifier
                    performances_temp = np.zeros((n_iter))
                    for n_crossval in range(0, n_iter):
                        # For each estimator, add the score to the ensemble and new ensemble performance
                        if iteration == 0:
                            # No y_score yet, so we need to build it instead of stacking
                            y_valid_score_new = Y_valid_score[n_crossval][sortedindices[iteration], :]
                        else:
                            # Stack scores of added model on top of previous scores and average
                            y_valid_score_new = np.mean(np.vstack((y_score[n_crossval], Y_valid_score[n_crossval][sortedindices[iteration], :])), axis=0)

                        perf = compute_performance(scoring, Y_valid_truth[n_crossval], y_valid_score_new)
                        performances_temp[n_crossval] = perf

                    # Check which ensemble should be in the ensemble to maximally improve
                    new_performance = np.mean(performances_temp)
                    performances_n_class.append(new_performance)
                    best_index = sortedindices[iteration]
                    iteration += 1

                # Select N_models for initialization
                new_performance = max(performances_n_class)
                N_models = performances_n_class.index(new_performance) + 1  # +1 due to python indexing
                ensemble = ensemble[0:N_models]
                best_performance = new_performance

                # Print the performance gain
                print(f"Ensembling best {scoring}: {best_performance}.")
                print(f"Single estimator best {scoring}: {single_estimator_performance}.")
                print(f'Ensemble consists of {len(ensemble)} estimators {ensemble}.')

        elif method == 'Caruana':
            # Use the method from Caruana
            if verbose:
                print('Creating ensemble with Caruana method.')

            # In order to speed up the process, we precompute all scores of the possible
            # classifiers in all cross validation estimatons

            # Create the training and validation set scores
            if verbose:
                print('Precomputing scores on training and validation set.')
            Y_valid_score = list()
            Y_valid_truth = list()
            performances = np.zeros((n_iter, n_classifiers))
            for it, (train, valid) in enumerate(self.cv_iter):
                if verbose:
                    print(f' - iteration {it + 1} / {n_iter}.')
                Y_valid_score_it = np.zeros((n_classifiers, len(valid)))

                # Loop over the 100 best estimators
                for num, p_all in enumerate(parameters_all):
                    # NOTE: Explicitly exclude validation set, elso refit and score
                    # somehow still seems to use it.
                    X_train_temp = [X_train[i] for i in train]
                    Y_train_temp = [Y_train[i] for i in train]
                    train_temp = np.arange(0, len(train))

                    # Refit a SearchCV object with the provided parameters
                    base_estimator.refit_and_score(X_train_temp, Y_train_temp, p_all,
                                                   train_temp, train_temp,
                                                   verbose=False)

                    # Predict and save scores
                    X_train_values = [x[0] for x in X_train] # Throw away labels
                    X_train_values_valid = [X_train_values[i] for i in valid]
                    Y_valid_score_temp = base_estimator.predict_proba(X_train_values_valid)

                    # Only take the probabilities for the second class
                    Y_valid_score_temp = Y_valid_score_temp[:, 1]

                    # Append to array for all classifiers on this validation set
                    Y_valid_score_it[num, :] = Y_valid_score_temp

                    if num == 0:
                        # Also store the validation ground truths
                        Y_valid_truth.append(Y_train[valid])

                    performances[it, num] = compute_performance(scoring,
                                                                Y_train[valid],
                                                                Y_valid_score_temp)

                Y_valid_score.append(Y_valid_score_it)

            # Sorted Ensemble Initialization -------------------------------------
            # Go on adding to the ensemble untill we find the optimal performance
            # Initialize variables

            # Note: doing this in a greedy way doesnt work. We compute the
            # performances for the ensembles of lengt [1, n_classifiers] and
            # select the optimum
            best_performance = 0
            new_performance = 0.001
            iteration = 0
            ensemble = list()
            y_score = [None]*n_iter
            best_index = 0
            single_estimator_performance = new_performance

            if initialize:
                # Rank the models based on scoring on the validation set
                performances = np.mean(performances, axis=0)
                sortedindices = np.argsort(performances)[::-1]
                performances_n_class = list()

                if verbose:
                    print("\n")
                    print('Sorted Ensemble Initialization.')
                # while new_performance > best_performance:
                for dummy in range(0, n_classifiers):
                    # Score is better, so expand ensemble and replace new best score
                    best_performance = new_performance

                    if iteration > 1:
                        # Stack scores: not needed for first iteration
                        ensemble.append(best_index)
                        # N_models += 1
                        for num in range(0, n_iter):
                            y_score[num] = np.vstack((y_score[num], Y_valid_score[num][ensemble[-1], :]))

                    elif iteration == 1:
                        # Create y_score object for second iteration
                        single_estimator_performance = new_performance
                        ensemble.append(best_index)
                        # N_models += 1
                        for num in range(0, n_iter):
                            y_score[num] = Y_valid_score[num][ensemble[-1], :]

                    # Perform n-fold cross validation to estimate performance of next best classifier
                    performances_temp = np.zeros((n_iter))
                    for n_crossval in range(0, n_iter):
                        # For each estimator, add the score to the ensemble and new ensemble performance
                        if iteration == 0:
                            # No y_score yet, so we need to build it instead of stacking
                            y_valid_score_new = Y_valid_score[n_crossval][sortedindices[iteration], :]
                        else:
                            # Stack scores of added model on top of previous scores and average
                            y_valid_score_new = np.mean(np.vstack((y_score[n_crossval], Y_valid_score[n_crossval][sortedindices[iteration], :])), axis=0)

                        perf = compute_performance(scoring, Y_valid_truth[n_crossval], y_valid_score_new)
                        performances_temp[n_crossval] = perf

                    # Check which ensemble should be in the ensemble to maximally improve
                    new_performance = np.mean(performances_temp)
                    performances_n_class.append(new_performance)
                    best_index = sortedindices[iteration]
                    iteration += 1

                # Select N_models for initialization
                new_performance = max(performances_n_class)
                N_models = performances_n_class.index(new_performance) + 1  # +1 due to python indexing
                ensemble = ensemble[0:N_models]
                best_performance = new_performance

                # Print the performance gain
                print(f"Ensembling best {scoring}: {best_performance}.")
                print(f"Single estimator best {scoring}: {single_estimator_performance}.")
                print(f'Ensemble consists of {len(ensemble)} estimators {ensemble}.')

            # Greedy selection  -----------------------------------------------
            # Initialize variables
            best_performance -= 1e-10
            iteration = 0

            # Go on adding to the ensemble untill we find the optimal performance
            if verbose:
                print("\n")
                print('Greedy selection.')
            while new_performance > best_performance:
                # Score is better, so expand ensemble and replace new best score
                if verbose:
                    print(f"Iteration: {iteration}, best {scoring}: {new_performance}.")
                best_performance = new_performance

                if iteration > 1:
                    # Stack scores: not needed for first iteration
                    ensemble.append(best_index)
                    for num in range(0, n_iter):
                        y_score[num] = np.vstack((y_score[num], Y_valid_score[num][ensemble[-1], :]))

                elif iteration == 1:
                    if not initialize:
                        # Create y_score object for second iteration
                        single_estimator_performance = new_performance
                        ensemble.append(best_index)
                        for num in range(0, n_iter):
                            y_score[num] = Y_valid_score[num][ensemble[-1], :]
                    else:
                        # Stack scores: not needed when ensemble initialization is already used
                        ensemble.append(best_index)
                        for num in range(0, n_iter):
                            y_score[num] = np.vstack((y_score[num], Y_valid_score[num][ensemble[-1], :]))

                # Perform n-fold cross validation to estimate performance of each possible addition to ensemble
                performances_temp = np.zeros((n_iter, n_classifiers))
                for n_crossval in range(0, n_iter):
                    # For each estimator, add the score to the ensemble and new ensemble performance
                    for n_estimator in range(0, n_classifiers):
                        if iteration == 0:
                            # No y_score yet, so we need to build it instead of stacking
                            y_valid_score_new = Y_valid_score[n_crossval][n_estimator, :]
                        else:
                            # Stack scores of added model on top of previous scores and average
                            y_valid_score_new = np.mean(np.vstack((y_score[n_crossval], Y_valid_score[n_crossval][n_estimator, :])), axis=0)

                        perf = compute_performance(scoring, Y_valid_truth[n_crossval], y_valid_score_new)
                        performances_temp[n_crossval, n_estimator] = perf

                # Average performances over crossval
                performances_temp = list(np.mean(performances_temp, axis=0))

                # Check which ensemble should be in the ensemble to maximally improve
                new_performance = max(performances_temp)
                best_index = performances_temp.index(new_performance)
                iteration += 1

            # Print the performance gain
            print(f"Ensembling best {scoring}: {best_performance}.")
            print(f"Single estimator best {scoring}: {single_estimator_performance}.")
            print(f'Ensemble consists of {len(ensemble)} estimators {ensemble}.')
        else:
            print(f'[WORC WARNING] No valid ensemble method given: {method}. Not ensembling')
            return self

        # Create the ensemble --------------------------------------------------
        train = np.arange(0, len(X_train))
        if self.fitted_workflows:
            # Simply select the required estimators
            print('\t - Detected already fitted workflows.')
            estimators = list()
            for i in ensemble:
                try:
                    # Try a prediction to see if estimator is truly fitted
                    self.fitted_workflows[i].predict(np.asarray([X_train[0][0], X_train[1][0]]))
                    estimators.append(self.fitted_workflows[i])
                except (NotFittedError, ValueError):
                    print(f'\t\t - Estimator {i} not fitted (correctly) yet, refit.')
                    estimator = self.fitted_workflows[i]
                    estimator.refit_and_score(X_train, Y_train,
                                              parameters_all[i],
                                              train, train,
                                              verbose=False)
                    estimators.append(estimator)
        else:
            # Create the ensemble trained on the full training set
            parameters_all = [parameters_all[i] for i in ensemble]
            estimators = list()
            nest = len(ensemble)
            for enum, p_all in enumerate(parameters_all):
                # Refit a SearchCV object with the provided parameters
                print(f"Refitting estimator {enum+1} / {nest}.")
                base_estimator = clone(base_estimator)

                # # Check if we need to create a multiclass estimator
                # if Y_train.shape[1] > 1 and type(base_estimator) != RankedSVM:
                #     # Multiclass, hence employ a multiclass classifier for SVM
                #     base_estimator = OneVsRestClassifier(base_estimator)

                base_estimator.refit_and_score(X_train, Y_train, p_all,
                                               train, train,
                                               verbose=False)

                # Determine whether to overfit the feature scaling on the test set
                base_estimator.overfit_scaler = overfit_scaler

                estimators.append(base_estimator)

        self.ensemble = Ensemble(estimators)
        self.best_estimator_ = self.ensemble
        print("\n")


class BaseSearchCVfastr(BaseSearchCV):
    """Base class for hyper parameter search with cross-validation."""

    def _fit(self, X, y, groups, parameter_iterable):
        """Actual fitting,  performing the search over parameters."""
        regressors = ['SVR', 'RFR', 'SGDR', 'Lasso', 'ElasticNet']
        isclassifier =\
            not any(clf in regressors for clf in self.param_distributions['classifiers'])

        # Check the cross-validation object and do the splitting
        cv = check_cv(self.cv, y, classifier=isclassifier)

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)
        if self.verbose > 0 and isinstance(parameter_iterable, Sized):
            n_candidates = len(parameter_iterable)
            print(f"Fitting {n_splits} folds for each of {n_candidates} candidates, totalling {n_candidates * n_splits} fits.")

        cv_iter = list(cv.split(X, y, groups))

        # NOTE: We do not check the scoring here, as this can differ
        # per estimator. Thus, this is done inside the fit and scoring

        # Check fitting parameters
        fit_params = _check_fit_params(X, self.fit_params)

        # Create temporary directory for fastr
        if DebugDetector().do_detection():
            # Specific name for easy debugging
            debugnum = 0
            name = 'DEBUG_' + str(debugnum)
            tempfolder = os.path.join(fastr.config.mounts['tmp'], 'GS', name)
            while os.path.exists(tempfolder):
                debugnum += 1
                name = 'DEBUG_' + str(debugnum)
                tempfolder = os.path.join(fastr.config.mounts['tmp'], 'GS', name)

        else:
            name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))

        tempfolder = os.path.join(fastr.config.mounts['tmp'], 'GS', name)
        if not os.path.exists(tempfolder):
            os.makedirs(tempfolder)

        # Draw parameter sample
        for num, parameters in enumerate(parameter_iterable):
            parameter_sample = parameters
            break

        # Preprocess features if required
        if 'FeatPreProcess' in parameter_sample:
            if parameter_sample['FeatPreProcess'] == 'True':
                print("Preprocessing features.")
                feature_values = np.asarray([x[0] for x in X])
                feature_labels = np.asarray([x[1] for x in X])
                preprocessor = Preprocessor(verbose=False)
                preprocessor.fit(feature_values, feature_labels=feature_labels[0, :])
                feature_values = preprocessor.transform(feature_values)
                feature_labels = preprocessor.transform(feature_labels)
                X = [(values, labels) for values, labels in zip(feature_values, feature_labels)]

        # Create the parameter files
        parameters_temp = dict()
        try:
            for num, parameters in enumerate(parameter_iterable):
                parameters["Number"] = str(num)
                parameters_temp[str(num)] = parameters

        except ValueError:
            # One of the parameters gives an error. Find out which one.
            param_grid = dict()
            for k, v in parameter_iterable.param_distributions.iteritems():
                param_grid[k] = v
                sampled_params = ParameterSampler(param_grid, 5)
                try:
                    for num, parameters in enumerate(sampled_params):
                        # Dummy operation
                        a = 1
                except ValueError:
                    break

            message = 'One or more of the values in your parameter sampler ' +\
                      'is either not iterable, or the distribution cannot ' +\
                      'generate valid samples. Please check your  ' +\
                      f' parameters. At least {k} gives an error.'
            raise WORCexceptions.WORCValueError(message)

        # Split the parameters files in equal parts
        keys = list(parameters_temp.keys())
        keys = chunks(keys, self.n_jobspercore)
        parameter_files = dict()
        for num, k in enumerate(keys):
            temp_dict = dict()
            for number in k:
                temp_dict[number] = parameters_temp[number]

            fname = f'settings_{num}.json'
            sourcename = os.path.join(tempfolder, 'parameters', fname)
            if not os.path.exists(os.path.dirname(sourcename)):
                os.makedirs(os.path.dirname(sourcename))
            with open(sourcename, 'w') as fp:
                json.dump(temp_dict, fp, indent=4)

            parameter_files[str(num).zfill(4)] =\
                f'vfs://tmp/GS/{name}/parameters/{fname}'

        # Create test-train splits
        traintest_files = dict()
        # TODO: ugly nummering solution
        num = 0
        for train, test in cv_iter:
            source_labels = ['train', 'test']

            source_data = pd.Series([train, test],
                                    index=source_labels,
                                    name='Train-test data')

            fname = f'traintest_{num}.hdf5'
            sourcename = os.path.join(tempfolder, 'traintest', fname)
            if not os.path.exists(os.path.dirname(sourcename)):
                os.makedirs(os.path.dirname(sourcename))
            traintest_files[str(num).zfill(4)] = f'vfs://tmp/GS/{name}/traintest/{fname}'

            sourcelabel = f"Source Data Iteration {num}"
            source_data.to_hdf(sourcename, sourcelabel)

            num += 1

        # Create the files containing the estimator and settings
        estimator_labels = ['X', 'y', 'scoring',
                            'verbose', 'fit_params', 'return_train_score',
                            'return_n_test_samples',
                            'return_times', 'return_parameters',
                            'return_estimator',
                            'error_score', 'return_all', 'refit_workflows']

        verbose = False
        return_n_test_samples = True
        return_times = True
        return_parameters = False
        return_estimator = False
        return_all = False
        estimator_data = pd.Series([X, y, self.scoring,
                                    verbose, fit_params,
                                    self.return_train_score,
                                    return_n_test_samples, return_times,
                                    return_parameters,
                                    return_estimator,
                                    self.error_score,
                                    return_all, self.refit_workflows],
                                   index=estimator_labels,
                                   name='estimator Data')
        fname = 'estimatordata.hdf5'
        estimatorname = os.path.join(tempfolder, fname)
        estimator_data.to_hdf(estimatorname, 'Estimator Data')

        estimatordata = f"vfs://tmp/GS/{name}/{fname}"

        # Create the fastr network
        network = fastr.create_network('WORC_GridSearch_' + name)
        estimator_data = network.create_source('HDF5', id='estimator_source')
        traintest_data = network.create_source('HDF5', id='traintest')
        parameter_data = network.create_source('JsonFile', id='parameters')
        sink_output = network.create_sink('HDF5', id='output')

        fitandscore =\
            network.create_node('worc/fitandscore:1.0',
                                tool_version='1.0',
                                id='fitandscore',
                                resources=ResourceLimit(memory=self.memory))

        fitandscore.inputs['estimatordata'].input_group = 'estimator'
        fitandscore.inputs['traintest'].input_group = 'traintest'
        fitandscore.inputs['parameters'].input_group = 'parameters'

        fitandscore.inputs['estimatordata'] = estimator_data.output
        fitandscore.inputs['traintest'] = traintest_data.output
        fitandscore.inputs['parameters'] = parameter_data.output
        sink_output.input = fitandscore.outputs['fittedestimator']

        source_data = {'estimator_source': estimatordata,
                       'traintest': traintest_files,
                       'parameters': parameter_files}
        sink_data = {'output': f"vfs://tmp/GS/{name}/output_{{sample_id}}_{{cardinality}}{{ext}}"}

        network.execute(source_data, sink_data,
                        tmpdir=os.path.join(tempfolder, 'tmp'),
                        execution_plugin=self.fastr_plugin)

        # Check whether all jobs have finished
        expected_no_files = len(list(traintest_files.keys())) * len(list(parameter_files.keys()))
        sink_files = glob.glob(os.path.join(fastr.config.mounts['tmp'], 'GS', name) + '/output*.hdf5')
        sink_files.sort()
        if len(sink_files) != expected_no_files:
            difference = expected_no_files - len(sink_files)
            fname = os.path.join(tempfolder, 'tmp')
            message = ('Fitting classifiers has failed for ' +
                       f'{difference} / {expected_no_files} files. The temporary ' +
                       f'results where not deleted and can be found in {tempfolder}. ' +
                       'Probably your fitting and scoring failed: check out ' +
                       'the tmp/fitandscore folder within the tempfolder for ' +
                       'the fastr job temporary results or run: fastr trace ' +
                       f'"{fname}{os.path.sep}__sink_data__.json" --samples.')
            raise WORCexceptions.WORCValueError(message)

        # Read in the output data once finished
        save_data = list()
        for output in sink_files:
            data = pd.read_hdf(output)
            save_data.extend(list(data['RET']))

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            if self.refit_workflows:
                (train_scores, test_scores, test_sample_counts,
                 fit_time, score_time, parameters_all, fitted_workflows) =\
                  zip(*save_data)
            else:
                fitted_workflows = None
                (train_scores, test_scores, test_sample_counts,
                 fit_time, score_time, parameters_all) =\
                    zip(*save_data)
        else:
            if self.refit_workflows:
                (test_scores, test_sample_counts,
                 fit_time, score_time, parameters_all, fitted_workflows) =\
                  zip(*save_data)
            else:
                fitted_workflows = None
                (test_scores, test_sample_counts,
                 fit_time, score_time, parameters_all) =\
                    zip(*save_data)

        # Remove the temporary folder used
        if name != 'DEBUG_0':
            # Do delete if not debugging for first iteration
            shutil.rmtree(tempfolder)

        # Process the results of the fitting procedure
        self.process_fit(n_splits=n_splits,
                         parameters_all=parameters_all,
                         test_sample_counts=test_sample_counts,
                         test_score_dicts=test_scores,
                         train_score_dicts=train_scores,
                         fit_time=fit_time,
                         score_time=score_time,
                         cv_iter=cv_iter,
                         X=X, y=y,
                         fitted_workflows=fitted_workflows)


class RandomizedSearchCVfastr(BaseSearchCVfastr):
    """Randomized search on hyper parameters.

    RandomizedSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Read more in the sklearn user guide.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer the sklearn user guide for the various
        cross-validation strategies that can be used here.

    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this RandomizedSearchCV instance after fitting.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    random_state : int or RandomState
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : boolean, default=True
        If ``'False'``, the ``cv_results_`` attribute will not include training
        scores.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |        0.8        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |        0.9        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |        0.7        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.8, 0.9, 0.7],
            'split1_test_score'  : [0.82, 0.5, 0.7],
            'mean_test_score'    : [0.81, 0.7, 0.7],
            'std_test_score'     : [0.02, 0.2, 0.],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.8, 0.9, 0.7],
            'split1_train_score' : [0.82, 0.5, 0.7],
            'mean_train_score'   : [0.81, 0.7, 0.7],
            'std_train_score'    : [0.03, 0.03, 0.04],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
            'params' : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE that the key ``'params'`` is used to store a list of parameter
        settings dict for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.

    best_score_ : float
        Score of best_estimator on the left out data.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    --------
    :class:`GridSearchCV`:
        Does exhaustive search over a grid of parameters.

    :class:`ParameterSampler`:
        A generator over parameter settings, constructed from
        param_distributions.

    """

    def __init__(self, param_distributions={}, n_iter=10, scoring=None,
                 fit_params=None, n_jobs=1, iid=True, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise', return_train_score=True,
                 n_jobspercore=100, fastr_plugin=None, memory='2G', maxlen=100,
                 ranking_score='test_score', refit_workflows=False):
        super(RandomizedSearchCVfastr, self).__init__(
             param_distributions=param_distributions, scoring=scoring, fit_params=fit_params,
             n_iter=n_iter, random_state=random_state, n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
             pre_dispatch=pre_dispatch, error_score=error_score,
             return_train_score=return_train_score,
             n_jobspercore=n_jobspercore, fastr_plugin=fastr_plugin,
             memory=memory, maxlen=maxlen, ranking_score=ranking_score,
             refit_workflows=refit_workflows)

    def fit(self, X, y=None, groups=None):
        """Randomized model selection and hyperparameter search.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        """
        print("Fit: " + str(self.n_iter))
        sampled_params = ParameterSampler(self.param_distributions,
                                          self.n_iter,
                                          random_state=self.random_state)
        return self._fit(X, y, groups, sampled_params)


class BaseSearchCVJoblib(BaseSearchCV):
    """Base class for hyper parameter search with cross-validation."""

    def _fit(self, X, y, groups, parameter_iterable):
        """Actual fitting,  performing the search over parameters."""

        regressors = ['SVR', 'RFR', 'SGDR', 'Lasso', 'ElasticNet']
        isclassifier =\
            not any(clf in regressors for clf in self.param_distributions['classifiers'])

        # Check the cross-validation object and do the splitting
        cv = check_cv(self.cv, y, classifier=isclassifier)

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)
        if self.verbose > 0 and isinstance(parameter_iterable, Sized):
            n_candidates = len(parameter_iterable)
            print(f"Fitting {n_splits} folds for each of {n_candidates}" +\
                  " candidates, totalling" +\
                  " {n_candidates * n_splits} fits")

        pre_dispatch = self.pre_dispatch
        cv_iter = list(cv.split(X, y, groups))

        # Check fitting parameters
        fit_params = _check_fit_params(X, self.fit_params)

        # Draw parameter sample
        for num, parameters in enumerate(parameter_iterable):
            parameter_sample = parameters
            break

        # Preprocess features if required
        if 'FeatPreProcess' in parameter_sample:
            if parameter_sample['FeatPreProcess'] == 'True':
                print("Preprocessing features.")
                feature_values = np.asarray([x[0] for x in X])
                feature_labels = np.asarray([x[1] for x in X])
                preprocessor = Preprocessor(verbose=False)
                preprocessor.fit(feature_values, feature_labels=feature_labels[0, :])
                feature_values = preprocessor.transform(feature_values)
                feature_labels = preprocessor.transform(feature_labels)
                X = [(values, labels) for values, labels in zip(feature_values, feature_labels)]

        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=pre_dispatch
        )(delayed(fit_and_score)(X, y, self.scoring,
                                 train, test, parameters,
                                 fit_params=fit_params,
                                 return_train_score=self.return_train_score,
                                 return_n_test_samples=True,
                                 return_times=True, return_parameters=False,
                                 return_estimator=False,
                                 error_score=self.error_score,
                                 verbose=False,
                                 return_all=False)
          for parameters in parameter_iterable
          for train, test in cv_iter)
        save_data = zip(*out)

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            (train_scores, test_scores, test_sample_counts,
             fit_time, score_time, parameters_all) =\
              save_data
        else:
            (test_scores, test_sample_counts,
             fit_time, score_time, parameters_all) =\
              save_data

        self.process_fit(n_splits=n_splits,
                         parameters_all=parameters_all,
                         test_sample_counts=test_sample_counts,
                         test_score_dicts=test_scores,
                         train_score_dicts=train_scores,
                         fit_time=fit_time,
                         score_time=score_time,
                         cv_iter=cv_iter,
                         X=X, y=y)

        return self


class GridSearchCVfastr(BaseSearchCVfastr):
    """Exhaustive search over specified parameter values for an estimator.

    Important members are fit, predict.

    GridSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.

    Read more in the sklearn user guide.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer the sklearn user guide for the various
        cross-validation strategies that can be used here.

    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this GridSearchCV instance after fitting.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : boolean, default=True
        If ``'False'``, the ``cv_results_`` attribute will not include training
        scores.


    Examples
    --------
    >>> from sklearn import svm, datasets
    >>> from sklearn.model_selection import GridSearchCV
    >>> iris = datasets.load_iris()
    >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    >>> svr = svm.SVC()
    >>> clf = GridSearchCV(svr, parameters)
    >>> clf.fit(iris.data, iris.target)
    ...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    GridSearchCV(cv=None, error_score=...,
           estimator=SVC(C=1.0, cache_size=..., class_weight=..., coef0=...,
                         decision_function_shape=None, degree=..., gamma=...,
                         kernel='rbf', max_iter=-1, probability=False,
                         random_state=None, shrinking=True, tol=...,
                         verbose=False),
           fit_params={}, iid=..., n_jobs=1,
           param_grid=..., pre_dispatch=..., refit=..., return_train_score=...,
           scoring=..., verbose=...)
    >>> sorted(clf.cv_results_.keys())
    ...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ['mean_fit_time', 'mean_score_time', 'mean_test_score',...
     'mean_train_score', 'param_C', 'param_kernel', 'params',...
     'rank_test_score', 'split0_test_score',...
     'split0_train_score', 'split1_test_score', 'split1_train_score',...
     'split2_test_score', 'split2_train_score',...
     'std_fit_time', 'std_score_time', 'std_test_score', 'std_train_score'...]

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +------------+-----------+------------+-----------------+---+---------+
        |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_....|
        +============+===========+============+=================+===+=========+
        |  'poly'    |     --    |      2     |        0.8      |...|    2    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'poly'    |     --    |      3     |        0.7      |...|    4    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.1   |     --     |        0.8      |...|    3    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.2   |     --     |        0.9      |...|    1    |
        +------------+-----------+------------+-----------------+---+---------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...)
            'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                        mask = [ True  True False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.8, 0.7, 0.8, 0.9],
            'split1_test_score'  : [0.82, 0.5, 0.7, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.82],
            'std_test_score'     : [0.02, 0.01, 0.03, 0.03],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.8, 0.9, 0.7],
            'split1_train_score' : [0.82, 0.5, 0.7],
            'mean_train_score'   : [0.81, 0.7, 0.7],
            'std_train_score'    : [0.03, 0.03, 0.04],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }

        NOTE that the key ``'params'`` is used to store a list of parameter
        settings dict for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.

    best_score_ : float
        Score of best_estimator on the left out data.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    Notes
    ------
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    ---------
    :class:`ParameterGrid`:
        generates all the combinations of a hyperparameter grid.

    :func:`sklearn.model_selection.train_test_split`:
        utility function to split the data into a development set usable
        for fitting a GridSearchCV instance and an evaluation set for
        its final evaluation.

    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.

    """

    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise',
                 return_train_score=True):
        super(GridSearchCVfastr, self).__init__(
            scoring=scoring, fit_params=fit_params,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score, fastr_plugin=None,
            memory='2G')
        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def fit(self, X, y=None, groups=None):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        """
        return self._fit(X, y, groups, ParameterGrid(self.param_grid))


class RandomizedSearchCVJoblib(BaseSearchCVJoblib):
    """Randomized search on hyper parameters.

    RandomizedSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Read more in the sklearn user guide.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer sklearn user guide for the various
        cross-validation strategies that can be used here.

    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this RandomizedSearchCV instance after fitting.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    random_state : int or RandomState
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : boolean, default=True
        If ``'False'``, the ``cv_results_`` attribute will not include training
        scores.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |        0.8        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |        0.9        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |        0.7        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.8, 0.9, 0.7],
            'split1_test_score'  : [0.82, 0.5, 0.7],
            'mean_test_score'    : [0.81, 0.7, 0.7],
            'std_test_score'     : [0.02, 0.2, 0.],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.8, 0.9, 0.7],
            'split1_train_score' : [0.82, 0.5, 0.7],
            'mean_train_score'   : [0.81, 0.7, 0.7],
            'std_train_score'    : [0.03, 0.03, 0.04],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
            'params' : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE that the key ``'params'`` is used to store a list of parameter
        settings dict for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.

    best_score_ : float
        Score of best_estimator on the left out data.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    --------
    :class:`GridSearchCV`:
        Does exhaustive search over a grid of parameters.

    :class:`ParameterSampler`:
        A generator over parameter settins, constructed from
        param_distributions.

    """

    def __init__(self, param_distributions={}, n_iter=10, scoring=None,
                 fit_params=None, n_jobs=1, iid=True, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise', return_train_score=True,
                 n_jobspercore=100, maxlen=100, ranking_score='test_score'):
        super(RandomizedSearchCVJoblib, self).__init__(
             param_distributions=param_distributions,
             n_iter=n_iter, scoring=scoring, fit_params=fit_params,
             n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
             pre_dispatch=pre_dispatch, error_score=error_score,
             return_train_score=return_train_score,
             n_jobspercore=n_jobspercore, random_state=random_state,
             maxlen=maxlen, ranking_score=ranking_score)

    def fit(self, X, y=None, groups=None):
        """Run fit on the estimator with randomly drawn parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        """
        sampled_params = ParameterSampler(self.param_distributions,
                                          self.n_iter,
                                          random_state=self.random_state)
        return self._fit(X, y, groups, sampled_params)


class GridSearchCVJoblib(BaseSearchCVJoblib):
    """Exhaustive search over specified parameter values for an estimator.

    Important members are fit, predict.

    GridSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.

    Read more in the sklearn user guide.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer sklearn user guide for the various
        cross-validation strategies that can be used here.

    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this GridSearchCV instance after fitting.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : boolean, default=True
        If ``'False'``, the ``cv_results_`` attribute will not include training
        scores.


    Examples
    --------
    >>> from sklearn import svm, datasets
    >>> from sklearn.model_selection import GridSearchCV
    >>> iris = datasets.load_iris()
    >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    >>> svr = svm.SVC()
    >>> clf = GridSearchCV(svr, parameters)
    >>> clf.fit(iris.data, iris.target)
    ...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    GridSearchCV(cv=None, error_score=...,
           estimator=SVC(C=1.0, cache_size=..., class_weight=..., coef0=...,
                         decision_function_shape=None, degree=..., gamma=...,
                         kernel='rbf', max_iter=-1, probability=False,
                         random_state=None, shrinking=True, tol=...,
                         verbose=False),
           fit_params={}, iid=..., n_jobs=1,
           param_grid=..., pre_dispatch=..., refit=..., return_train_score=...,
           scoring=..., verbose=...)
    >>> sorted(clf.cv_results_.keys())
    ...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ['mean_fit_time', 'mean_score_time', 'mean_test_score',...
     'mean_train_score', 'param_C', 'param_kernel', 'params',...
     'rank_test_score', 'split0_test_score',...
     'split0_train_score', 'split1_test_score', 'split1_train_score',...
     'split2_test_score', 'split2_train_score',...
     'std_fit_time', 'std_score_time', 'std_test_score', 'std_train_score'...]

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +------------+-----------+------------+-----------------+---+---------+
        |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_....|
        +============+===========+============+=================+===+=========+
        |  'poly'    |     --    |      2     |        0.8      |...|    2    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'poly'    |     --    |      3     |        0.7      |...|    4    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.1   |     --     |        0.8      |...|    3    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.2   |     --     |        0.9      |...|    1    |
        +------------+-----------+------------+-----------------+---+---------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...)
            'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                        mask = [ True  True False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.8, 0.7, 0.8, 0.9],
            'split1_test_score'  : [0.82, 0.5, 0.7, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.82],
            'std_test_score'     : [0.02, 0.01, 0.03, 0.03],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.8, 0.9, 0.7],
            'split1_train_score' : [0.82, 0.5, 0.7],
            'mean_train_score'   : [0.81, 0.7, 0.7],
            'std_train_score'    : [0.03, 0.03, 0.04],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }

        NOTE that the key ``'params'`` is used to store a list of parameter
        settings dict for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.

    best_score_ : float
        Score of best_estimator on the left out data.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    Notes
    ------
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    ---------
    :class:`ParameterGrid`:
        generates all the combinations of a hyperparameter grid.

    :func:`sklearn.model_selection.train_test_split`:
        utility function to split the data into a development set usable
        for fitting a GridSearchCV instance and an evaluation set for
        its final evaluation.

    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.

    """

    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise',
                 return_train_score=True):
        super(GridSearchCVJoblib, self).__init__(
            scoring=scoring, fit_params=fit_params,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def fit(self, X, y=None, groups=None):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        """
        return self._fit(X, y, groups, ParameterGrid(self.param_grid))
