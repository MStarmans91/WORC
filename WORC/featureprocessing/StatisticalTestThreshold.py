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

from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
import numpy as np
from scipy.stats import ttest_ind, ranksums, mannwhitneyu


class StatisticalTestThreshold(BaseEstimator, SelectorMixin):
    '''
    Object to fit feature selection based on statistical tests.
    '''
    def __init__(self, metric='ttest', threshold=0.05):
        '''
        Parameters
        ----------
        metric: string, default 'ttest'
                Statistical test used for selection. Options are ttest,
                Welch, Wilcoxon, MannWhitneyU
        threshold: float, default 0.05
                Threshold for p-value in order for feature to be selected

        '''
        self.metric = metric
        self.threshold = threshold

    def fit(self, X_train, Y_train):
        '''
        Select only features specificed by the metric and threshold per patient.

        Parameters
        ----------
        X_train: numpy array, mandatory
                Array containing feature values used for model_selection.
                Number of objects on first axis, features on second axis.

        Y_train: numpy array, mandatory
                Array containing the binary labels for each object in X_train.
        '''

        self.selectrows = list()
        self.metric_values = list()

        # Set the metric function
        if self.metric == 'ttest':
            self.metric_function = ttest_ind
            self.parameters = {'equal_var': True}
        elif self.metric == 'Welch':
            self.metric_function = ttest_ind
            self.parameters = {'equal_var': False}
        elif self.metric == 'Wilcoxon':
            self.metric_function = ranksums
            self.parameters = {}
        elif self.metric == 'MannWhitneyU':
            self.metric_function = mannwhitneyu
            self.parameters = {}

        # Perform the statistical test for each feature
        multilabel = type(Y_train[0]) is np.ndarray
        for n_feat in range(0, X_train.shape[1]):
            # Select only this specific feature for all objects

            fv = X_train[:, n_feat]
            if multilabel:
                # print('Multilabel: take minimum p-value for all label tests.')
                # We do a statistical test per label and take the minimum p-value
                n_label = Y_train[0].shape[0]
                metric_values = list()
                for i in range(n_label):
                    class1 = [i for j, i in enumerate(fv) if np.argmax(Y_train[j]) == n_label]
                    class2 = [i for j, i in enumerate(fv) if np.argmax(Y_train[j]) != n_label]

                    try:
                        metric_value_temp = self.metric_function(class1, class2, **self.parameters)[1]
                    except ValueError as e:
                        print("[WORC Warning] " + str(e) + '. Replacing metric value by 1.')
                        metric_value_temp

                    metric_values.append(metric_value_temp)

                metric_value = np.min(metric_values)

            else:
                # Singlelabel
                class1 = [i for j, i in enumerate(fv) if Y_train[j] == 1]
                class2 = [i for j, i in enumerate(fv) if Y_train[j] == 0]

                try:
                    metric_value = self.metric_function(class1, class2, **self.parameters)[1]
                except ValueError as e:
                    print("[WORC Warning] " + str(e) + '. Replacing metric value by 1.')
                    metric_value = 1

            self.metric_values.append(metric_value)
            if metric_value < self.threshold:
                self.selectrows.append(n_feat)

    def transform(self, inputarray):
        '''
        Transform the inputarray to select only the features based on the
        result from the fit function.

        Parameters
        ----------
        inputarray: numpy array, mandatory
                Array containing the items to use selection on. The type of
                item in this list does not matter, e.g. floats, strings etc.
        '''
        return np.asarray([np.asarray(x)[self.selectrows].tolist() for x in inputarray])

    def _get_support_mask(self):
        # NOTE: metric is required for the Selector class, but can be empty
        pass
