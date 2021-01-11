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

from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
import numpy as np
import WORC.addexceptions as ae


class VarianceThresholdMean(BaseEstimator, SelectorMixin):
    '''
    Select features based on variance among objects. Similar to VarianceThreshold
    from sklearn, but does take the mean of the feature into account.
    '''
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, image_features):
        selectrows = list()
        means = np.mean(image_features, axis=0)
        variances = np.var(image_features, axis=0)

        for i in range(image_features.shape[1]):
            if variances[i] > self.threshold*(1-self.threshold)*means[i]:
                selectrows.append(i)

        self.selectrows = selectrows
        return self

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
        # NOTE: Method is required for the Selector class, but can be empty
        pass


def selfeat_variance(image_features, labels=None, thresh=0.99,
                     method='nomean'):
    '''
    Select features using a variance threshold.

    Parameters
    ----------
    image_features: numpy array, mandatory
            Array containing the feature values to apply the variance threshold
            selection on. The rows correspond to the patients, the column to the
            features.

    labels: numpy array, optional
            Array containing the labels of the corresponding features. Array
            should therefore have the same shape as the image_features array.

    thresh: float, default 0.99
            Threshold to be used as lower boundary for feature variance among
            patients.
    method: string, default nomean.
            Method to use for selection. Default: do not use the mean of the
            features. Other valid option is 'mean'.

    Returns
    ----------
    image_features: numpy array
            Transformed features array.

    labels: list or None
            When labels are given, returns the transformed labels. That object
            contains a list of all label names kept.

    sel: VarianceThreshold object
            The fitted variance threshold object.

    '''
    if method == 'nomean':
        sel = VarianceThreshold(threshold=thresh*(1 - thresh))
    elif method == 'mean':
        sel = VarianceThresholdMean(threshold=thresh*(1 - thresh))
    else:
        raise ae.PREDICTKeyError(('Invalid method {} given for ' +
                                  'VarianceThreshold feature selection. ' +
                                  'Should be "mean" or ' +
                                  '"nomean".').format(str(method)))

    sel = sel.fit(image_features)
    image_features = sel.transform(image_features)
    if labels is not None:
        labels = sel.transform(labels)

    return image_features, labels, sel
