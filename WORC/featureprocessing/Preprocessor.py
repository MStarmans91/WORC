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


class Preprocessor(object):
    """Module for feature preprocessing.

    Currently implemented:
        - Remove features with > 80% NaNs
    """

    def __init__(self, verbose=True):
        """Init preprocessor of features."""
        # initiate varables
        self.selectcolumns = list()
        self.verbose = verbose

    def fit(self, X, y=None, feature_labels=None):
        """Select columns with to many missing values (>80%)."""
        self.selectcolumns = list()
        nrows = float(X.shape[0])
        for column in range(0, X.shape[1]):
            nans = np.count_nonzero(np.isnan(X[:, column]))
            missing_percentage = float(nans) / nrows
            if missing_percentage > 0.80:
                if feature_labels is not None:
                    name = feature_labels[column]
                else:
                    name = column

                if self.verbose:
                    print(f'\t [WORC WARNING] More than 80% ({missing_percentage * 100.0}%) is missing for feature # {name}: removing.')

                continue
            else:
                self.selectcolumns.append(column)

    def transform(self, inputarray):
        """Transform feature array.

        Transform the inputarray to select only the features based on the
        result from the fit function.

        Parameters
        ----------
        inputarray: numpy array, mandatory
                Array containing the items to use selection on. The type of
                item in this list does not matter, e.g. floats, strings etc.

        """
        return np.asarray([np.asarray(x)[self.selectcolumns].tolist() for x in inputarray])
