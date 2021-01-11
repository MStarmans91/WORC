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

from sklearn.impute import SimpleImputer, KNNImputer


class Imputer(object):
        """Module for feature imputation."""

        def __init__(self, missing_values='nan', strategy='mean',
                     n_neighbors=5):
            '''
            Imputation of feature values using either sklearn, missingpy or
            (WIP) fancyimpute approaches.

            Parameters
            ----------
            missing_values : number, string, np.nan (default) or None
                The placeholder for the missing values. All occurrences of
                `missing_values` will be imputed.


            strategy : string, optional (default="mean")
                The imputation strategy.

                Supported using sklearn:
                - If "mean", then replace missing values using the mean along
                  each column. Can only be used with numeric data.
                - If "median", then replace missing values using the median along
                  each column. Can only be used with numeric data.
                - If "most_frequent", then replace missing using the most frequent
                  value along each column. Can be used with strings or numeric data.
                - If "constant", then replace missing values with fill_value. Can be
                  used with strings or numeric data.

                Supported using missingpy:
                - If 'knn', then use a nearest neighbor search. Can be
                  used with strings or numeric data.

                WIP: More strategies using fancyimpute

            n_neighbors : int, optional (default = 5)
                Number of neighboring samples to use for imputation if method
                is knn.

            '''

            # Set parameters to objects
            self.missing_values = missing_values
            self.strategy = strategy
            self.n_neighbors = n_neighbors

            # Depending on the imputations strategy, use a specific toolbox
            if strategy in ['mean', 'median', 'most_frequent', 'constant']:
                self.Imputer =\
                 SimpleImputer(missing_values=self.missing_values,
                               strategy=self.strategy)
            elif strategy == 'knn':
                if missing_values == 'nan':
                    # Slightly different API for missingpy
                    self.missing_values = 'NaN'
                self.Imputer = KNNImputer(missing_values=self.missing_values,
                                          n_neighbors=self.n_neighbors)

        def fit(self, X, y=None):
            self.Imputer.fit(X, y)

        def transform(self, X):
            return self.Imputer.transform(X)
