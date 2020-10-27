#!/usr/bin/env python

# Copyright 2020 Biomedical Imaging Group Rotterdam, Departments of
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
from sklearn.preprocessing import OneHotEncoder


class OneHotEncoderWrapper(object):
    """Module for OneHotEncoding features."""

    def __init__(self, feature_labels_tofit, handle_unknown='ignore',
                 verbose=False):
        """Init preprocessor of features."""
        # Initiate varables
        self.handle_unknown = handle_unknown
        self.verbose = verbose
        self.feature_labels_tofit = feature_labels_tofit

    def fit(self, X, feature_labels, y=None):
        """Fit OneHotEncoder for labels in feature_labels."""
        self.selectcolumns = list()
        self.selectlabels = list()
        self.skipcolumns = list()
        for num, label in enumerate(feature_labels):
            if any(fl in label for fl in self.feature_labels_tofit):
                # This feature needs to be one hot encoded
                self.selectcolumns.append(num)
                self.selectlabels.append(label)
            else:
                # This feature needs to be skipped from onehotencoding
                self.skipcolumns.append(num)

        if self.verbose:
            print(f'\t Fitting one-hot-encoder for features {self.selectlabels}.')

        if len(self.selectcolumns) == 0:
            if self.verbose:
                print('\t No features selected, skip one-hot-encoding')
            self.encoder = None
            return

        # Gather skipped feature values and labels and selected ones
        skipped_feature_labels = list(np.asarray(feature_labels)[self.skipcolumns])

        select_feature_values = X[:, self.selectcolumns]
        select_feature_labels = list(np.asarray(feature_labels)[self.selectcolumns])

        # Apply the onehotencoding
        self.encoder = OneHotEncoder(handle_unknown=self.handle_unknown)
        self.encoder.fit(select_feature_values)

        # Adjust feature labels
        categories = self.encoder.categories_
        self.encoded_feature_labels = skipped_feature_labels
        for fl, cat in zip(select_feature_labels, categories):
            for c in range(cat.shape[0]):
                self.encoded_feature_labels.append(fl + f'_{c}')

        if self.verbose:
            print(f'\t Encoded feature labels: {self.encoded_feature_labels}.')

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
        if self.encoder is None:
            # No features encoded
            outputarray = inputarray
        else:
            # Gather skipped feature values and labels and selected ones
            skipped_feature_values = inputarray[:, self.skipcolumns]
            select_feature_values = inputarray[:, self.selectcolumns]

            # Transform selected features
            encoded_feature_values = self.encoder.transform(select_feature_values).toarray()

            # Recombine both
            outputarray = np.concatenate((skipped_feature_values,
                                         encoded_feature_values), axis=1)

        return outputarray


def test():
    """Test OneHotEncoderWrapper object."""
    # Objects
    X_train = np.asarray([['Male', 1, 5], ['Female', 3, 6], ['Female', 2, 7]])
    X_test = np.asarray([['Male', 2, 7], ['Unknown', 10, 10]])
    feature_labels = ['Gender', 'Numeric0', 'Numeric1']
    feature_labels_tofit = ['Gender', '0']

    # Fit and transform
    enc = OneHotEncoderWrapper(feature_labels_tofit=feature_labels_tofit,
                               verbose=True)
    enc.fit(X_train, feature_labels)
    X_train_encoded = enc.transform(X_train)
    X_test_encoded = enc.transform(X_test)

    # Print results
    print("X_train:")
    print(f"Input: {X_train}.")
    print(f"Output: {X_train_encoded}.")
    print("X_test:")
    print(f"Input: {X_test}.")
    print(f"Output: {X_test_encoded}.")
    print("Encoded feature labels:")
    print(enc.encoded_feature_labels)


if __name__ == "__main__":
    test()
