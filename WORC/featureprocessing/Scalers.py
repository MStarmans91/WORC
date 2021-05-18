#!/usr/bin/env python

# Copyright 2020 - 2021 Biomedical Imaging Group Rotterdam, Departments of
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

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.utils.validation import check_is_fitted
import numpy as np
import WORC.addexceptions as ae

accepted_scalers = ['robust_z_score', 'z_score', 'robust', 'minmax',
                    'log_z_score']


class WORCScaler(TransformerMixin, BaseEstimator):
    """Scale features using an sklearn scaler.

    Additionally, several features can be excluded. Mostly useful when using
    also categorical features such as patient sex.

    """

    def __init__(self, method='robust_z_score', skip_features=None,
                 verbose=False):
        """Initialize object.

        Parameters
        ------------
        method: string
            Name of scaler used: robust_z_score, z_score, robust, or minmax
        skip_features: list of strings
            If any of these elements occur as substring in a feature label,
            this feature is excluded.

        """
        self.method = method
        self.skip_features = skip_features
        self.verbose = verbose

        if method not in accepted_scalers:
            raise ae.WORCKeyError(f'{method} is not a ' +
                                  'valid scaling method. Should be any of ' +
                                  f'{accepted_scalers}.')

        self.included_feature_indices = list()
        self.excluded_feature_indices = list()

    def fit(self, X_train, feature_labels=None):
        """Fit the scaler."""
        # Determine whether features should be skipped
        if feature_labels is None or self.skip_features is None or not self.skip_features:
            # Nothing should be skipped
            X_train_scaling = X_train

            self.included_feature_indices = range(0, X_train.shape[1])
            self.excluded_feature_indices = list()

        else:
            # Skip part of features in scaling
            if self.verbose:
                print(f'\t Excluding features containing: {self.skip_features}')

            # Determine indices of excluded features
            included_feature_indices = []
            excluded_feature_indices = []
            for fnum, i in enumerate(feature_labels):
                if not any(e in i for e in self.skip_features):
                    included_feature_indices.append(fnum)
                else:
                    excluded_feature_indices.append(fnum)

            # Actually exclude the features
            X_train_scaling = [np.asarray(i)[included_feature_indices].tolist() for i in X_train]

            self.included_feature_indices = included_feature_indices
            self.excluded_feature_indices = excluded_feature_indices

        # Fit the actual scaler
        if self.method == 'robust_z_score':
            scaler = RobustStandardScaler().fit(X_train_scaling)
        elif self.method == 'z_score':
            scaler = StandardScaler().fit(X_train_scaling)
        elif self.method == 'robust':
            scaler = RobustScaler().fit(X_train_scaling)
        elif self.method == 'minmax':
            scaler = MinMaxScaler().fit(X_train_scaling)
        elif self.method == 'log_z_score':
            scaler = LogStandardScaler().fit(X_train_scaling)
        else:
            raise ae.WORCKeyError(f'{self.method} is not a ' +
                                  'valid scaling method. Should be any of ' +
                                  f'{accepted_scalers}.')

        self.scaler = scaler

    def transform(self, X_test):
        """Transform feature values with fitted scaler."""
        # Check if fit has been applied first
        check_is_fitted(self.scaler)

        # First exclude features which should be skipped
        if self.excluded_feature_indices:
            incl = self.included_feature_indices
            excl = self.excluded_feature_indices
            X_test_scaling = [np.asarray(i)[incl].tolist() for i in X_test]
            X_test_nonscaling = np.asarray([np.asarray(i)[excl].tolist() for i in X_test])
        else:
            X_test_scaling = X_test

        # Check if we should apply a logit transform and if so, do so
        if self.method == 'log_z_score':
            # See LogStandardScaler docstring: apply logit after adding constants
            X_test_scaling = np.log(X_test_scaling - self.scaler.F_min +
                                    self.scaler.F_median - self.scaler.F_min)

        # Apply scaling to included features
        X_test_scaling = self.scaler.transform(X_test_scaling)
        X_test_scaling = np.asarray(X_test_scaling)

        if self.excluded_feature_indices:
            # Recombine in same order as original
            if isinstance(X_test, list):
                X_test = np.asarray(X_test)

            X_test_out = np.zeros(X_test.shape)
            for inum, i in enumerate(incl):
                X_test_out[:, i] = X_test_scaling[:, inum]

            for inum, i in enumerate(excl):
                X_test_out[:, i] = X_test_nonscaling[:, inum]

            return X_test_out
        else:
            return X_test_scaling


class RobustStandardScaler(StandardScaler):
    """Scale features using z-score that is robust to outliers.

    This scaler removes outliers (<5th and >95th percentile) and
    afterwards uses z-scoring to scale the features.

    This scaler is thus a combination of the RobustScaler and StandardScaler
    from sklearn, hence please see those respective documentations for
    more information:

    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html

    """

    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.

        Note: if over 80% of the features are excluded in robustness,
        we switch to the standardscaler, as otherwise all numbers will be NaN
        after scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y
            Ignored

        """
        # Reset internal state before fitting
        self._reset()

        # Remove outliers, see docstring
        percentiles = [np.nanpercentile(X, 5, axis=0),
                       np.nanpercentile(X, 95, axis=0)]

        self.percentile_ = percentiles

        X_original = np.copy(X)
        X_selected = np.copy(X)
        total_patients = X_selected.shape[0]
        for n_feature in range(X_selected.shape[1]):
            if percentiles[0][n_feature] != percentiles[1][n_feature]:
                X_selected[:, n_feature] =\
                    np.where(X_selected[:, n_feature] > percentiles[0][n_feature],
                             X_selected[:, n_feature],
                             np.NaN)

                X_selected[:, n_feature] =\
                    np.where(X_selected[:, n_feature] < percentiles[1][n_feature],
                             X_selected[:, n_feature],
                             np.NaN)

                if total_patients - np.sum(np.isnan(X_selected[:, n_feature])) < 2:
                    # Keep original, as from zero or one value we cannot scale
                    # If that was originally so, than we let the scaler handle this
                    X_selected[:, n_feature] = X_original[:, n_feature]

        return self.partial_fit(X_selected, y)


class LogStandardScaler(StandardScaler):
    """Scale features using z-score and a logit transform.

    This scaler first applies a logit transform to each feature before
    applying a z-score, i.e. the standard scaler. To handle negative
    and zero values, a constant is added before applying the logit transform:

    lij = log(fij - min(Fj) + median(Fj) - min(Fj))
    Zij = (lij - mu)/ sigma

    Based on https://arxiv.org/pdf/2012.06875v1.pdf.


    """

    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y
            Ignored

        """
        # Reset internal state before fitting
        self._reset()

        # Determine constant terms
        self.F_median = np.median(X, axis=0)
        self.F_min = np.min(X, axis=0)

        # Apply logit transform
        X = np.log(X - self.F_min + self.F_median - self.F_min)

        return self.partial_fit(X, y)


def test():
    """Test Scaling."""
    # Small test
    a = np.random.rand(8, 10)
    a[5, 8] = 2
    a[2, 2] = 0
    a[:, 4] = [0, 0, 1, 1, 1, 1, 1, 0]
    feature_labels = ['Random'] * 10
    feature_labels[4] = skip_features = 'Skip'
    print('Original:')
    print(a)

    s = WORCScaler(method='robust_z_score', skip_features=skip_features)
    s.fit(a, feature_labels)
    b = s.transform(a)

    print('Output robust z_score:')
    print(b)
    print('Percentiles:')
    print(s.scaler.percentile_[0])
    print(s.scaler.percentile_[1])
    print('Mean and std:')
    print(s.scaler.mean_)
    print(s.scaler.var_)

    # Compare with Log StandardScaler
    s2 = WORCScaler(method='log_z_score', skip_features=skip_features)
    s2.fit(a, feature_labels)
    b = s2.transform(a)
    print('Output log z_score:')
    print(b)
    print('F_median and F_min:')
    print(s2.scaler.F_median)
    print(s2.scaler.F_min)
    print('Mean and std:')
    print(s2.scaler.mean_)
    print(s2.scaler.var_)

    # Compare with StandardScaler
    s3 = StandardScaler().fit(a)
    print('Standard scaler mean and std:')
    print(s3.mean_)
    print(s3.var_)
    print('Output:')
    print(s3.transform(a))

    # See if we're robust to NaN's
    nanmatrix = np.squeeze([[[np.nan]*400] * 10])
    print(nanmatrix.shape)
    nanmatrix[1, :] = 1
    nanmatrix[2, :] = 2
    s = WORCScaler()
    s.fit(nanmatrix)
    print(s.transform(nanmatrix))
    print(np.sum(np.isnan(s.transform(nanmatrix))))


if __name__ == "__main__":
    test()
