#!/usr/bin/env python

# Copyright 2020 - 2020 Biomedical Imaging Group Rotterdam, Departments of
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

from sklearn.preprocessing import StandardScaler
import numpy as np


class RobustStandardScaler(StandardScaler):
    """Scale features using statistics that are robust to outliers.

    This scaler removes outliers (<5th and >95th percentile) and
    afterwards uses z-scoring to scale the features.

    This scaler is thus a combination of the RobustScaler and StandardScaler
    from sklearn, hence please see those respective documentations for
    more information:

    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html

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

        # Remove outliers, see docstring
        percentiles = [np.nanpercentile(X, 5, axis=0),
                       np.nanpercentile(X, 95, axis=0)]

        self.percentile_ = percentiles

        X_selected = np.copy(X)
        for n_feature in range(X_selected.shape[1]):
            X_selected[:, n_feature] =\
                np.where(X_selected[:, n_feature] > percentiles[0][n_feature],
                         X_selected[:, n_feature],
                         np.NaN)

            X_selected[:, n_feature] =\
                np.where(X_selected[:, n_feature] < percentiles[1][n_feature],
                         X_selected[:, n_feature],
                         np.NaN)

        return self.partial_fit(X_selected, y)


def test():
    # Small test
    a = np.random.rand(8, 10)
    a[5, 8] = 2
    a[2, 2] = 0
    print('Original:')
    print(a)

    s = RobustStandardScaler().fit(a)
    b = s.transform(a)

    print('Output:')
    print(b)
    print('Percentiles:')
    print(s.percentile_[0])
    print(s.percentile_[1])
    print('Mean and std:')
    print(s.mean_)
    print(s.var_)

    # Compare with StandardScaler
    s2 = StandardScaler().fit(a)
    print('Standard scaler mean and std:')
    print(s2.mean_)
    print(s2.var_)


if __name__ == "__main__":
    test()
