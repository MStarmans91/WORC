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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from WORC.classification.RankedSVM import RankSVM_train, RankSVM_test


class RankedSVM(BaseEstimator, ClassifierMixin):
    """ An example classifier which implements a 1-NN algorithm.

    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        The input passed during :meth:`fit`
    y_ : array, shape = [n_samples]
        The labels passed during :meth:`fit`
    """
    def __init__(self, cost=1, lambda_tol=1e-6,
                 norm_tol=1e-4, max_iter=500, svm='Poly', gamma=0.05,
                 coefficient=0.05, degree=3):
        self.cost = cost
        self.lambda_tol = lambda_tol
        self.norm_tol = norm_tol
        self.max_iter = max_iter
        self.svm = svm
        self.gamma = gamma
        self.coefficient = coefficient
        self.degree = 3

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # RankedSVM requires a very specific format of y
        # Each row should represent a label, consisiting of ones and minus ones
        y = np.transpose(y).astype(np.int16)
        y[y == 0] = -1
        self.X_ = X
        self.y_ = y
        self.num_class = y.shape[0]

        Weights, Bias, SVs =\
            RankSVM_train(train_data=X,
                               train_target=y,
                               cost=self.cost,
                               lambda_tol=self.lambda_tol,
                               norm_tol=self.norm_tol,
                               max_iter=self.max_iter,
                               svm=self.svm, gamma=self.gamma,
                               coefficient=self.coefficient,
                               degree=self.degree)

        self.Weights = Weights
        self.Bias = Bias
        self.SVs = SVs

        return self

    def predict(self, X, y=None):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of int of shape = [n_samples]
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        _, Predicted_Labels =\
            RankSVM_test(test_data=X,
                         num_class=self.num_class,
                         Weights=self.Weights,
                         Bias=self.Bias,
                         SVs=self.SVs,
                         svm=self.svm, gamma=self.gamma,
                         coefficient=self.coefficient,
                         degree=self.degree)

        return Predicted_Labels

    def predict_proba(self, X, y):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of int of shape = [n_samples]
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        Probs, _ =\
            RankSVM_test(test_data=X,
                         num_class=self.num_class,
                         Weights=self.Weights,
                         Bias=self.Bias,
                         svm=self.svm, gamma=self.gamma,
                         coefficient=self.coefficient,
                         degree=self.degree)

        return Probs
