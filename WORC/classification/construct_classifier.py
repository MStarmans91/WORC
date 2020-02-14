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
from sklearn.base import is_regressor
from sksurv.svm import FastKernelSurvivalSVM, FastSurvivalSVM
from sklearn.svm import SVC
from sklearn.svm import SVR as SVMR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import SGDClassifier, ElasticNet, SGDRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import scipy
from WORC.classification.estimators import RankedSVM
from WORC.classification.AdvancedSampler import log_uniform, discrete_uniform
import WORC.addexceptions as ae


survival_classifiers = {
        'FastKernelSurvivalSVM': FastKernelSurvivalSVM,
        'FastSurvivalSVM': FastSurvivalSVM,
}

regression_classifiers = [
    'SVR', 'RFR', 'SGDR', 'Lasso', 'ElasticNet'
]

multilabel_classifiers = {
    'RankedSVM': RankedSVM,
    'RandomForestClassifier': RandomForestClassifier,
    **survival_classifiers
}

def construct_classifier(config):
    """Interface to create classification

    Different classifications can be created using this common interface

    Parameters
    ----------
        config: dict, mandatory
                Contains the required config settings. See the Github Wiki for
                all available fields.

    Returns:
        Constructed classifier
    """

    # NOTE: Function is not working anymore for regression: need
    # to move param grid creation to the create_param_grid function
    max_iter = config['max_iter']

    if any(x == config['classifiers'] for x in survival_classifiers):
        # Survival! Exciting! :D
        # Added this as first if-statement so that the if-statement for (normal) SVM would continue to work
        classifier = construct_survival_classifier(config)
    elif 'SVM' in config['classifiers']:
        # Support Vector Machine
        classifier = construct_SVM(config)

    elif config['classifiers'] == 'SVR':
        # Support Vector Regression
        classifier = construct_SVM(config, True)

    elif config['classifiers'] == 'RF':
        # Random forest kernel
        classifier = RandomForestClassifier(verbose=0,
                                            class_weight='balanced',
                                            n_estimators=config['RFn_estimators'],
                                            min_samples_split=config['RFmin_samples_split'],
                                            max_depth=config['RFmax_depth'])

    elif config['classifiers'] == 'RFR':
        # Random forest kernel regression
        classifier = RandomForestRegressor(verbose=0,
                                           n_estimators=config['RFn_estimators'],
                                           min_samples_split=config['RFmin_samples_split'],
                                           max_depth=config['RFmax_depth'])

    elif config['classifiers'] == 'ElasticNet':
        # Elastic Net Regression
        classifier = ElasticNet(alpha=config['ElasticNet_alpha'],
                                l1_ratio=config['ElasticNet_l1_ratio'],
                                max_iter=max_iter)

    elif config['classifiers'] == 'Lasso':
        # LASSO Regression
        param_grid = {'alpha': scipy.stats.uniform(loc=1.0, scale=0.5)}
        classifier = Lasso(max_iter=max_iter)

    elif config['classifiers'] == 'SGD':
        # Stochastic Gradient Descent classifier
        classifier = SGDClassifier(n_iter=config['max_iter'],
                                   alpha=config['SGD_alpha'],
                                   l1_ratio=config['SGD_l1_ratio'],
                                   loss=config['SGD_loss'],
                                   penalty=config['SGD_penalty'])

    elif config['classifiers'] == 'SGDR':
        # Stochastic Gradient Descent regressor
        classifier = SGDRegressor(n_iter=config['max_iter'],
                                  alpha=config['SGD_alpha'],
                                  l1_ratio=config['SGD_l1_ratio'],
                                  loss=config['SGD_loss'],
                                  penalty=config['SGD_penalty'])

    elif config['classifiers'] == 'LR':
        # Logistic Regression
        classifier = LogisticRegression(max_iter=max_iter,
                                        penalty=config['LRpenalty'],
                                        C=config['LRC'])
    elif config['classifiers'] == 'GaussianNB':
        # Naive Bayes classifier using Gaussian distributions
        classifier = GaussianNB()

    elif config['classifiers'] == 'ComplementNB':
        # Complement Naive Bayes classifier
        classifier = ComplementNB()

    elif config['classifiers'] == 'LDA':
        # Linear Discriminant Analysis
        if config['LDA_solver'] == 'svd':
            # Shrinkage does not work with svd solver
            shrinkage = None
        else:
            shrinkage = config['LDA_shrinkage']

        classifier = LDA(solver=config['LDA_solver'],
                         shrinkage=shrinkage)

    elif config['classifiers'] == 'QDA':
        # Linear Discriminant Analysis
        classifier = QDA(reg_param=config['QDA_reg_param'])
    else:
        message = ('Classifier {} unknown.').format(str(config['classifiers']))
        raise ae.WORCKeyError(message)

    return classifier


def is_regression_classifier(obj):
    return is_regressor(obj) or is_survival_classifier(obj)  # return true for survival as well, since it usually uses regression


def list_regression_classifiers():
    return regression_classifiers.copy() + list_survival_classifiers()  # survival usually uses regression as well so we should return the product of these two lists


def list_survival_classifiers():
    return list(survival_classifiers.keys())


def is_survival_classifier(obj):
    return any([isinstance(obj, x) for x in survival_classifiers.values()])


def list_multilabel_classifiers():
    return list(multilabel_classifiers.keys())


def is_multilabel_classifier(obj):
    return any([isinstance(obj, x) for x in multilabel_classifiers.values()])


def construct_survival_classifier(config):
    # clf.C = config['SVMC']  # whats this?

    ret = None

    if 'FastKernelSurvivalSVM' == config['classifiers']:
        svm_config = {
            'max_iter': config['max_iter'],
            'kernel': str(config['SVMKernel']),
            'degree': config['SVMdegree'],
            'coef0': config['SVMcoef0'],
            'gamma': config['SVMgamma'],
        }
        ret = FastKernelSurvivalSVM(**svm_config)
    elif 'FastSurvivalSVM' == config['classifiers']:
        svm_config = {
            'max_iter': config['max_iter']
        }
        ret = FastSurvivalSVM(**svm_config)

    return ret

"""
FastKernelSurvivalSVM:
    alpha (float, positive, default: 1) – Weight of penalizing the squared hinge loss in the objective function
    rank_ratio (float, optional, default: 1.0) – Mixing parameter between regression and ranking objective with 0 <= rank_ratio <= 1. If rank_ratio = 1, only ranking is performed, if rank_ratio = 0, only regression is performed. A non-zero value is only allowed if optimizer is one of ‘avltree’, ‘PRSVM’, or ‘rbtree’.
    fit_intercept (boolean, optional, default: False) – Whether to calculate an intercept for the regression model. If set to False, no intercept will be calculated. Has no effect if rank_ratio = 1, i.e., only ranking is performed.
    kernel ("linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed") – Kernel. Default: “linear”
    degree (int, default: 3) – Degree for poly kernels. Ignored by other kernels.
    gamma (float, optional) – Kernel coefficient for rbf and poly kernels. Default: 1/n_features. Ignored by other kernels.
    coef0 (float, optional) – Independent term in poly and sigmoid kernels. Ignored by other kernels.
    kernel_params (mapping of string to any, optional) – Parameters (keyword arguments) and values for kernel passed as call
    max_iter (int, optional, default: 20) – Maximum number of iterations to perform in Newton optimization
    verbose (bool, optional, default: False) – Whether to print messages during optimization
    tol (float, optional) – Tolerance for termination. For detailed control, use solver-specific options.
    optimizer ("avltree" | "rbtree", optional, default: "rbtree") – Which optimizer to use.
    random_state (int or numpy.random.RandomState instance, optional) – Random number generator (used to resolve ties in survival times).
    timeit (False or int) – If non-zero value is provided the time it takes for optimization is measured. The given number of repetitions are performed. Results can be accessed from the optimizer_result_ attribute.
"""

"""
FastSurvivalSVM:
    alpha (float, positive, default: 1) – Weight of penalizing the squared hinge loss in the objective function
    rank_ratio (float, optional, default: 1.0) – Mixing parameter between regression and ranking objective with 0 <= rank_ratio <= 1. If rank_ratio = 1, only ranking is performed, if rank_ratio = 0, only regression is performed. A non-zero value is only allowed if optimizer is one of ‘avltree’, ‘rbtree’, or ‘direct-count’.
    fit_intercept (boolean, optional, default: False) – Whether to calculate an intercept for the regression model. If set to False, no intercept will be calculated. Has no effect if rank_ratio = 1, i.e., only ranking is performed.
    max_iter (int, optional, default: 20) – Maximum number of iterations to perform in Newton optimization
    verbose (bool, optional, default: False) – Whether to print messages during optimization
    tol (float, optional) – Tolerance for termination. For detailed control, use solver-specific options.
    optimizer ("avltree" | "direct-count" | "PRSVM" | "rbtree" | "simple", optional, default: avltree)) – Which optimizer to use.
    random_state (int or numpy.random.RandomState instance, optional) – Random number generator (used to resolve ties in survival times).
    timeit (False or int) – If non-zero value is provided the time it takes for optimization is measured. The given number of repetitions are performed. Results can be accessed from the optimizer_result_ attribute.
"""


def construct_SVM(config, regression=False):
    """
    Constructs a SVM classifier

    Args:
        config (dict): Dictionary of the required config settings
        features (pandas dataframe): A pandas dataframe containing the features
         to be used for classification

    Returns:
        SVM/SVR classifier, parameter grid
    """

    max_iter = config['max_iter']
    if not regression:
        clf = SVC(class_weight='balanced', probability=True, max_iter=max_iter)
    else:
        clf = SVMR(max_iter=max_iter)

    clf.kernel = str(config['SVMKernel'])
    clf.C = config['SVMC']
    clf.degree = config['SVMdegree']
    clf.coef0 = config['SVMcoef0']
    clf.gamma = config['SVMgamma']

    # Check if we need to use a ranked SVM
    if config['classifiers'] == 'RankedSVM':
        clf = RankedSVM()
        param_grid = {'svm': ['Poly'],
                      'degree': [2, 3, 4, 5],
                      'gamma':  scipy.stats.uniform(loc=0, scale=1e-3),
                      'coefficient': scipy.stats.uniform(loc=0, scale=1e-2),
                      }

    return clf


def create_param_grid(config):
    ''' Create a parameter grid for the WORC classifiers based on the
        provided configuration. '''

    # We only need parameters from the Classification part of the config
    config = config['Classification']

    # Create grid and put in name of classifiers and maximum iterations
    param_grid = dict()
    param_grid['classifiers'] = config['classifiers']
    param_grid['max_iter'] = config['max_iter']

    # SVM parameters
    param_grid['SVMKernel'] = config['SVMKernel']
    param_grid['SVMC'] = log_uniform(loc=config['SVMC'][0],
                                     scale=config['SVMC'][1])
    param_grid['SVMdegree'] = scipy.stats.uniform(loc=config['SVMdegree'][0],
                                                  scale=config['SVMdegree'][1])
    param_grid['SVMcoef0'] = scipy.stats.uniform(loc=config['SVMcoef0'][0],
                                                 scale=config['SVMcoef0'][1])
    param_grid['SVMgamma'] = log_uniform(loc=config['SVMgamma'][0],
                                         scale=config['SVMgamma'][1])

    # RF parameters
    # RF parameters
    param_grid['RFn_estimators'] =\
        discrete_uniform(loc=config['RFn_estimators'][0],
                         scale=config['RFn_estimators'][1])
    param_grid['RFmin_samples_split'] =\
        discrete_uniform(loc=config['RFmin_samples_split'][0],
                         scale=config['RFmin_samples_split'][1])
    param_grid['RFmax_depth'] =\
        discrete_uniform(loc=config['RFmax_depth'][0],
                         scale=config['RFmax_depth'][1])

    # Logistic Regression parameters
    param_grid['LRpenalty'] = config['LRpenalty']
    param_grid['LRC'] = scipy.stats.uniform(loc=config['LRC'][0],
                                            scale=config['LRC'][1])

    # LDA/QDA parameters
    param_grid['LDA_solver'] = config['LDA_solver']
    param_grid['LDA_shrinkage'] = log_uniform(loc=config['LDA_shrinkage'][0],
                                              scale=config['LDA_shrinkage'][1])
    param_grid['QDA_reg_param'] = log_uniform(loc=config['QDA_reg_param'][0],
                                              scale=config['QDA_reg_param'][1])

    # ElasticNet parameters
    param_grid['ElasticNet_alpha'] =\
        log_uniform(loc=config['ElasticNet_alpha'][0],
                    scale=config['ElasticNet_alpha'][1])
    param_grid['ElasticNet_l1_ratio'] =\
        scipy.stats.uniform(loc=config['ElasticNet_l1_ratio'][0],
                            scale=config['ElasticNet_l1_ratio'][1])

    # SGD Regression parameters
    param_grid['SGD_alpha'] =\
        log_uniform(loc=config['SGD_alpha'][0],
                    scale=config['SGD_alpha'][1])

    param_grid['SGD_l1_ratio'] =\
        scipy.stats.uniform(loc=config['SGD_l1_ratio'][0],
                            scale=config['SGD_l1_ratio'][1])
    param_grid['SGD_loss'] = config['SGD_loss']
    param_grid['SGD_penalty'] = config['SGD_penalty']

    # Naive Bayes parameters
    param_grid['CNB_alpha'] =\
        scipy.stats.uniform(loc=config['CNB_alpha'][0],
                            scale=config['CNB_alpha'][1])

    return param_grid
