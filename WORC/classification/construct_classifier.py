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

from sklearn.svm import SVC
from sklearn.svm import SVR as SVMR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import SGDClassifier, ElasticNet, SGDRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import scipy
from WORC.classification.estimators import RankedSVM
from WORC.classification.AdvancedSampler import log_uniform, discrete_uniform
import WORC.addexceptions as ae
from xgboost import XGBClassifier, XGBRegressor


def construct_classifier(config):
    """Interface to create classification.

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
    if 'SVM' in config['classifiers']:
        # Support Vector Machine
        classifier = construct_SVM(config)

    elif config['classifiers'] == 'SVR':
        # Support Vector Regression
        classifier = construct_SVM(config, True)

    elif config['classifiers'] == 'AdaBoostClassifier':
        # AdaBoost classifier
        learning_rate = config['AdaBoost_learning_rate']
        n_estimators = config['AdaBoost_n_estimators']
        classifier = AdaBoostClassifier(n_estimators=n_estimators,
                                        learning_rate=learning_rate,
                                        random_state=config['random_seed'])

    elif config['classifiers'] == 'AdaBoostRegressor':
        # AdaBoost regressor
        learning_rate = config['AdaBoost_learning_rate']
        n_estimators = config['AdaBoost_n_estimators']
        classifier = AdaBoostRegressor(n_estimators=n_estimators,
                                       learning_rate=learning_rate,
                                       random_state=config['random_seed'])

    elif config['classifiers'] == 'XGBClassifier':
        # XGB Classifier
        max_depth = config['XGB_max_depth']
        learning_rate = config['XGB_learning_rate']
        gamma = config['XGB_gamma']
        min_child_weight = config['XGB_min_child_weight']
        boosting_rounds = config['XGB_boosting_rounds']
        colsample_bytree = config['XGB_colsample_bytree']
        classifier = XGBClassifier(max_depth=max_depth,
                                   learning_rate=learning_rate,
                                   gamma=gamma,
                                   min_child_weight=min_child_weight,
                                   n_estimators=boosting_rounds,
                                   colsample_bytree=colsample_bytree,
                                   random_state=config['random_seed'])

    elif config['classifiers'] == 'XGBRegressor':
        # XGB Classifier
        max_depth = config['XGB_max_depth']
        learning_rate = config['XGB_learning_rate']
        gamma = config['XGB_gamma']
        min_child_weight = config['XGB_min_child_weight']
        boosting_rounds = config['XGB_boosting_rounds']
        colsample_bytree = config['XGB_colsample_bytree']
        classifier = XGBRegressor(max_depth=max_depth,
                                  learning_rate=learning_rate,
                                  gamma=gamma,
                                  min_child_weight=min_child_weight,
                                  n_estimators=boosting_rounds,
                                  colsample_bytree=colsample_bytree,
                                  random_state=config['random_seed'])

    elif config['classifiers'] == 'RF':
        # Random forest kernel
        classifier = RandomForestClassifier(verbose=0,
                                            class_weight='balanced',
                                            n_estimators=config['RFn_estimators'],
                                            min_samples_split=config['RFmin_samples_split'],
                                            max_depth=config['RFmax_depth'],
                                            random_state=config['random_seed'])

    elif config['classifiers'] == 'RFR':
        # Random forest kernel regression
        classifier = RandomForestRegressor(verbose=0,
                                           n_estimators=config['RFn_estimators'],
                                           min_samples_split=config['RFmin_samples_split'],
                                           max_depth=config['RFmax_depth'],
                                           random_state=config['random_seed'])

    elif config['classifiers'] == 'ElasticNet':
        # Elastic Net Regression
        classifier = ElasticNet(alpha=config['ElasticNet_alpha'],
                                l1_ratio=config['ElasticNet_l1_ratio'],
                                max_iter=max_iter,
                                random_state=config['random_seed'])

    elif config['classifiers'] == 'Lasso':
        # LASSO Regression
        classifier = Lasso(max_iter=max_iter,
                           random_state=config['random_seed'])

    elif config['classifiers'] == 'SGD':
        # Stochastic Gradient Descent classifier
        classifier = SGDClassifier(n_iter=config['max_iter'],
                                   alpha=config['SGD_alpha'],
                                   l1_ratio=config['SGD_l1_ratio'],
                                   loss=config['SGD_loss'],
                                   penalty=config['SGD_penalty'],
                                   random_state=config['random_seed'])

    elif config['classifiers'] == 'SGDR':
        # Stochastic Gradient Descent regressor
        classifier = SGDRegressor(max_iter=config['max_iter'],
                                  alpha=config['SGD_alpha'],
                                  l1_ratio=config['SGD_l1_ratio'],
                                  loss=config['SGD_loss'],
                                  penalty=config['SGD_penalty'])

    elif config['classifiers'] == 'LR':
        # Logistic Regression
        if config['LRpenalty'] == 'elasticnet' or config['LRpenalty'] == 'l1':
            # saga solver required for elasticnet
            if config['LR_solver'] != 'saga':
                p = config['LRpenalty']
                print(f"[WORC Warning] {p} penalty requires saga " +\
                      f"solver, got {config['LR_solver']}. Changing solver.")
                config['LR_solver'] = 'saga'

        classifier = LogisticRegression(max_iter=max_iter,
                                        penalty=config['LRpenalty'],
                                        solver=config['LR_solver'],
                                        l1_ratio=config['LR_l1_ratio'],
                                        C=config['LRC'],
                                        random_state=config['random_seed'])

    elif config['classifiers'] == 'LinR':
        # Linear Regression
        classifier = LinearRegression()

    elif config['classifiers'] == 'Ridge':
        # Ridge Regression
        classifier = Ridge(alpha=config['ElasticNet_alpha'],
                           max_iter=max_iter,
                           random_state=config['random_seed'])

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


def construct_SVM(config, regression=False):
    """Construct a SVM classifier.

    Args:
        config (dict): Dictionary of the required config settings
        features (pandas dataframe): A pandas dataframe containing the features
         to be used for classification

    Returns:
        SVM/SVR classifier, parameter grid

    """
    max_iter = config['max_iter']
    if not regression:
        clf = SVC(class_weight='balanced', probability=True, max_iter=max_iter,
                  random_state=config['random_seed'])
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
    """Create a parameter grid for the WORC classifiers."""
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
    param_grid['LR_solver'] = config['LR_solver']
    param_grid['LR_l1_ratio'] =\
        scipy.stats.uniform(loc=config['LR_l1_ratio'][0],
                            scale=config['LR_l1_ratio'][1])
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

    # AdaBoost parameters
    param_grid['AdaBoost_n_estimators'] =\
        discrete_uniform(loc=config['AdaBoost_n_estimators'][0],
                         scale=config['AdaBoost_n_estimators'][1])

    param_grid['AdaBoost_learning_rate'] =\
        scipy.stats.uniform(loc=config['AdaBoost_learning_rate'][0],
                            scale=config['AdaBoost_learning_rate'][1])

    # XGDBoost parameters
    param_grid['XGB_boosting_rounds'] =\
        discrete_uniform(loc=config['XGB_boosting_rounds'][0],
                         scale=config['XGB_boosting_rounds'][1])

    param_grid['XGB_max_depth'] =\
        discrete_uniform(loc=config['XGB_max_depth'][0],
                         scale=config['XGB_max_depth'][1])

    param_grid['XGB_learning_rate'] =\
        scipy.stats.uniform(loc=config['XGB_learning_rate'][0],
                            scale=config['XGB_learning_rate'][1])

    param_grid['XGB_gamma'] =\
        scipy.stats.uniform(loc=config['XGB_gamma'][0],
                            scale=config['XGB_gamma'][1])

    param_grid['XGB_min_child_weight'] =\
        discrete_uniform(loc=config['XGB_min_child_weight'][0],
                         scale=config['XGB_min_child_weight'][1])

    param_grid['XGB_colsample_bytree'] =\
        scipy.stats.uniform(loc=config['XGB_colsample_bytree'][0],
                            scale=config['XGB_colsample_bytree'][1])

    return param_grid
