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

from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.configspace import ConfigurationSpace
from WORC.classification.fitandscore import fit_and_score


def build_smac_config(parameters):
    """Reads a parameter dictionary and constructs a SMAC configuration object
    from it, to be used as input for the Bayesian optimization

    Parameters
    ----------
        parameters: dict, mandatory
                Contains the required config settings

    Returns:
        ConfigurationSpace object that defines the search space of the hyperparameter
        optimization
    """

    cs = ConfigurationSpace()

    # The first argument to parse is the choice of classifier
    classifier = CategoricalHyperparameter('classifiers',
                                           choices=['SVM', 'RF'])
    cs.add_hyperparameter(classifier)

    # SVM (5 hyperparameters)
    # kernel and C are directly conditional on the SVM choice
    kernel = CategoricalHyperparameter('SVMKernel',
                                       choices=parameters['Classification']['SVMKernel'])
    C = UniformFloatHyperparameter('SVMC',
                                   lower=0.001,
                                   upper=1000,
                                   log=True)
    cs.add_hyperparameters([kernel, C])
    # Add parameters conditional on SVM choice
    cs.add_conditions([InCondition(child=kernel, parent=classifier, values=['SVM']),
                       InCondition(child=C, parent=classifier, values=['SVM'])])

    # degree, coef0 and gamma are conditional on the kernel choice
    degree = UniformIntegerHyperparameter('SVMdegree',
                                          lower=parameters['Classification']['SVMdegree'][0],
                                          upper=parameters['Classification']['SVMdegree'][0] + \
                                                parameters['Classification']['SVMdegree'][1])
    coef0 = UniformFloatHyperparameter('SVMcoef0',
                                       lower=parameters['Classification']['SVMcoef0'][0],
                                       upper=parameters['Classification']['SVMcoef0'][1])
    gamma = UniformFloatHyperparameter('SVMgamma',
                                       lower=0.001,
                                       upper=10,
                                       log=True)
    cs.add_hyperparameters([degree, coef0, gamma])
    cs.add_conditions([InCondition(child=degree, parent=kernel, values=['poly']),
                       InCondition(child=coef0, parent=kernel, values=['poly']),
                       InCondition(child=gamma, parent=kernel, values=['poly', 'rbf'])])

    # RF (3 hyperparameters)
    n_estimators = UniformIntegerHyperparameter('RFn_estimators',
                                                lower=10,
                                                upper=100)
    max_depth = UniformIntegerHyperparameter('RFmax_depth',
                                             lower=5,
                                             upper=10)
    min_samples_split = UniformIntegerHyperparameter('RFmin_samples_split',
                                                     lower=2,
                                                     upper=5)
    cs.add_hyperparameters([n_estimators, max_depth, min_samples_split])
    cs.add_conditions([InCondition(child=n_estimators, parent=classifier, values=['RF']),
                       InCondition(child=max_depth, parent=classifier, values=['RF']),
                       InCondition(child=min_samples_split, parent=classifier, values=['RF'])])

    return cs



