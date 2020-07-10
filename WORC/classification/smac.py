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

    cf = parameters['Classification']
    cs = ConfigurationSpace()

    # The first argument to parse is the choice of classifier
    classifier = CategoricalHyperparameter('classifiers',
                                           choices=['SVM', 'RF', 'LR', 'LDA', 'QDA', 'GaussianNB'])
    cs.add_hyperparameter(classifier)

    # SVM
    # 5 hyperparameters:
    #   1) kernel       | conditional on classifier: SVM
    #   2) C            | conditional on classifier: SVM
    #
    #   3) degree       | conditional on kernel: poly
    #   4) gamma        | conditional on kernel: poly, rbf
    #   5) coef0        | conditional on kernel: poly
    kernel = CategoricalHyperparameter('SVMKernel',
                                       choices=cf['SVMKernel'])
    C = UniformFloatHyperparameter('SVMC',
                                   lower=pow(10, cf['SVMC'][0]),
                                   upper=pow(10, cf['SVMC'][0] + cf['SVMC'][1]),
                                   log=True)
    cs.add_hyperparameters([kernel, C])
    cs.add_conditions([InCondition(child=kernel, parent=classifier, values=['SVM']),
                       InCondition(child=C, parent=classifier, values=['SVM'])])

    degree = UniformIntegerHyperparameter('SVMdegree',
                                          lower=cf['SVMdegree'][0],
                                          upper=cf['SVMdegree'][0] + cf['SVMdegree'][1])
    coef0 = UniformFloatHyperparameter('SVMcoef0',
                                       lower=cf['SVMcoef0'][0],
                                       upper=cf['SVMcoef0'][1])
    gamma = UniformFloatHyperparameter('SVMgamma',
                                       lower=pow(10, cf['SVMgamma'][0]),
                                       upper=pow(10, cf['SVMgamma'][0] + cf['SVMgamma'][1]),
                                       log=True)
    cs.add_hyperparameters([degree, coef0, gamma])
    cs.add_conditions([InCondition(child=degree, parent=kernel, values=['poly']),
                       InCondition(child=coef0, parent=kernel, values=['poly']),
                       InCondition(child=gamma, parent=kernel, values=['poly', 'rbf'])])

    # RF
    # 3 hyperparameters:
    #   1) n_estimators         | conditional on classifier: RF
    #   2) max_depth            | conditional on classifier: RF
    #   3) min_samples_split    | conditional on classifier: RF
    n_estimators = UniformIntegerHyperparameter('RFn_estimators',
                                                lower=cf['RFn_estimators'][0],
                                                upper=cf['RFn_estimators'][0] + cf['RFn_estimators'][1])
    max_depth = UniformIntegerHyperparameter('RFmax_depth',
                                             lower=cf['RFmax_depth'][0],
                                             upper=cf['RFmax_depth'][0] + cf['RFmax_depth'][1])
    min_samples_split = UniformIntegerHyperparameter('RFmin_samples_split',
                                                     lower=cf['RFmin_samples_split'][0],
                                                     upper=cf['RFmin_samples_split'][0] + cf['RFmin_samples_split'][1])
    cs.add_hyperparameters([n_estimators, max_depth, min_samples_split])
    cs.add_conditions([InCondition(child=n_estimators, parent=classifier, values=['RF']),
                       InCondition(child=max_depth, parent=classifier, values=['RF']),
                       InCondition(child=min_samples_split, parent=classifier, values=['RF'])])

    # LR
    # 2 hyperparameters:
    #   1) penalty          | conditional on classifier: LR
    #   2) C                | conditional on classifier: LR
    penalty = CategoricalHyperparameter('LRpenalty', choices=cf['LRpenalty'])
    C = UniformFloatHyperparameter('LRC',
                                   lower=cf['LRC'][0],
                                   upper=cf['LRC'][0] + cf['LRC'][1])
    cs.add_hyperparameters([penalty, C])
    cs.add_conditions([InCondition(child=penalty, parent=classifier, values=['LR']),
                       InCondition(child=C, parent=classifier, values=['LR'])])

    # LDA
    # 2 hyperparameters:
    #   1) solver           | conditional on classifier: LDA
    #
    #   2) shrinkage        | conditional on solver: lsqr, eigen
    solver = CategoricalHyperparameter('LDA_solver', choices=cf['LDA_solver'])
    cs.add_hyperparameter(solver)
    cs.add_condition(InCondition(child=solver, parent=classifier, values=['LDA']))

    shrinkage = UniformFloatHyperparameter('LDA_shrinkage',
                                           lower=pow(10, cf['LDA_shrinkage'][0]),
                                           upper=pow(10, cf['LDA_shrinkage'][0] + cf['LDA_shrinkage'][1]),
                                           log=True)
    cs.add_hyperparameter(shrinkage)
    cs.add_condition(InCondition(child=shrinkage, parent=solver, values=['lsqr', 'eigen']))

    # QDA
    # 1 hyperparameter:
    #   1) reg_param        | conditional on classifier: QDA
    reg_param = UniformFloatHyperparameter('QDA_reg_param',
                                           lower=pow(10, cf['QDA_reg_param'][0]),
                                           upper=pow(10, cf['QDA_reg_param'][0] + cf['QDA_reg_param'][1]),
                                           log=True)
    cs.add_hyperparameter(reg_param)
    cs.add_condition(InCondition(child=reg_param, parent=classifier, values=['QDA']))

    # GaussianNB
    # 0 hyperparameters


    ### Preprocessing ###
    # 9 preprocessing steps are included:
    #   1. Scaling
    #   2. Imputation
    #   3. Groupwise selection
    #   4. Variance selection
    #   5. Relief
    #   6. Select from model
    #   7. PCA
    #   8. Statistical test
    #   9. Oversampling

    # Feature scaling
    # 1 hyperparameter:
    #   1) scaling method
    scaling = CategoricalHyperparameter('use_featureScaling',
                                        choices=['True', 'False'])
    cs.add_hyperparameter(scaling)

    scaling_method = CategoricalHyperparameter('FeatureScaling',
                                               choices=[parameters['FeatureScaling']['scaling_method']])
    cs.add_hyperparameter(scaling_method)
    cs.add_condition(InCondition(child=scaling_method, parent=scaling, values=['True']))

    # Feature imputation
    # 2 hyperparameters:
    #   1) strategy
    #   2) n_neighbors          | Conditional on strategy: knn
    imputation = CategoricalHyperparameter('Imputation',
                                           choices=['True', 'False'])
    cs.add_hyperparameter(imputation)

    imputation_strategy = CategoricalHyperparameter('ImputationMethod',
                                                    choices=parameters['Imputation']['strategy'])
    cs.add_hyperparameter(imputation_strategy)
    cs.add_condition(InCondition(child=imputation_strategy, parent=imputation,
                                 values=['True']))

    imputation_n_neighbors = UniformIntegerHyperparameter('ImputationNeighbours',
                                                          lower=parameters['Imputation']['n_neighbors'][0],
                                                          upper=parameters['Imputation']['n_neighbors'][0] +
                                                          parameters['Imputation']['n_neighbors'][1])
    cs.add_hyperparameter(imputation_n_neighbors)
    cs.add_condition(InCondition(child=imputation_n_neighbors, parent=imputation_strategy,
                                 values=['knn']))

    # PCA
    # 2 hyperparameter:
    #   1) type
    #
    #   2) n_components     | Conditional on type: n_components
    pca = CategoricalHyperparameter('UsePCA', choices=['True', 'False'])
    cs.add_hyperparameter(pca)

    pca_type = CategoricalHyperparameter('PCAType', choices=['95variance', 'n_components'])
    cs.add_hyperparameter(pca_type)
    cs.add_condition(InCondition(child=pca_type, parent=pca, values=['True']))

    pca_n_components = UniformIntegerHyperparameter('n_components',
                                                    lower=10,
                                                    upper=100)
    cs.add_hyperparameter(pca_n_components)
    cs.add_condition(InCondition(child=pca_n_components, parent=pca_type,
                                 values=['n_components']))

    # Variance selection
    # 0 hyperparameters
    variance_selection = CategoricalHyperparameter('Featsel_Variance', choices=['True', 'False'])
    cs.add_hyperparameter(variance_selection)


    return cs



