#!/usr/bin/env python

# Copyright 2016-2022 Biomedical Imaging Group Rotterdam, Departments of
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

try:
    from ConfigSpace.conditions import InCondition
    from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
        UniformFloatHyperparameter, UniformIntegerHyperparameter, \
        Constant
    from smac.configspace import ConfigurationSpace
except:
    print("[INFO] Bayesian optimization through SMAC functionality currently not available. Please see https://worc.readthedocs.io/en/latest/static/additionalfunctionality.html.")



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
                                           choices=cf['classifiers'])
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
                                       upper=cf['SVMcoef0'][0] + cf['SVMcoef0'][1])
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
    #   3) solver           | conditional on classifier: LR
    #   4) l1_ratio         | conditional on penalty: elasticnet
    penalty = CategoricalHyperparameter('LRpenalty', choices=cf['LRpenalty'])
    C = UniformFloatHyperparameter('LRC',
                                   lower=cf['LRC'][0],
                                   upper=cf['LRC'][0] + cf['LRC'][1])
    lr_solver = CategoricalHyperparameter('LR_solver', choices=cf['LR_solver'])
    lr_l1_ratio = UniformFloatHyperparameter('LR_l1_ratio',
                                             lower=cf['LR_l1_ratio'][0],
                                             upper=cf['LR_l1_ratio'][0] + cf['LR_l1_ratio'][1])
    cs.add_hyperparameters([penalty, C, lr_solver, lr_l1_ratio])
    cs.add_conditions([InCondition(child=penalty, parent=classifier, values=['LR']),
                       InCondition(child=C, parent=classifier, values=['LR']),
                       InCondition(child=lr_solver, parent=classifier, values=['LR']),
                       InCondition(child=lr_l1_ratio, parent=penalty, values=['elasticnet'])])

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

    # AdaBoostClassifier
    # 2 hyperparameters:
    #   1) n_estimators     | conditional on classifier: AdaBoost
    #   2) learning_rate    | conditional on classifier: AdaBoost
    ada_n_estimators = UniformIntegerHyperparameter('AdaBoost_n_estimators',
                                                    lower=cf['AdaBoost_n_estimators'][0],
                                                    upper=cf['AdaBoost_n_estimators'][0] +
                                                          cf['AdaBoost_n_estimators'][1])
    ada_learning_rate = UniformFloatHyperparameter('AdaBoost_learning_rate',
                                                   lower=cf['AdaBoost_learning_rate'][0],
                                                   upper=cf['AdaBoost_learning_rate'][0] +
                                                         cf['AdaBoost_learning_rate'][1])
    cs.add_hyperparameters([ada_n_estimators, ada_learning_rate])
    cs.add_conditions([InCondition(child=ada_n_estimators, parent=classifier, values=['AdaBoostClassifier']),
                       InCondition(child=ada_learning_rate, parent=classifier, values=['AdaBoostClassifier'])])

    # XGBClassifier
    # 6 hyperparameters:
    #   1) boosting_rounds  | conditional on classifier: XGB
    #   2) max_depth        | conditional on classifier: XGB
    #   3) learning_rate    | conditional on classifier: XGB
    #   4) gamma            | conditional on classifier: XGB
    #   5) min_child_weight | conditional on classifier: XGB
    #   6) colsample_bytree | conditional on classifier: XGB
    boosting_rounds = UniformIntegerHyperparameter('XGB_boosting_rounds',
                                                   lower=cf['XGB_boosting_rounds'][0],
                                                   upper=cf['XGB_boosting_rounds'][0] +
                                                         cf['XGB_boosting_rounds'][1])
    xgb_max_depth = UniformIntegerHyperparameter('XGB_max_depth',
                                                 lower=cf['XGB_max_depth'][0],
                                                 upper=cf['XGB_max_depth'][0] +
                                                       cf['XGB_max_depth'][1])
    xgb_learning_rate = UniformFloatHyperparameter('XGB_learning_rate',
                                                   lower=cf['XGB_learning_rate'][0],
                                                   upper=cf['XGB_learning_rate'][0] +
                                                         cf['XGB_learning_rate'][1])
    xgb_gamma = UniformFloatHyperparameter('XGB_gamma',
                                           lower=cf['XGB_gamma'][0],
                                           upper=cf['XGB_gamma'][0] +
                                                 cf['XGB_gamma'][1])
    min_child_weight = UniformIntegerHyperparameter('XGB_min_child_weight',
                                                    lower=cf['XGB_min_child_weight'][0],
                                                    upper=cf['XGB_min_child_weight'][0] +
                                                          cf['XGB_min_child_weight'][1])
    colsample_bytree = UniformFloatHyperparameter('XGB_colsample_bytree',
                                                  lower=cf['XGB_colsample_bytree'][0],
                                                  upper=cf['XGB_colsample_bytree'][0] +
                                                        cf['XGB_colsample_bytree'][1])
    cs.add_hyperparameters([boosting_rounds, xgb_max_depth, xgb_learning_rate,
                            xgb_gamma, min_child_weight, colsample_bytree])
    cs.add_conditions([InCondition(child=boosting_rounds, parent=classifier, values=['XGBClassifier']),
                       InCondition(child=xgb_max_depth, parent=classifier, values=['XGBClassifier']),
                       InCondition(child=xgb_learning_rate, parent=classifier, values=['XGBClassifier']),
                       InCondition(child=xgb_gamma, parent=classifier, values=['XGBClassifier']),
                       InCondition(child=min_child_weight, parent=classifier, values=['XGBClassifier']),
                       InCondition(child=colsample_bytree, parent=classifier, values=['XGBClassifier'])])

    # LightGBMclassifier
    # 5 hyperparameters:
    #   1) num_leaves           | conditional on classifier: LightGBM
    #   2) max_depth            | conditional on classifier: LightGBM
    #   3) min_child_samples    | conditional on classifier: LightGBM
    #   4) reg_alpha            | conditional on classifier: LightGBM
    #   5) reg_lambda           | conditional on classifier: LightGBM
    #   6) min_child_weight     | conditional on classifier: LightGBM
    # LightGBM_num_leaves = UniformIntegerHyperparameter('LightGBM_num_leaves',
    #                                                lower=cf['LightGBM_num_leaves'][0],
    #                                                upper=cf['LightGBM_num_leaves'][0] +
    #                                                      cf['LightGBM_num_leaves'][1])
    # LightGBM_max_depth = UniformIntegerHyperparameter('LightGBM_max_depth',
    #                                              lower=cf['LightGBM_max_depth'][0],
    #                                              upper=cf['LightGBM_max_depth'][0] +
    #                                                    cf['LightGBM_max_depth'][1])
    # LightGBM_min_child_samples = UniformIntegerHyperparameter('LightGBM_min_child_samples',
    #                                                lower=cf['LightGBM_min_child_samples'][0],
    #                                                upper=cf['LightGBM_min_child_samples'][0] +
    #                                                      cf['LightGBM_min_child_samples'][1])
    #
    # LightGBM_reg_alpha = UniformFloatHyperparameter('LightGBM_reg_alpha',
    #                                        lower=cf['LightGBM_reg_alpha'][0],
    #                                        upper=cf['LightGBM_reg_alpha'][0] +
    #                                              cf['LightGBM_reg_alpha'][1])
    # LightGBM_reg_lambda = UniformFloatHyperparameter('LightGBM_reg_lambda',
    #                                                 lower=cf['LightGBM_reg_lambda'][0],
    #                                                 upper=cf['LightGBM_reg_lambda'][0] +
    #                                                       cf['LightGBM_reg_lambda'][1])
    # LightGBM_min_child_weight = UniformFloatHyperparameter('LightGBM_min_child_weight',
    #                                        lower=pow(10, cf['LightGBM_min_child_weight'][0]),
    #                                        upper=pow(10, cf['LightGBM_min_child_weight'][0] + cf['LightGBM_min_child_weight'][1]),
    #                                        log=True)
    #
    # cs.add_hyperparameters([LightGBM_num_leaves, LightGBM_max_depth, LightGBM_min_child_samples,
    #                         LightGBM_reg_alpha, LightGBM_reg_lambda, LightGBM_min_child_weight])
    # cs.add_conditions([InCondition(child=LightGBM_num_leaves, parent=classifier, values=['LightGBMClassifier']),
    #                    InCondition(child=LightGBM_max_depth, parent=classifier, values=['LightGBMClassifier']),
    #                    InCondition(child=LightGBM_min_child_samples, parent=classifier, values=['LightGBMClassifier']),
    #                    InCondition(child=LightGBM_reg_alpha, parent=classifier, values=['LightGBMClassifier']),
    #                    InCondition(child=LightGBM_reg_lambda, parent=classifier, values=['LightGBMClassifier']),
    #                    InCondition(child=LightGBM_min_child_weight, parent=classifier, values=['LightGBMClassifier'])])

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
    # ! NO LONGER PART OF THE SEARCH!
    # 1 hyperparameter:
    #   1) scaling method
    scaling = Constant('FeatureScaling', value=parameters['FeatureScaling']['scaling_method'][0])
    cs.add_hyperparameter(scaling)
    scaling_skip_features = Constant('FeatureScaling_skip_features', value=parameters['FeatureScaling']['skip_features'][0])
    cs.add_hyperparameter(scaling_skip_features)

    # Feature imputation --> always on in RS
    # 2 hyperparameters:
    #   1) strategy
    #   2) n_neighbors          | Conditional on strategy: knn
    imputation = CategoricalHyperparameter('Imputation',
                                           choices=parameters['Imputation']['use'])
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

    # Variance selection --> always on in RS
    # 0 hyperparameters
    variance_selection = CategoricalHyperparameter('Featsel_Variance', choices=['True'])
    cs.add_hyperparameter(variance_selection)

    # Relief
    # 4 hyperparameters:
    #   1) NN
    #   2) Sample size
    #   3) DistanceP
    #   4) Numfeatures

    relief = CategoricalHyperparameter('ReliefUse', choices=['True', 'False'])
    cs.add_hyperparameter(relief)

    relief_NN = UniformIntegerHyperparameter('ReliefNN',
                                             lower=parameters['Featsel']['ReliefNN'][0],
                                             upper=parameters['Featsel']['ReliefNN'][0] +
                                                   parameters['Featsel']['ReliefNN'][1])
    cs.add_hyperparameter(relief_NN)
    cs.add_condition(InCondition(child=relief_NN, parent=relief, values=['True']))

    relief_sample_size = UniformFloatHyperparameter('ReliefSampleSize',
                                                    lower=parameters['Featsel']['ReliefSampleSize'][0],
                                                    upper=parameters['Featsel']['ReliefSampleSize'][0] +
                                                          parameters['Featsel']['ReliefSampleSize'][1])
    cs.add_hyperparameter(relief_sample_size)
    cs.add_condition(InCondition(child=relief_sample_size, parent=relief, values=['True']))

    relief_distanceP = UniformIntegerHyperparameter('ReliefDistanceP',
                                                      lower=parameters['Featsel']['ReliefDistanceP'][0],
                                                      upper=parameters['Featsel']['ReliefDistanceP'][0] +
                                                            parameters['Featsel']['ReliefDistanceP'][1])
    cs.add_hyperparameter(relief_distanceP)
    cs.add_condition(InCondition(child=relief_distanceP, parent=relief, values=['True']))

    relief_numFeatures = UniformIntegerHyperparameter('ReliefNumFeatures',
                                                      lower=parameters['Featsel']['ReliefNumFeatures'][0],
                                                      upper=parameters['Featsel']['ReliefNumFeatures'][0] +
                                                            parameters['Featsel']['ReliefNumFeatures'][1])
    cs.add_hyperparameter(relief_numFeatures)
    cs.add_condition(InCondition(child=relief_numFeatures, parent=relief, values=['True']))

    # Select from model --> turned off in RS
    # 3 hyperparameters
    #   1) estimator
    #   2) lasso_alpha      | conditional on estimator: lasso
    #   3) rf_n_trees       | conditional on estimator: rf
    select_from_model = CategoricalHyperparameter('SelectFromModel', choices=['True', 'False'])
    cs.add_hyperparameter(select_from_model)

    estimator = CategoricalHyperparameter('SelectFromModel_estimator', choices=parameters['Featsel']['SelectFromModel_estimator'])
    lasso_alpha = UniformFloatHyperparameter('SelectFromModel_lasso_alpha',
                                             lower=parameters['Featsel']['SelectFromModel_lasso_alpha'][0],
                                             upper=parameters['Featsel']['SelectFromModel_lasso_alpha'][0] +
                                                   parameters['Featsel']['SelectFromModel_lasso_alpha'][1])
    n_trees = UniformIntegerHyperparameter('SelectFromModel_n_trees',
                                           lower=parameters['Featsel']['SelectFromModel_n_trees'][0],
                                           upper=parameters['Featsel']['SelectFromModel_n_trees'][0] + parameters['Featsel']['SelectFromModel_n_trees'][1])
    cs.add_hyperparameters([estimator, lasso_alpha, n_trees])
    cs.add_conditions([InCondition(child=estimator, parent=select_from_model, values=['True']),
                       InCondition(child=lasso_alpha, parent=estimator, values=['Lasso']),
                       InCondition(child=n_trees, parent=estimator, values=['RF'])])

    # PCA
    # 2 hyperparameters:
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

    # Statistical test
    # 2 hyperparameters:
    #   1) Metric
    #   2) Threshold
    statistical_test = CategoricalHyperparameter('StatisticalTestUse', choices=['False', 'True'])
    cs.add_hyperparameter(statistical_test)

    statistical_test_metric = CategoricalHyperparameter('StatisticalTestMetric',
                                                        choices=parameters['Featsel']['StatisticalTestMetric'])
    cs.add_hyperparameter(statistical_test_metric)
    cs.add_condition(InCondition(child=statistical_test_metric, parent=statistical_test,
                                 values=['True']))

    statistical_test_threshold = UniformFloatHyperparameter('StatisticalTestThreshold',
                                                            lower=pow(10, parameters['Featsel']['StatisticalTestThreshold'][0]),
                                                            upper=pow(10, parameters['Featsel']['StatisticalTestThreshold'][0] +
                                                                  parameters['Featsel']['StatisticalTestThreshold'][1]),
                                                            log=True)
    cs.add_hyperparameter(statistical_test_threshold)
    cs.add_condition(InCondition(child=statistical_test_threshold, parent=statistical_test,
                                 values=['True']))

    # Groupwise feature selection
    groupwise_search = CategoricalHyperparameter('SelectGroups', choices=parameters['Featsel']['GroupwiseSearch'])
    cs.add_hyperparameter(groupwise_search)

    for group in parameters['SelectFeatGroup'].keys():
        group_parameter = CategoricalHyperparameter(group,
                                                    choices=parameters['SelectFeatGroup'][group])
        cs.add_hyperparameter(group_parameter)
        cs.add_condition(InCondition(child=group_parameter, parent=groupwise_search,
                                     values=['True']))

    # Resampling
    # 2 hyperparameters:
    #     1. method
    #     2. sampling_strategy | conditional on method
    #     3. n_neighbors | conditional on method
    #     4. k_neighbors | conditional on method
    #     5. threshold_cleaning | conditional on method

    resampling = CategoricalHyperparameter('Resampling_Use', choices=['True', 'False'])
    cs.add_hyperparameter(resampling)

    resampling_method = CategoricalHyperparameter('Resampling_Method',
                                                    choices=parameters['Resampling']['Method'])
    cs.add_hyperparameter(resampling_method)
    cs.add_condition(InCondition(child=resampling_method, parent=resampling,
                                 values=['True']))

    resampling_strategy = CategoricalHyperparameter('Resampling_sampling_strategy',
                                                    choices=parameters['Resampling']['sampling_strategy'])
    cs.add_hyperparameter(resampling_strategy)
    cs.add_condition(InCondition(child=resampling_strategy, parent=resampling_method,
                                 values=['RandomUnderSampling', 'NearMiss',
                                         'NeighbourhoodCleaningRule',
                                         'RandomOverSampling',
                                         'ADASYN',
                                         'SMOTEENN',
                                         'SMOTETomek']))

    resampling_n_neighbors = UniformIntegerHyperparameter('Resampling_n_neighbors',
                                                          lower=parameters['Resampling']['n_neighbors'][0],
                                                          upper=parameters['Resampling']['n_neighbors'][0] +
                                                                parameters['Resampling']['n_neighbors'][1])
    cs.add_hyperparameter(resampling_n_neighbors)
    cs.add_condition(InCondition(child=resampling_n_neighbors, parent=resampling_method,
                                 values=['NeighbourhoodCleaningRule', 'ADASYN']))

    resampling_k_neighbors = UniformIntegerHyperparameter('Resampling_k_neighbors',
                                                          lower=parameters['Resampling']['k_neighbors'][0],
                                                          upper=parameters['Resampling']['k_neighbors'][0] +
                                                                parameters['Resampling']['k_neighbors'][1])
    cs.add_hyperparameter(resampling_k_neighbors)
    cs.add_condition(InCondition(child=resampling_k_neighbors, parent=resampling_method,
                                 values=['BorderlineSMOTE']))

    resampling_threshold_cleaning = UniformFloatHyperparameter('Resampling_threshold_cleaning',
                                                            lower=parameters['Resampling']['threshold_cleaning'][0],
                                                            upper=parameters['Resampling']['threshold_cleaning'][0] +
                                                                  parameters['Resampling']['threshold_cleaning'][1])
    cs.add_hyperparameter(resampling_threshold_cleaning)
    cs.add_condition(InCondition(child=resampling_threshold_cleaning, parent=resampling_method,
                                 values=['NeighbourhoodCleaningRule']))

    # Other parameters not part of the optimization but are needed in
    # fit_and_score workflow execution function
    random_seed = Constant('random_seed', value=parameters['Other']['random_seed'])
    cs.add_hyperparameter(random_seed)

    max_iter = Constant('max_iter', value=cf['max_iter'][0])
    cs.add_hyperparameter(max_iter)

    n_cores = Constant('Resampling_n_cores', value=parameters['General']['Joblib_ncores'])
    cs.add_hyperparameter(n_cores)

    return cs
