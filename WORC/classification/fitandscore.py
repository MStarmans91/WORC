#!/usr/bin/env python

# Copyright 2016-2023 Biomedical Imaging Group Rotterdam, Departments of
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

from sklearn.model_selection._validation import _fit_and_score
import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from WORC.classification.ObjectSampler import ObjectSampler
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import _num_samples
from WORC.classification import construct_classifier as cc
from WORC.classification.metrics import check_multimetric_scoring
from WORC.featureprocessing.Relief import SelectMulticlassRelief
from WORC.featureprocessing.Imputer import Imputer
from WORC.featureprocessing.Scalers import WORCScaler
from WORC.featureprocessing.VarianceThreshold import selfeat_variance
from WORC.featureprocessing.StatisticalTestThreshold import StatisticalTestThreshold
from WORC.featureprocessing.SelectGroups import SelectGroups
from WORC.featureprocessing.OneHotEncoderWrapper import OneHotEncoderWrapper
import WORC
import WORC.addexceptions as ae
import time
from xgboost.sklearn import XGBRegressor

# Specific imports for error management
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from numpy.linalg import LinAlgError

# Suppress some sklearn warnings. These occur when unused hyperparameters are
# supplied, when estimators that are refitted do not converge, or parts
# are deprecated
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def fit_and_score(X, y, scoring,
                  train, test, parameters,
                  fit_params=None,
                  return_train_score=True,
                  return_n_test_samples=True,
                  return_times=True, return_parameters=False,
                  return_estimator=False,
                  error_score='raise', verbose=False,
                  return_all=True, refit_training_workflows=False,
                  refit_validation_workflows=False,
                  skip=False):
    """Fit an estimator to a dataset and score the performance.

    The following
    methods can currently be applied as preprocessing before fitting, in
    this order:
    0. Apply OneHotEncoder
    1. Apply feature imputation
    2. Select features based on feature type group (e.g. shape, histogram).
    3. Scale features with e.g. z-scoring.
    4. Apply feature selection based on variance of feature among patients.
    5. Univariate statistical testing (e.g. t-test, Wilcoxon).
    6. Use Relief feature selection.
    7. Select features based on a fit with a LASSO model.
    8. Select features using PCA.
    9. Resampling
    10. If a SingleLabel classifier is used for a MultiLabel problem,
        a OneVsRestClassifier is employed around it.

    All of the steps are optional.

    Parameters
    ----------
    estimator: sklearn estimator, mandatory
            Unfitted estimator which will be fit.

    X: array, mandatory
            Array containingfor each object (rows) the feature values
            (1st Column) and the associated feature label (2nd Column).

    y: list(?), mandatory
            List containing the labels of the objects.

    scorer: sklearn scorer, mandatory
            Function used as optimization criterion for the hyperparamater optimization.

    train: list, mandatory
            Indices of the objects to be used as training set.

    test: list, mandatory
            Indices of the objects to be used as testing set.

    parameters: dictionary, mandatory
            Contains the settings used for the above preprocessing functions
            and the fitting. TODO: Create a default object and show the
            fields.

    fit_params:dictionary, default None
            Parameters supplied to the estimator for fitting. See the SKlearn
            site for the parameters of the estimators.

    return_train_score: boolean, default True
            Save the training score to the final SearchCV object.

    return_n_test_samples: boolean, default True
            Save the number of times each sample was used in the test set
            to the final SearchCV object.

    return_times: boolean, default True
            Save the time spend for each fit to the final SearchCV object.

    return_parameters: boolean, default True
            Return the parameters used in the final fit to the final SearchCV
            object.

    return_estimator : bool, default=False
        Whether to return the fitted estimator.

    error_score: numeric or "raise" by default
            Value to assign to the score if an error occurs in estimator
            fitting. If set to "raise", the error is raised. If a numeric
            value is given, FitFailedWarning is raised. This parameter
            does not affect the refit step, which will always raise the error.

    verbose: boolean, default=True
            If True, print intermediate progress to command line. Warnings are
            always printed.

    return_all: boolean, default=True
            If False, only the ret object containing the performance will be
            returned. If True, the ret object plus all fitted objects will be
            returned.

    Returns
    ----------
    Depending on the return_all input parameter, either only ret or all objects
    below are returned.

    ret: list
        Contains optionally the train_scores and the test_scores,
        fit_time, score_time, parameters_est
        and parameters_all.

    GroupSel: WORC GroupSel Object
        Either None if the groupwise feature selection is not used, or
        the fitted object.

    VarSel: WORC VarSel Object
        Either None if the variance threshold feature selection is not used, or
        the fitted object.

    SelectModel: WORC SelectModel Object
        Either None if the feature selection based on a fittd model is not
        used, or the fitted object.

    feature_labels: list
        Labels of the features. Only one list is returned, not one per
        feature object, as we assume all samples have the same feature names.

    scaler: scaler object
        Either None if feature scaling is not used, or
        the fitted object.

    encoder: WORC Encoder Object
        Either None if feature OneHotEncoding is not used, or
        the fitted object.

    imputer: WORC Imputater Object
        Either None if feature imputation is not used, or
        the fitted object.

    pca: WORC PCA Object
        Either None if PCA based feature selection is not used, or
        the fitted object.

    StatisticalSel: WORC StatisticalSel Object
        Either None if the statistical test feature selection is not used, or
        the fitted object.

    RFESel: WORC RFESel Object
        Either None if the recursive feature elimination feature selection is not used, or
        the fitted object.
        
    ReliefSel: WORC ReliefSel Object
        Either None if the RELIEF feature selection is not used, or
        the fitted object.

    Sampler: WORC ObjectSampler Object
        Either None if no resampling is used, or an ObjectSampler object


    """
    # We copy the parameter object so we can alter it and keep the original
    if verbose:
        print("\n")
        print('#######################################')
        print('Starting fit and score of new workflow.')

    para_estimator = parameters.copy()
    estimator = cc.construct_classifier(para_estimator)

    # Check the scorer
    scorers, __ = check_multimetric_scoring(estimator, scoring=scoring)

    para_estimator = delete_cc_para(para_estimator)

    # Get random seed from parameters
    random_seed = para_estimator['random_seed']
    del para_estimator['random_seed']

    # X is a tuple: split in two arrays
    feature_values = np.asarray([x[0] for x in X])
    feature_labels = np.asarray([x[1] for x in X])

    # Split in train and testing
    X_train, y_train = _safe_split(estimator, feature_values, y, train)
    X_test, y_test = _safe_split(estimator, feature_values, y, test, train)
    new_train = np.arange(0, len(y_train))
    new_test = np.arange(len(y_train), len(y_train) + len(y_test))

    # Set some defaults for if a part fails and we return a dummy
    fit_time = np.inf
    score_time = np.inf
    Sampler = None
    encoder = None
    imputer = None
    scaler = None
    GroupSel = None
    SelectModel = None
    pca = None
    StatisticalSel = None
    RFESel = None
    VarSel = None
    ReliefSel = None
    if isinstance(scorers, dict):
        test_scores = {name: np.nan for name in scorers}
        if return_train_score:
            train_scores = test_scores.copy()
    else:
        test_scores = error_score
        if return_train_score:
            train_scores = error_score

    # Initiate dummy return object for when fit and scoring failes: sklearn defaults
    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(para_estimator)
    if return_estimator:
        ret.append(estimator)

    # Additional to sklearn defaults: return all parameters and refitted estimator
    ret.append(parameters)

    if refit_training_workflows:
        ret.append(None)
        
    if refit_validation_workflows:
        ret.append(None)

    # ------------------------------------------------------------------------
    # OneHotEncoder
    if 'OneHotEncoding' in para_estimator.keys():
        if para_estimator['OneHotEncoding'] == 'True':
            if verbose:
                print(f'Applying OneHotEncoding, will ignore unknowns.')
            feature_labels_tofit =\
                para_estimator['OneHotEncoding_feature_labels_tofit']
            encoder =\
                OneHotEncoderWrapper(handle_unknown='ignore',
                                     feature_labels_tofit=feature_labels_tofit,
                                     verbose=verbose)
            encoder.fit(X_train, feature_labels)

            if encoder.encoder is not None:
                # Encoder is fitted
                feature_labels = encoder.encoder.encoded_feature_labels
                X_train = encoder.transform(X_train)
                X_test = encoder.transform(X_test)

        del para_estimator['OneHotEncoding']
        del para_estimator['OneHotEncoding_feature_labels_tofit']

    # Delete the object if we do not need to return it
    if not return_all:
        del encoder

    # Start the timing
    start_time = time.time()

    # ------------------------------------------------------------------------
    # Feature imputation
    if 'Imputation' in para_estimator.keys():
        if para_estimator['Imputation'] == 'True':
            imp_type = para_estimator['ImputationMethod']
            if verbose:
                print(f'Imputing NaN with {imp_type}.')
                
            # Only used with KNN in SMAC, otherwise assign default
            if 'ImputationNeighbours' in para_estimator.keys():
                imp_nn = para_estimator['ImputationNeighbours']
            else:
                imp_nn = 8

            imputer = Imputer(missing_values=np.nan, strategy=imp_type,
                              n_neighbors=imp_nn)
            imputer.fit(X_train)

            original_shape = X_train.shape
            imputed_shape = imputer.transform(X_train).shape
            
            if original_shape != imputed_shape:
                removed_features = original_shape[1] - imputed_shape[1]
                if para_estimator['ImputationSkipAllNaN'] == 'True':
                    print(f"[WARNING]: Several features ({removed_features}) were np.NaN for all objects. config['Imputation']['skipallNaN'] set to True, so simply eliminate these features.")
                    if hasattr(imputer.Imputer, 'statistics_'):
                        X_train = imputer.transform(X_train)
                        X_test = imputer.transform(X_test)
                        feature_labels_zero = [fl for fnum, fl in enumerate(feature_labels[0]) if not np.isnan(imputer.Imputer.statistics_[fnum])]
                        feature_labels = [feature_labels_zero for i in X_train]
                    else:
                        # Fit a mean imputer to transform the labels
                        temp_imputer = Imputer(missing_values=np.nan, strategy='mean')
                        temp_imputer.fit(X_train)
                        X_train = imputer.transform(X_train)
                        X_test = imputer.transform(X_test)
                        feature_labels_zero = [fl for fnum, fl in enumerate(feature_labels[0]) if not np.isnan(temp_imputer.Imputer.statistics_[fnum])]
                        feature_labels = [feature_labels_zero for i in X_train]
                else:
                    raise ae.WORCValueError(f'Several features ({removed_features}) were np.NaN for all objects. Hence, imputation was not possible. Either make sure this is correct and turn of imputation, or correct the feature.')
            else:
                X_train = imputer.transform(X_train)
                X_test = imputer.transform(X_test)

        if 'ImputationSkipAllNaN' in para_estimator.keys():
            del para_estimator['ImputationSkipAllNaN']
            
        del para_estimator['Imputation']
        del para_estimator['ImputationMethod']
        if 'ImputationNeighbours' in para_estimator.keys():
            del para_estimator['ImputationNeighbours']

    # Delete the object if we do not need to return it
    if not return_all:
        del imputer

    # Remove any NaN feature values if these are still left after imputation
    X_train = replacenan(X_train, verbose=verbose, feature_labels=feature_labels[0])
    X_test = replacenan(X_test, verbose=verbose, feature_labels=feature_labels[0])

    # ------------------------------------------------------------------------
    # Groupwise feature selection
    if 'SelectGroups' in para_estimator:
        if verbose:
            print("Selecting groups of features.")
        del para_estimator['SelectGroups']
        # TODO: more elegant way to solve this
        feature_groups = ['shape_features',
                          'histogram_features',
                          'orientation_features',
                          'texture_gabor_features',
                          'texture_glcm_features',
                          'texture_gldm_features',
                          'texture_glcmms_features',
                          'texture_glrlm_features',
                          'texture_glszm_features',
                          'texture_gldzm_features',
                          'texture_ngtdm_features',
                          'texture_ngldm_features',
                          'texture_lbp_features',
                          'dicom_features',
                          'semantic_features',
                          'coliage_features',
                          'vessel_features',
                          'phase_features',
                          'fractal_features',
                          'location_features',
                          'rgrd_features',
                          'original_features',
                          'wavelet_features',
                          'log_features']

        # First take out the toolbox selection, which is a list
        toolboxes = para_estimator['toolbox']
        del para_estimator['toolbox']

        # Check per feature group if the parameter is present
        parameters_featsel = dict()
        for group in feature_groups:
            if group not in para_estimator:
                # Default: do use the group, except for texture features
                if group == 'texture_features':
                    value = 'False'
                else:
                    value = 'True'
            else:
                value = para_estimator[group]
                del para_estimator[group]

            parameters_featsel[group] = value

        # Fit groupwise feature selection object
        GroupSel = SelectGroups(parameters=parameters_featsel,
                                toolboxes=toolboxes)
        GroupSel.fit(feature_labels[0])
        if verbose:
            print("\t Original Length: " + str(len(X_train[0])))

        # Transform all objectd accordingly
        X_train = GroupSel.transform(X_train)
        X_test = GroupSel.transform(X_test)
        if verbose:
            print("\t New Length: " + str(len(X_train[0])))
        feature_labels = GroupSel.transform(feature_labels)

    # Delete the object if we do not need to return it
    if not return_all:
        del GroupSel

    # Check whether there are any features left
    if len(X_train[0]) == 0:
        # TODO: Make a specific WORC exception for this warning.
        if verbose:
            print('[WARNING]: No features are selected! Probably all feature groups were set to False. Parameters:')
            print(parameters)

        # Delete the non-used fields
        para_estimator = delete_nonestimator_parameters(para_estimator)

        # Update the runtime
        end_time = time.time()
        runtime = end_time - start_time
        if return_train_score:
            ret[3] = runtime
        else:
            ret[2] = runtime

        if return_all:
            return ret, GroupSel, VarSel, SelectModel, feature_labels[0], scaler, encoder, imputer, pca, StatisticalSel, RFESel, ReliefSel, Sampler
        else:
            return ret

    # --------------------------------------------------------------------
    # Feature selection based on variance
    if para_estimator['Featsel_Variance'] == 'True':
        if verbose:
            print("Selecting features based on variance.")
        if verbose:
            print("\t Original Length: " + str(len(X_train[0])))
        try:
            X_train, feature_labels, VarSel =\
                selfeat_variance(X_train, feature_labels)
            X_test = VarSel.transform(X_test)
        except ValueError:
            if verbose:
                print('[WARNING]: No features meet the selected variance threshold.')

            VarSel = None
            if skip:
                if verbose:
                    print('[WARNING] Refitting, so we need an estimator, thus skipping this step.')
                parameters['Featsel_Variance'] = 'False'
            else:
                if verbose:
                    print('[WARNING] Returning NaN as performance.')
                
                # return NaN as performance
                para_estimator = delete_nonestimator_parameters(para_estimator)
                
                # Update the runtime
                end_time = time.time()
                runtime = end_time - start_time
                if return_train_score:
                    ret[3] = runtime
                else:
                    ret[2] = runtime
                if return_all:
                    return ret, GroupSel, VarSel, SelectModel,\
                        feature_labels[0], scaler, encoder, imputer, pca,\
                        StatisticalSel, RFESel, ReliefSel, Sampler
                else:
                    return ret
            
        if verbose:
            print("\t New Length: " + str(len(X_train[0])))

    del para_estimator['Featsel_Variance']

    # Delete the object if we do not need to return it
    if not return_all:
        del VarSel

    # ------------------------------------------------------------------------
    # Feature scaling
    if verbose and para_estimator['FeatureScaling'] != 'None':
        print('Fitting scaler and transforming features, method ' +
              f'{para_estimator["FeatureScaling"]}.')

    scaling_method = para_estimator['FeatureScaling']
    if scaling_method == 'None':
        scaler = None
    else:
        skip_features = para_estimator['FeatureScaling_skip_features']
        n_skip_feat = len([i for i in feature_labels[0] if any(e in i for e in skip_features)])
        if n_skip_feat == len(X_train[0]):
            # Don't need to scale any features
            if verbose:
                print('[WARNING] Skipping scaling, only skip features selected.')
            scaler = None
        else:
            scaler = WORCScaler(method=scaling_method, skip_features=skip_features)
            scaler.fit(X_train, feature_labels[0])

    if scaler is not None:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    del para_estimator['FeatureScaling']
    del para_estimator['FeatureScaling_skip_features']

    # Delete the object if we do not need to return it
    if not return_all:
        del scaler

    # --------------------------------------------------------------------
    # Relief feature selection, possibly multi class.
    # Needs to be done after scaling!
    # para_estimator['ReliefUse'] = 'True'
    if 'ReliefUse' in para_estimator.keys():
        if para_estimator['ReliefUse'] == 'True':
            if verbose:
                print("Selecting features using relief.")

            # Get parameters from para_estimator
            n_neighbours = para_estimator['ReliefNN']
            sample_size = para_estimator['ReliefSampleSize']
            distance_p = para_estimator['ReliefDistanceP']
            numf = para_estimator['ReliefNumFeatures']

            # Fit RELIEF object
            ReliefSel = SelectMulticlassRelief(n_neighbours=n_neighbours,
                                               sample_size=sample_size,
                                               distance_p=distance_p,
                                               numf=numf,
                                               random_state=random_seed)
            ReliefSel.fit(X_train, y)
            if verbose:
                print("\t Original Length: " + str(len(X_train[0])))

            # Transform all objects accordingly
            X_train_temp = ReliefSel.transform(X_train)
            if len(X_train_temp[0]) == 0:
                if verbose:
                    print('[WARNING]: No features are selected! Probably RELIEF could not properly select features.')
            
                ReliefSel = None
                if skip:
                    if verbose:
                        print('[WARNING] Refitting, so we need an estimator, thus skipping this step.')
                    parameters['ReliefUse'] = 'False'
                else:
                    if verbose:
                        print('[WARNING] Returning NaN as performance.')
                    
                    # return NaN as performance
                    para_estimator = delete_nonestimator_parameters(para_estimator)
                    
                    # Update the runtime
                    end_time = time.time()
                    runtime = end_time - start_time
                    if return_train_score:
                        ret[3] = runtime
                    else:
                        ret[2] = runtime
                    if return_all:
                        return ret, GroupSel, VarSel, SelectModel,\
                            feature_labels[0], scaler, encoder, imputer, pca,\
                            StatisticalSel, RFESel, ReliefSel, Sampler
                    else:
                        return ret
            else:
                X_train = X_train_temp
                X_test = ReliefSel.transform(X_test)

                if verbose:
                    print("\t New Length: " + str(len(X_train[0])))
                feature_labels = ReliefSel.transform(feature_labels)

        del para_estimator['ReliefUse']
        del para_estimator['ReliefNN']
        del para_estimator['ReliefSampleSize']
        del para_estimator['ReliefDistanceP']
        del para_estimator['ReliefNumFeatures']

    # Delete the object if we do not need to return it
    if not return_all:
        del ReliefSel

    # ------------------------------------------------------------------------
    # Perform feature selection using a model
    if 'SelectFromModel' in para_estimator.keys():
        if para_estimator['SelectFromModel'] == 'True':
            model = para_estimator['SelectFromModel_estimator']
            if verbose:
                print(f"Selecting features using model {model}.")

            if model == 'Lasso':
                # Use lasso model for feature selection
                alpha = para_estimator['SelectFromModel_lasso_alpha']
                selectestimator = Lasso(alpha=alpha, random_state=random_seed)

            elif model == 'LR':
                # Use logistic regression model for feature selection
                selectestimator = LogisticRegression(random_state=random_seed)

            elif model == 'RF':
                # Use random forest model for feature selection
                n_estimators = para_estimator['SelectFromModel_n_trees']
                selectestimator = RandomForestClassifier(n_estimators=n_estimators,
                                                         random_state=random_seed)
            else:
                raise ae.WORCKeyError(f'Model {model} is not known for SelectFromModel. Use Lasso, LR, or RF.')

            if len(y_train.shape) >= 2:
                # Multilabel or regression. Regression: second dimension has length 1
                if y_train.shape[1] > 1 and model != 'RF':
                    raise ae.WORCValueError(f'Model {model} is not suitable for multiclass classification. Please use RF or do not use SelectFromModel.')

            # Prefit model
            selectestimator.fit(X_train, y_train)

            # Use fit to select optimal features
            SelectModel = SelectFromModel(selectestimator, prefit=True)
            if verbose:
                print("\t Original Length: " + str(len(X_train[0])))

            X_train_temp = SelectModel.transform(X_train)
            if len(X_train_temp[0]) == 0:
                if verbose:
                    print('[WARNING]: No features are selected! Probably your data is too noisy or the selection too strict.')
                
                SelectModel = None
                if skip:
                    if verbose:
                        print('[WARNING] Refitting, so we need an estimator, thus skipping this step.')
                    parameters['SelectFromModel'] = 'False'
                else:
                    if verbose:
                        print('[WARNING] Returning NaN as performance.')
                    
                    # return NaN as performance
                    para_estimator = delete_nonestimator_parameters(para_estimator)
                    
                    # Update the runtime
                    end_time = time.time()
                    runtime = end_time - start_time
                    if return_train_score:
                        ret[3] = runtime
                    else:
                        ret[2] = runtime
                    if return_all:
                        return ret, GroupSel, VarSel, SelectModel,\
                            feature_labels[0], scaler, encoder, imputer, pca,\
                            StatisticalSel, RFESel, ReliefSel, Sampler
                    else:
                        return ret
                    
            else:
                X_train = SelectModel.transform(X_train)
                X_test = SelectModel.transform(X_test)
                feature_labels = SelectModel.transform(feature_labels)

                if verbose:
                    print("\t New Length: " + str(len(X_train[0])))

        del para_estimator['SelectFromModel']
        if 'SelectFromModel_lasso_alpha' in para_estimator.keys():
            del para_estimator['SelectFromModel_lasso_alpha']
        if 'SelectFromModel_estimator' in para_estimator.keys():
            del para_estimator['SelectFromModel_estimator']
        if 'SelectFromModel_n_trees' in para_estimator.keys():
            del para_estimator['SelectFromModel_n_trees']

    # Delete the object if we do not need to return it
    if not return_all:
        del SelectModel

    # --------------------------------------------------------------------
    # Feature selection based on a statistical test
    if 'StatisticalTestUse' in para_estimator.keys():
        if para_estimator['StatisticalTestUse'] == 'True':
            metric = para_estimator['StatisticalTestMetric']
            threshold = para_estimator['StatisticalTestThreshold']
            if verbose:
                print(f"Selecting features based on statistical test. Method {metric}, threshold {round(threshold, 5)}.")
                print("\t Original Length: " + str(len(X_train[0])))

            StatisticalSel = StatisticalTestThreshold(metric=metric,
                                                      threshold=threshold)

            StatisticalSel.fit(X_train, y)
            X_train_temp = StatisticalSel.transform(X_train)
            if len(X_train_temp[0]) == 0:   
                if verbose:
                    print('[WARNING] No features are selected! Probably your statistical test feature selection was too strict.')
                         
                StatisticalSel = None           
                if skip:
                    if verbose:
                        print('[WARNING] Refitting, so we need an estimator, thus skipping this step.')
                    parameters['StatisticalTestUse'] = 'False'
                else:
                    if verbose:
                        print('[WARNING] Returning NaN as performance.')
                    
                    # return NaN as performance
                    para_estimator = delete_nonestimator_parameters(para_estimator)
                    
                    # Update the runtime
                    end_time = time.time()
                    runtime = end_time - start_time
                    if return_train_score:
                        ret[3] = runtime
                    else:
                        ret[2] = runtime
                    if return_all:
                        return ret, GroupSel, VarSel, SelectModel,\
                            feature_labels[0], scaler, encoder, imputer, pca,\
                            StatisticalSel, RFESel, ReliefSel, Sampler
                    else:
                        return ret
                        
            else:
                X_train = StatisticalSel.transform(X_train)
                X_test = StatisticalSel.transform(X_test)
                feature_labels = StatisticalSel.transform(feature_labels)

            if verbose:
                print("\t New Length: " + str(len(X_train[0])))

        # Delete the statistical test keys
        del para_estimator['StatisticalTestUse']
        if 'StatisticalTestMetric' in para_estimator.keys():
            del para_estimator['StatisticalTestMetric']

        if 'StatisticalTestThreshold' in para_estimator.keys():
            del para_estimator['StatisticalTestThreshold']

    # Delete the object if we do not need to return it
    if not return_all:
        del StatisticalSel
    
    # --------------------------------------------------------------------
    # Feature selection through recursive feature elimination
    if 'RFE' in para_estimator.keys():
        model = para_estimator['RFE_estimator']
        if para_estimator['RFE'] == 'True':
            if verbose:
                print(f"Selecting features using recursive feature elimination using model {model}.")
        
            if model == 'Lasso':
                # Use lasso model for feature selection
                alpha = para_estimator['RFE_lasso_alpha']
                selectestimator = Lasso(alpha=alpha, random_state=random_seed)

            elif model == 'LR':
                # Use logistic regression model for feature selection
                selectestimator = LogisticRegression(random_state=random_seed)

            elif model == 'RF':
                # Use random forest model for feature selection
                n_estimators = para_estimator['RFE_n_trees']
                selectestimator = RandomForestClassifier(n_estimators=n_estimators,
                                                         random_state=random_seed)
            else:
                raise ae.WORCKeyError(f'Model {model} is not known for RFE. Use Lasso, LR, or RF.')
            
            if len(y_train.shape) >= 2:
                # Multilabel or regression. Regression: second dimension has length 1
                if y_train.shape[1] > 1 and model != 'RF':
                    raise ae.WORCValueError(f'Model {model} is not suitable for multiclass classification. Please use RF or do not use RFE.')

            # Prefit model
            selectestimator.fit(X_train, y_train)

            # Use fit to select optimal features
            n_features_to_select = para_estimator['RFE_n_features_to_select']
            step = para_estimator['RFE_step']
            RFESel = RFE(selectestimator,
                         n_features_to_select=n_features_to_select,
                         step=step)
            try:
                RFESel.fit(X_train, y_train)
            except ValueError:
                if skip:
                    if verbose:
                        print('[WARNING] Refitting, so we need an estimator, thus skipping this step.')
                    parameters['RFE'] = 'False'
                    
                else:
                    if verbose:
                        print('[WARNING] RFE cannot be fitted with these settings, too few features left, returning NaN as performance.')
                    
                    # return NaN as performance
                    para_estimator = delete_nonestimator_parameters(para_estimator)
                    RFESel = None
                    
                    # Update the runtime
                    end_time = time.time()
                    runtime = end_time - start_time
                    if return_train_score:
                        ret[3] = runtime
                    else:
                        ret[2] = runtime
                    if return_all:
                        return ret, GroupSel, VarSel, SelectModel,\
                            feature_labels[0], scaler, encoder, imputer, pca,\
                            StatisticalSel, RFESel, ReliefSel, Sampler
                    else:
                        return ret
            else:
                if verbose:
                    print("\t Original Length: " + str(len(X_train[0])))

                X_train_temp = RFESel.transform(X_train)
                if len(X_train_temp[0]) == 0:
                    if verbose:
                        print('[WARNING]: No features are selected! Probably your data is too noisy or the selection too strict.')
                    
                    RFESel = None
                    if skip:
                        if verbose:
                            print('[WARNING] Refitting, so we need an estimator, thus skipping this step.')
                        parameters['RFE'] = 'False'
                    else:
                        if verbose:
                            print('[WARNING] Returning NaN as performance.')
                        
                        # return NaN as performance
                        para_estimator = delete_nonestimator_parameters(para_estimator)
                        
                        # Update the runtime
                        end_time = time.time()
                        runtime = end_time - start_time
                        if return_train_score:
                            ret[3] = runtime
                        else:
                            ret[2] = runtime
                        if return_all:
                            return ret, GroupSel, VarSel, SelectModel,\
                                feature_labels[0], scaler, encoder, imputer, pca,\
                                StatisticalSel, RFESel, ReliefSel, Sampler
                        else:
                            return ret
                        
                else:
                    X_train = RFESel.transform(X_train)
                    X_test = RFESel.transform(X_test)
                    feature_labels = RFESel.transform(feature_labels)

                    if verbose:
                        print("\t New Length: " + str(len(X_train[0])))

        del para_estimator['RFE']
        if 'RFE_lasso_alpha' in para_estimator.keys():
            del para_estimator['RFE_lasso_alpha']
        if 'RFE_estimator' in para_estimator.keys():
            del para_estimator['RFE_estimator']
        if 'RFE_n_trees' in para_estimator.keys():
            del para_estimator['RFE_n_trees']
        if 'RFE_n_features_to_select' in para_estimator.keys():
            del para_estimator['RFE_n_features_to_select']
        if 'RFE_n_trees' in para_estimator.keys():
            del para_estimator['RFE_n_trees']
    
    # Delete the object if we do not need to return it
    if not return_all:
        del RFESel
    
    # ----------------------------------------------------------------
    # PCA dimensionality reduction
    # Principle Component Analysis
    if 'UsePCA' in para_estimator.keys() and para_estimator['UsePCA'] == 'True':
        if verbose:
            print('Fitting PCA')
            print("\t Original Length: " + str(len(X_train[0])))
        if para_estimator['PCAType'] == '95variance':
            # Select first X components that describe 95 percent of the explained variance
            pca = PCA(n_components=None, random_state=random_seed)
            try:
                pca.fit(X_train)
            except (ValueError, LinAlgError) as e:
                if verbose:
                    print(f'[WARNING] PCA Error: {e}.')

                pca = None
                if skip:
                    if verbose:
                        print('[WARNING] Refitting, so we need an estimator, thus skipping this step.')
                    parameters['UsePCA'] = 'False'
                else:
                    if verbose:
                        print('[WARNING] Returning NaN as performance.')
                    
                    # return NaN as performance
                    para_estimator = delete_nonestimator_parameters(para_estimator)
                    
                    # Update the runtime
                    end_time = time.time()
                    runtime = end_time - start_time
                    if return_train_score:
                        ret[3] = runtime
                    else:
                        ret[2] = runtime
                    if return_all:
                        return ret, GroupSel, VarSel, SelectModel,\
                            feature_labels[0], scaler, encoder, imputer, pca,\
                            StatisticalSel, RFESel, ReliefSel, Sampler
                    else:
                        return ret

            else:
                evariance = pca.explained_variance_ratio_
                num = 0
                sum = 0
                while sum < 0.95:
                    sum += evariance[num]
                    num += 1

                # Make a PCA based on the determined amound of components
                pca = PCA(n_components=num, random_state=random_seed)
                try:
                    pca.fit(X_train)
                except (ValueError, LinAlgError) as e:
                    if verbose:
                        print(f'[WARNING]: PCA Error: {e}.')

                    pca = None
                    if skip:
                        if verbose:
                            print('[WARNING] Refitting, so we need an estimator, thus skipping this step.')
                        parameters['UsePCA'] = 'False'
                    else:
                        if verbose:
                            print('[WARNING] Returning NaN as performance.')
                        
                        # return NaN as performance
                        para_estimator = delete_nonestimator_parameters(para_estimator)
                        
                        # Update the runtime
                        end_time = time.time()
                        runtime = end_time - start_time
                        if return_train_score:
                            ret[3] = runtime
                        else:
                            ret[2] = runtime
                        if return_all:
                            return ret, GroupSel, VarSel, SelectModel,\
                                feature_labels[0], scaler, encoder, imputer, pca,\
                                StatisticalSel, RFESel, ReliefSel, Sampler
                        else:
                            return ret
                else:
                    X_train = pca.transform(X_train)
                    X_test = pca.transform(X_test)

        else:
            # Assume a fixed number of components: cannot be larger than
            # n_samples
            n_components = min(len(X_train), int(para_estimator['PCAType']))

            if n_components >= len(X_train[0]):
                if verbose:
                    print(f"[WARNING] PCA n_components ({n_components})> n_features ({len(X_train[0])}): skipping PCA.")
            else:
                pca = PCA(n_components=n_components, random_state=random_seed)
                try:
                    pca.fit(X_train)
                    X_train = pca.transform(X_train)
                    X_test = pca.transform(X_test)
                except (ValueError, LinAlgError) as e:
                    if verbose:
                        print(f'[WARNING] PCA Error: {e}.')

                    pca = None
                    if skip:
                        if verbose:
                            print('[WARNING] Refitting, so we need an estimator, thus skipping this step.')
                        parameters['UsePCA'] = 'False'
                    else:
                        if verbose:
                            print('[WARNING] Returning NaN as performance.')
                        
                        # return NaN as performance
                        para_estimator = delete_nonestimator_parameters(para_estimator)
                        
                        # Update the runtime
                        end_time = time.time()
                        runtime = end_time - start_time
                        if return_train_score:
                            ret[3] = runtime
                        else:
                            ret[2] = runtime
                        if return_all:
                            return ret, GroupSel, VarSel, SelectModel,\
                                feature_labels[0], scaler, encoder, imputer, pca,\
                                StatisticalSel, RFESel, ReliefSel, Sampler
                        else:
                            return ret
                        
        if verbose:
            print("\t New Length: " + str(len(X_train[0])))

    # Delete the object if we do not need to return it
    if not return_all:
        del pca

    if 'UsePCA' in para_estimator.keys():
        del para_estimator['UsePCA']
        if 'PCAType' in para_estimator.keys():
            del para_estimator['PCAType']

    # ------------------------------------------------------------------------
    # Use object resampling
    if 'Resampling_Use' in para_estimator.keys():
        if para_estimator['Resampling_Use'] == 'True':

            # Determine our starting balance
            pos_initial = int(np.sum(y_train))
            neg_initial = int(len(y_train) - pos_initial)
            len_in = len(y_train)

            # If SMAC has removed a certain parameter, add a dummy altough
            # it's not actually used
            if 'Resampling_sampling_strategy' not in para_estimator.keys():
                para_estimator['Resampling_sampling_strategy'] = None

            if 'Resampling_n_neighbors' not in para_estimator.keys():
                para_estimator['Resampling_n_neighbors'] = None

            if 'Resampling_k_neighbors' not in para_estimator.keys():
                para_estimator['Resampling_k_neighbors'] = None

            if 'Resampling_threshold_cleaning' not in para_estimator.keys():
                para_estimator['Resampling_threshold_cleaning'] = None

            # Fit ObjectSampler and transform dataset
            Sampler =\
                ObjectSampler(method=para_estimator['Resampling_Method'],
                              sampling_strategy=para_estimator['Resampling_sampling_strategy'],
                              n_jobs=para_estimator['Resampling_n_cores'],
                              n_neighbors=para_estimator['Resampling_n_neighbors'],
                              k_neighbors=para_estimator['Resampling_k_neighbors'],
                              threshold_cleaning=para_estimator['Resampling_threshold_cleaning'],
                              verbose=verbose,
                              random_seed=random_seed)

            try:
                Sampler.fit(X_train, y_train)
                X_train_temp, y_train_temp = Sampler.transform(X_train, y_train)

            except ae.WORCValueError as e:
                message = str(e)
                if verbose:
                    print('[WARNING] Skipping resampling: ' + message)
                Sampler = None
                parameters['Resampling_Use'] = 'False'

            except RuntimeError as e:
                if 'ADASYN is not suited for this specific dataset. Use SMOTE instead.' in str(e):
                    # Seldomly occurs, therefore return performance dummy
                    if verbose:
                        print(f'[WARNING]: {e}. Returning dummies. Parameters: ')
                        print(parameters)
                    para_estimator = delete_nonestimator_parameters(para_estimator)

                    # Update the runtime
                    end_time = time.time()
                    runtime = end_time - start_time
                    if return_train_score:
                        ret[3] = runtime
                    else:
                        ret[2] = runtime

                    if return_all:
                        return ret, GroupSel, VarSel, SelectModel,\
                            feature_labels[0], scaler, encoder, imputer,\
                            pca, StatisticalSel, RFESel, ReliefSel, Sampler
                    else:
                        return ret
                else:
                    raise e
            else:
                pos = int(np.sum(y_train_temp))
                neg = int(len(y_train_temp) - pos)
                if pos < 10 or neg < 10:
                    if verbose:
                        print(f'[WARNING] Skipping resampling: to few objects returned in one or both classes (pos: {pos}, neg: {neg}).')
                        
                    Sampler = None
                    parameters['Resampling_Use'] = 'False'
                else:
                    X_train = X_train_temp
                    y_train = y_train_temp

                    # Notify the user what the resampling did
                    pos = int(np.sum(y_train))
                    neg = int(len(y_train) - pos)
                    if verbose:
                        message = f"Resampling from {len_in} ({pos_initial} pos," +\
                                  f" {neg_initial} neg) to {len(y_train)} ({pos} pos, {neg} neg) patients."
                        print(message)

                    # Also reset train and test indices
                    new_train = np.arange(0, len(y_train))
                    new_test = np.arange(len(y_train), len(y_train) + len(y_test))

        # Delete the resampling parameters
        del para_estimator['Resampling_Use']
        if 'Resampling_Method' in para_estimator.keys():
            del para_estimator['Resampling_Method']

        if 'Resampling_sampling_strategy' in para_estimator.keys():
            del para_estimator['Resampling_sampling_strategy']

        if 'Resampling_n_neighbors' in para_estimator.keys():
            del para_estimator['Resampling_n_neighbors']

        if 'Resampling_k_neighbors' in para_estimator.keys():
            del para_estimator['Resampling_k_neighbors']

        if 'Resampling_threshold_cleaning' in para_estimator.keys():
            del para_estimator['Resampling_threshold_cleaning']

        if 'Resampling_n_cores' in para_estimator.keys():
            del para_estimator['Resampling_n_cores']

    # Delete the object if we do not need to return it
    if not return_all:
        del Sampler

    # ----------------------------------------------------------------
    # Fitting and scoring
    # Only when using fastr this is an entry
    if 'Number' in para_estimator.keys():
        del para_estimator['Number']

    # For certainty, we delete all parameters again
    para_estimator = delete_nonestimator_parameters(para_estimator)

    # NOTE: This just has to go to the construct classifier function,
    # although it is more convenient here due to the hyperparameter search
    if type(y) is list:
        labellength = 1
    else:
        try:
            labellength = y.shape[1]
        except IndexError:
            labellength = 1

    if labellength > 1 and type(estimator) not in [RandomForestClassifier]:
        # Multiclass, hence employ a multiclass classifier for e.g. SVM, LR
        estimator.set_params(**para_estimator)
        estimator = OneVsRestClassifier(estimator)

    if verbose:
        print(f"Fitting ML method: {parameters['classifiers']}.")

    # Recombine feature values and label for train and test set
    feature_values = np.concatenate((X_train, X_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    para_estimator = None
        
    try:
        ret = _fit_and_score(estimator, feature_values, y_all,
                             scorers, new_train,
                             new_test, verbose,
                             para_estimator, fit_params,
                             return_train_score=return_train_score,
                             return_parameters=return_parameters,
                             return_n_test_samples=return_n_test_samples,
                             return_times=return_times,
                             return_estimator=return_estimator,
                             error_score=error_score)
    except (ValueError, LinAlgError) as e:
        if type(estimator) == LDA:
            if verbose:
                print(f'[WARNING]: skipping this setting due to LDA Error: {e}.')

            # Update the runtime
            end_time = time.time()
            runtime = end_time - start_time
            if return_train_score:
                ret[3] = runtime
            else:
                ret[2] = runtime

            if return_all:
                return ret, GroupSel, VarSel, SelectModel, feature_labels[0], scaler, encoder, imputer, pca, StatisticalSel, RFESel, ReliefSel, Sampler
            else:
                return ret
        else:
            raise e

    # Add original parameters to return object
    ret.append(parameters)

    if refit_training_workflows:
        # Refit estimator on train-test training dataset
        indices = np.arange(0, len(y))
        estimator = WORC.classification.SearchCV.RandomizedSearchCVfastr()
        estimator.refit_and_score(X, y, parameters,
                                  train=indices, test=indices)
        ret.append(estimator)
        
    if refit_validation_workflows:
        # Refit estimator on train-validation training dataset
        estimator = WORC.classification.SearchCV.RandomizedSearchCVfastr()
        estimator.refit_and_score(X, y, parameters,
                                  train=train, test=test)
        ret.append(estimator)
        
    # End the timing and store the fit_time
    end_time = time.time()
    runtime = end_time - start_time
    if return_train_score:
        ret[3] = runtime
    else:
        ret[2] = runtime

    if return_all:
        return ret, GroupSel, VarSel, SelectModel, feature_labels[0], scaler, encoder, imputer, pca, StatisticalSel, RFESel, ReliefSel, Sampler
    else:
        return ret


def delete_nonestimator_parameters(parameters):
    """Delete non-estimator parameters.

    Delete all parameters in a parameter dictionary that are not used for the
    actual estimator.
    """
    deletekeys = ['Number',
                  'UsePCA',
                  'PCAType',
                  'ReliefUse',
                  'ReliefNN',
                  'ReliefSampleSize',
                  'ReliefNumFeatures',
                  'OneHotEncoding',
                  'OneHotEncoding_feature_labels_tofit',
                  'Imputation',
                  'ImputationMethod',
                  'ImputationNeighbours',
                  'ImputationSkipAllNaN',
                  'SelectFromModel',
                  'SelectFromModel_lasso_alpha',
                  'SelectFromModel_estimator',
                  'SelectFromModel_n_trees',
                  'RFE',
                  'RFE_lasso_alpha',
                  'RFE_estimator',
                  'RFE_n_trees',
                  'RFE_n_features_to_select',
                  'RFE_step',
                  'Featsel_Variance',
                  'FeatPreProcess',
                  'FeatureScaling',
                  'StatisticalTestUse',
                  'StatisticalTestMetric',
                  'StatisticalTestThreshold',
                  'Resampling_Use',
                  'Resampling_Method',
                  'Resampling_sampling_strategy',
                  'Resampling_n_cores',
                  'Resampling_n_neighbors',
                  'Resampling_k_neighbors',
                  'Resampling_threshold_cleaning',
                  'random_seed'
                ]

    for k in deletekeys:
        if k in parameters.keys():
            del parameters[k]

    return parameters


def replacenan(image_features, verbose=True, feature_labels=None):
    '''
    Replace the NaNs in an image feature matrix.
    '''
    image_features_temp = image_features.copy()
    for pnum, x in enumerate(image_features_temp):
        for fnum, value in enumerate(x):
            if np.isnan(value):
                if verbose:
                    if feature_labels is not None:
                        print(f"[WARNING] NaN found, patient {pnum}, label {feature_labels[fnum]}. Replacing with zero.")
                    else:
                        print(f"[WARNING] NaN found, patient {pnum}, label {fnum}. Replacing with zero.")
                # Note: X is a list of lists, hence we cannot index the element directly
                image_features_temp[pnum, fnum] = 0

    return image_features_temp


def delete_cc_para(para):
    """Delete all parameters that are involved in classifier construction."""
    deletekeys = ['classifiers',
                  'max_iter',
                  'SVMKernel',
                  'SVMC',
                  'SVMdegree',
                  'SVMcoef0',
                  'SVMgamma',
                  'RFn_estimators',
                  'RFmin_samples_split',
                  'RFmax_depth',
                  'LRpenalty',
                  'LR_l1_ratio',
                  'LR_solver',
                  'LRC',
                  'LDA_solver',
                  'LDA_shrinkage',
                  'QDA_reg_param',
                  'ElasticNet_alpha',
                  'ElasticNet_l1_ratio',
                  'SGD_alpha',
                  'SGD_l1_ratio',
                  'SGD_loss',
                  'SGD_penalty',
                  'CNB_alpha',
                  'AdaBoost_learning_rate',
                  'AdaBoost_n_estimators',
                  'XGB_boosting_rounds',
                  'XGB_max_depth',
                  'XGB_learning_rate',
                  'XGB_gamma',
                  'XGB_min_child_weight',
                  'XGB_colsample_bytree',
                  'LightGBM_num_leaves',
                  'LightGBM_max_depth',
                  'LightGBM_min_child_samples',
                  'LightGBM_reg_alpha',
                  'LightGBM_reg_lambda',
                  'LightGBM_min_child_weight']

    for k in deletekeys:
        if k in para.keys():
            del para[k]

    return para
