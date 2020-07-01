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

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection._validation import _fit_and_score
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import scipy
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.utils import check_random_state
import random
from sklearn.metrics import make_scorer, average_precision_score
from WORC.classification.estimators import RankedSVM
from WORC.classification import construct_classifier as cc
from WORC.classification.metrics import check_scoring
from WORC.featureprocessing.Relief import SelectMulticlassRelief
from WORC.featureprocessing.Imputer import Imputer
from WORC.featureprocessing.VarianceThreshold import selfeat_variance
from WORC.featureprocessing.StatisticalTestThreshold import StatisticalTestThreshold
from WORC.featureprocessing.SelectGroups import SelectGroups

# Specific imports for error management
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from numpy.linalg import LinAlgError


def fit_and_score(X, y, scoring,
                  train, test, para,
                  fit_params=None,
                  return_train_score=True,
                  return_n_test_samples=True,
                  return_times=True, return_parameters=True,
                  error_score='raise', verbose=True,
                  return_all=True):
    '''
    Fit an estimator to a dataset and score the performance. The following
    methods can currently be applied as preprocessing before fitting, in
    this order:
    1. Select features based on feature type group (e.g. shape, histogram).
    2. Oversampling
    3. Apply feature imputation (WIP).
    4. Apply feature selection based on variance of feature among patients.
    5. Univariate statistical testing (e.g. t-test, Wilcoxon).
    6. Scale features with e.g. z-scoring.
    7. Use Relief feature selection.
    8. Select features based on a fit with a LASSO model.
    9. Select features using PCA.
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

    para: dictionary, mandatory
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
        test_sample_counts, fit_time, score_time, parameters_est
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

    imputer: WORC Imputater Object
        Either None if feature imputation is not used, or
        the fitted object.

    pca: WORC PCA Object
        Either None if PCA based feature selection is not used, or
        the fitted object.

    StatisticalSel: WORC StatisticalSel Object
        Either None if the statistical test feature selection is not used, or
        the fitted object.

    ReliefSel: WORC ReliefSel Object
        Either None if the RELIEF feature selection is not used, or
        the fitted object.

    Snote: WORC SMOTE Object
        Either None if the SMOTE oversampling is not used, or
        the fitted object.

    RandOverSample: WORC RandomOverSampler Object
        Either None if Random Oversampling is not used, or
        the fitted object.

    '''
    # Set some defaults for if a part fails and we return a dummy
    test_sample_counts = len(test)
    fit_time = np.inf
    score_time = np.inf
    train_score = np.nan
    test_score = np.nan
    Smote = None
    imputer = None
    scaler = None
    GroupSel = None
    SelectModel = None
    pca = None
    StatisticalSel = None
    VarSel = None
    ReliefSel = None
    RandOverSample = None

    if return_train_score:
        ret = [train_score, test_score, test_sample_counts,
               fit_time, score_time, para, para]
    else:
        ret = [test_score, test_sample_counts,
               fit_time, score_time, para, para]

    # We copy the parameter object so we can alter it and keep the original
    if verbose:
        print("\n")
        print('#######################################')
        print('Starting fit and score of new workflow.')
    para_estimator = para.copy()
    estimator = cc.construct_classifier(para_estimator)
    if scoring != 'average_precision_weighted':
        scorer = check_scoring(estimator, scoring=scoring)
    else:
        scorer = make_scorer(average_precision_score, average='weighted')

    para_estimator = delete_cc_para(para_estimator)

    # Get random seed from parameters
    random_seed = para_estimator['random_seed']
    random_state = check_random_state(random_seed)
    del para_estimator['random_seed']

    # X is a tuple: split in two arrays
    feature_values = np.asarray([x[0] for x in X])
    feature_labels = np.asarray([x[1] for x in X])

    # ------------------------------------------------------------------------
    # Feature scaling
    if 'FeatureScaling' in para_estimator.keys():
        if verbose:
            print("Fitting scaler and transforming features.")

        if para_estimator['FeatureScaling'] == 'z_score':
            scaler = StandardScaler().fit(feature_values)
        elif para_estimator['FeatureScaling'] == 'robust':
            scaler = RobustScaler().fit(feature_values)
        elif para_estimator['FeatureScaling'] == 'minmax':
            scaler = MinMaxScaler().fit(feature_values)

        if scaler is not None:
            feature_values = scaler.transform(feature_values)
        del para_estimator['FeatureScaling']

    # Delete the object if we do not need to return it
    if not return_all:
        del scaler

    # ------------------------------------------------------------------------
    # Feature imputation
    if 'Imputation' in para_estimator.keys():
        if para_estimator['Imputation'] == 'True':
            imp_type = para_estimator['ImputationMethod']
            if verbose:
                print(f'Imputing NaN with {imp_type}.')
            imp_nn = para_estimator['ImputationNeighbours']

            imputer = Imputer(missing_values=np.nan, strategy=imp_type,
                              n_neighbors=imp_nn)
            imputer.fit(feature_values)
            feature_values = imputer.transform(feature_values)

    if 'Imputation' in para_estimator.keys():
        del para_estimator['Imputation']
        del para_estimator['ImputationMethod']
        del para_estimator['ImputationNeighbours']

    # Delete the object if we do not need to return it
    if not return_all:
        del imputer

    # Remove any NaN feature values if these are still left after imputation
    feature_values = replacenan(feature_values, verbose=verbose, feature_labels=feature_labels[0])

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
                          'patient_features',
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

        GroupSel = SelectGroups(parameters=parameters_featsel,
                                toolboxes=toolboxes)
        GroupSel.fit(feature_labels[0])
        if verbose:
            print("Original Length: " + str(len(feature_values[0])))
        feature_values = GroupSel.transform(feature_values)
        if verbose:
            print("New Length: " + str(len(feature_values[0])))
        feature_labels = GroupSel.transform(feature_labels)

    # Delete the object if we do not need to return it
    if not return_all:
        del GroupSel

    # Check whether there are any features left
    if len(feature_values[0]) == 0:
        # TODO: Make a specific WORC exception for this warning.
        if verbose:
            print('[WARNING]: No features are selected! Probably all feature groups were set to False. Parameters:')
            print(para)

        # Delete the non-used fields
        para_estimator = delete_nonestimator_parameters(para_estimator)

        if return_all:
            return ret, GroupSel, VarSel, SelectModel, feature_labels[0], scaler, imputer, pca, StatisticalSel, ReliefSel, Smote, RandOverSample
        else:
            return ret

    # ------------------------------------------------------------------------
    # FIXME: When only using LBP feature, X is 3 dimensional with 3rd dimension length 1
    if len(feature_values.shape) == 3:
        feature_values = np.reshape(feature_values, (feature_values.shape[0], feature_values.shape[1]))
    if len(feature_labels.shape) == 3:
        feature_labels = np.reshape(feature_labels, (feature_labels.shape[0], feature_labels.shape[1]))

    # Remove any NaN feature values if these are still left after imputation
    feature_values = replacenan(feature_values, verbose=verbose, feature_labels=feature_labels[0])

    # --------------------------------------------------------------------
    # Feature selection based on variance
    if para_estimator['Featsel_Variance'] == 'True':
        if verbose:
            print("Selecting features based on variance.")
        if verbose:
            print("Original Length: " + str(len(feature_values[0])))
        try:
            feature_values, feature_labels, VarSel =\
                selfeat_variance(feature_values, feature_labels)
        except ValueError:
            if verbose:
                print('[WARNING]: No features meet the selected Variance threshold! Skipping selection.')
        if verbose:
            print("New Length: " + str(len(feature_values[0])))

    del para_estimator['Featsel_Variance']

    # Delete the object if we do not need to return it
    if not return_all:
        del VarSel

    # Check whether there are any features left
    if len(feature_values[0]) == 0:
        # TODO: Make a specific WORC exception for this warning.
        if verbose:
            print('[WARNING]: No features are selected! Probably you selected a feature group that is not in your feature file. Parameters:')
            print(para)
        para_estimator = delete_nonestimator_parameters(para_estimator)

        # Return a zero performance dummy
        ret = [train_score, test_score, test_sample_counts,
               fit_time, score_time, para_estimator, para]

        if return_all:
            return ret, GroupSel, VarSel, SelectModel, feature_labels[0], scaler, imputer, pca, StatisticalSel, ReliefSel, Smote, RandOverSample
        else:
            return ret

    # Check whether there are any features left
    if len(feature_values[0]) == 0:
        # TODO: Make a specific WORC exception for this warning.
        if verbose:
            print('[WORC WARNING]: No features are selected! Probably you selected a feature group that is not in your feature file. Parameters:')
            print(para)

        para_estimator = delete_nonestimator_parameters(para_estimator)

        # Return a zero performance dummy
        ret = [train_score, test_score, test_sample_counts,
               fit_time, score_time, para_estimator, para]

        if return_all:
            return ret, GroupSel, VarSel, SelectModel, feature_labels[0], scaler, imputer, pca, StatisticalSel, ReliefSel, Smote, RandOverSample
        else:
            return ret

    # --------------------------------------------------------------------
    # Relief feature selection, possibly multi classself.
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

            ReliefSel = SelectMulticlassRelief(n_neighbours=n_neighbours,
                                               sample_size=sample_size,
                                               distance_p=distance_p,
                                               numf=numf,
                                               random_state=random_seed)
            ReliefSel.fit(feature_values, y)
            if verbose:
                print("Original Length: " + str(len(feature_values[0])))
            feature_values = ReliefSel.transform(feature_values)
            if verbose:
                print("New Length: " + str(len(feature_values[0])))
            feature_labels = ReliefSel.transform(feature_labels)
            feature_labels.sort()
            for l in feature_labels[0]:
                print(l)

    # Delete the object if we do not need to return it
    if not return_all:
        del ReliefSel

    if 'ReliefUse' in para_estimator.keys():
        del para_estimator['ReliefUse']
        del para_estimator['ReliefNN']
        del para_estimator['ReliefSampleSize']
        del para_estimator['ReliefDistanceP']
        del para_estimator['ReliefNumFeatures']

    # ------------------------------------------------------------------------
    # Perform feature selection using a model
    if 'SelectFromModel' in para_estimator.keys() and para_estimator['SelectFromModel'] == 'True':
        if verbose:
            print("Selecting features using lasso model.")
        # Use lasso model for feature selection

        # First, draw a random value for alpha and the penalty ratio
        alpha = scipy.stats.uniform(loc=0.0, scale=1.5).rvs()
        # l1_ratio = scipy.stats.uniform(loc=0.5, scale=0.4).rvs()

        # Create and fit lasso model
        lassomodel = Lasso(alpha=alpha)
        lassomodel.fit(feature_values, y)

        # Use fit to select optimal features
        SelectModel = SelectFromModel(lassomodel, prefit=True)
        if verbose:
            print("Original Length: " + str(len(feature_values[0])))
        feature_values = SelectModel.transform(feature_values)
        if verbose:
            print("New Length: " + str(len(feature_values[0])))
        feature_labels = SelectModel.transform(feature_labels)

    if 'SelectFromModel' in para_estimator.keys():
        del para_estimator['SelectFromModel']

    # Delete the object if we do not need to return it
    if not return_all:
        del SelectModel

    # ----------------------------------------------------------------
    # PCA dimensionality reduction
    # Principle Component Analysis
    if 'UsePCA' in para_estimator.keys() and para_estimator['UsePCA'] == 'True':
        if verbose:
            print('Fitting PCA')
            print("Original Length: " + str(len(feature_values[0])))
        if para_estimator['PCAType'] == '95variance':
            # Select first X components that describe 95 percent of the explained variance
            pca = PCA(n_components=None, random_state=random_seed)
            try:
                pca.fit(feature_values)
            except (ValueError, LinAlgError) as e:
                print(f'[WARNING]: skipping this setting due to PCA Error: {e}.')
                ret = [train_score, test_score, test_sample_counts,
                       fit_time, score_time, para_estimator, para]

                if return_all:
                    return ret, GroupSel, VarSel, SelectModel, feature_labels[0], scaler, imputer, pca, StatisticalSel, ReliefSel, Smote, RandOverSample
                else:
                    return ret

            evariance = pca.explained_variance_ratio_
            num = 0
            sum = 0
            while sum < 0.95:
                sum += evariance[num]
                num += 1

            # Make a PCA based on the determined amound of components
            pca = PCA(n_components=num, random_state=random_seed)
            try:
                pca.fit(feature_values)
            except (ValueError, LinAlgError) as e:
                print(f'[WARNING]: skipping this setting due to PCA Error: {e}.')
                ret = [train_score, test_score, test_sample_counts,
                       fit_time, score_time, para_estimator, para]

                if return_all:
                    return ret, GroupSel, VarSel, SelectModel, feature_labels[0], scaler, imputer, pca, StatisticalSel, ReliefSel, Smote, RandOverSample
                else:
                    return ret

            feature_values = pca.transform(feature_values)

        else:
            # Assume a fixed number of components: cannot be larger than
            # n_samples
            n_components = min(len(feature_values), int(para_estimator['PCAType']))

            if n_components >= len(feature_values[0]):
                print(f"[WORC WARNING] PCA n_components ({n_components})> n_features ({len(feature_values[0])}): skipping PCA.")
            else:
                pca = PCA(n_components=n_components, random_state=random_seed)
                pca.fit(feature_values)
                feature_values = pca.transform(feature_values)

        if verbose:
            print("New Length: " + str(len(feature_values[0])))

    # Delete the object if we do not need to return it
    if not return_all:
        del pca

    if 'UsePCA' in para_estimator.keys():
        del para_estimator['UsePCA']
        del para_estimator['PCAType']

    # --------------------------------------------------------------------
    # Feature selection based on a statistical test
    if 'StatisticalTestUse' in para_estimator.keys():
        if para_estimator['StatisticalTestUse'] == 'True':
            metric = para_estimator['StatisticalTestMetric']
            threshold = para_estimator['StatisticalTestThreshold']
            if verbose:
                print(f"Selecting features based on statistical test. Method {metric}, threshold {round(threshold, 2)}.")
            if verbose:
                print("Original Length: " + str(len(feature_values[0])))

            StatisticalSel = StatisticalTestThreshold(metric=metric,
                                                      threshold=threshold)

            StatisticalSel.fit(feature_values, y)
            feature_values = StatisticalSel.transform(feature_values)
            feature_labels = StatisticalSel.transform(feature_labels)
            if verbose:
                print("New Length: " + str(len(feature_values[0])))
        del para_estimator['StatisticalTestUse']
        del para_estimator['StatisticalTestMetric']
        del para_estimator['StatisticalTestThreshold']

    # Delete the object if we do not need to return it
    if not return_all:
        del StatisticalSel

    # --------------------------------------------------------------------
    # Final check if there are still features left
    # Check whether there are any features left
    if len(feature_values[0]) == 0:
        # TODO: Make a specific WORC exception for this warning.
        if verbose:
            print('[WORC WARNING]: No features are selected! Probably you selected a feature group that is not in your feature file. Parameters:')
            print(para)

        para_estimator = delete_nonestimator_parameters(para_estimator)

        # Return a zero performance dummy
        ret = [train_score, test_score, test_sample_counts,
               fit_time, score_time, para_estimator, para]

        if return_all:
            return ret, GroupSel, VarSel, SelectModel, feature_labels[0], scaler, imputer, pca, StatisticalSel, ReliefSel, Smote, RandOverSample
        else:
            return ret

    # ------------------------------------------------------------------------
    # Use SMOTE oversampling
    if 'SampleProcessing_SMOTE' in para_estimator.keys():
        if para_estimator['SampleProcessing_SMOTE'] == 'True':

            # Determine our starting balance
            pos_initial = int(np.sum(y))
            neg_initial = int(len(y) - pos_initial)
            len_in = len(y)

            # Fit SMOTE object and transform dataset
            # NOTE: need to save random state for this one as well!
            Smote = SMOTE(random_state=random_state,
                          ratio=para_estimator['SampleProcessing_SMOTE_ratio'],
                          m_neighbors=para_estimator['SampleProcessing_SMOTE_neighbors'],
                          kind='borderline1',
                          n_jobs=para_estimator['SampleProcessing_SMOTE_n_cores'])

            feature_values, y = Smote.fit_sample(feature_values, y)

            # Also make sure our feature label object has the same size
            # NOTE: Not sure if this is the best implementation
            feature_labels = np.asarray([feature_labels[0] for x in X])

            # Note the user what SMOTE did
            pos = int(np.sum(y))
            neg = int(len(y) - pos)
            if verbose:
                message = f"Sampling with SMOTE from {len_in} ({pos_initial} pos," +\
                          f" {neg_initial} neg) to {len(y)} ({pos} pos, {neg} neg) patients."
                print(message)

        del para_estimator['SampleProcessing_SMOTE']
        del para_estimator['SampleProcessing_SMOTE_ratio']
        del para_estimator['SampleProcessing_SMOTE_neighbors']
        del para_estimator['SampleProcessing_SMOTE_n_cores']

    # Delete the object if we do not need to return it
    if not return_all:
        del Smote

    # ------------------------------------------------------------------------
    # Full Oversampling: To Do
    if 'SampleProcessing_Oversampling' in para_estimator.keys():
        if para_estimator['SampleProcessing_Oversampling'] == 'True':
            if verbose:
                print('Oversample underrepresented classes in training.')

            # Oversample underrepresented classes in training
            # We always use a factor 1, e.g. all classes end up with an
            # equal number of samples
            if len(y.shape) == 1:
                # Single Class, use imblearn oversampling
                RandOverSample = RandomOverSampler(random_state=random_state)
                feature_values, y = RandOverSample.fit_sample(feature_values, y)

            else:
                # Multi class, use own method as imblearn cannot do this
                sumclass = [np.sum(y[:, i]) for i in range(y.shape[1])]
                maxclass = np.argmax(sumclass)
                for i in range(y.shape[1]):
                    if i != maxclass:
                        # Oversample
                        nz = np.nonzero(y[:, i])[0]
                        noversample = sumclass[maxclass] - sumclass[i]
                        while noversample > 0:
                            n_sample = random.randint(0, len(nz) - 1)
                            n_sample = nz[n_sample]
                            i_sample = y[n_sample, :]
                            x_sample = feature_values[n_sample]
                            y = np.vstack((y, i_sample))
                            feature_values.append(x_sample)
                            noversample -= 1

        del para_estimator['SampleProcessing_Oversampling']

    # Delete the object if we do not need to return it
    if not return_all:
        del RandOverSample

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

    if labellength > 1 and type(estimator) not in [RankedSVM,
                                                   RandomForestClassifier]:
        # Multiclass, hence employ a multiclass classifier for e.g. SVM, LR
        estimator.set_params(**para_estimator)
        estimator = OneVsRestClassifier(estimator)
        para_estimator = {}

    if verbose:
        print("Fitting ML.")

    try:
        ret = _fit_and_score(estimator, feature_values, y,
                             scorer, train,
                             test, verbose,
                             para_estimator, fit_params, return_train_score,
                             return_parameters,
                             return_n_test_samples,
                             return_times, error_score)
    except (ValueError, LinAlgError) as e:
        if type(estimator) == LDA:
            print(f'[WARNING]: skipping this setting due to LDA Error: {e}.')
            ret = [train_score, test_score, test_sample_counts,
                   fit_time, score_time, para_estimator, para]

            if return_all:
                return ret, GroupSel, VarSel, SelectModel, feature_labels[0], scaler, imputer, pca, StatisticalSel, ReliefSel, Smote, RandOverSample
            else:
                return ret
        else:
            raise e

    # Remove 'estimator object', it's the causes of a bug.
    # Somewhere between scikit-learn 0.18.2 and 0.20.2
    # the estimator object return value was added
    # removing this element fixes a bug that occurs later
    # in SearchCV.py, where an array without estimator
    # object is expected.
    del ret[-1]

    # Paste original parameters in performance
    ret.append(para)

    if return_all:
        return ret, GroupSel, VarSel, SelectModel, feature_labels[0], scaler, imputer, pca, StatisticalSel, ReliefSel, Smote, RandOverSample
    else:
        return ret


def delete_nonestimator_parameters(parameters):
    '''
    Delete all parameters in a parameter dictionary that are not used for the
    actual estimator.
    '''
    if 'Number' in parameters.keys():
        del parameters['Number']

    if 'UsePCA' in parameters.keys():
        del parameters['UsePCA']
        del parameters['PCAType']

    if 'Imputation' in parameters.keys():
        del parameters['Imputation']
        del parameters['ImputationMethod']
        del parameters['ImputationNeighbours']

    if 'SelectFromModel' in parameters.keys():
        del parameters['SelectFromModel']

    if 'Featsel_Variance' in parameters.keys():
        del parameters['Featsel_Variance']

    if 'FeatPreProcess' in parameters.keys():
        del parameters['FeatPreProcess']

    if 'FeatureScaling' in parameters.keys():
        del parameters['FeatureScaling']

    if 'StatisticalTestUse' in parameters.keys():
        del parameters['StatisticalTestUse']
        del parameters['StatisticalTestMetric']
        del parameters['StatisticalTestThreshold']

    if 'SampleProcessing_SMOTE' in parameters.keys():
        del parameters['SampleProcessing_SMOTE']
        del parameters['SampleProcessing_SMOTE_ratio']
        del parameters['SampleProcessing_SMOTE_neighbors']
        del parameters['SampleProcessing_SMOTE_n_cores']

    if 'SampleProcessing_Oversampling' in parameters.keys():
        del parameters['SampleProcessing_Oversampling']

    if 'random_seed' in parameters.keys():
        del parameters['random_seed']

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
                        print(f"[WORC WARNING] NaN found, patient {pnum}, label {feature_labels[fnum]}. Replacing with zero.")
                    else:
                        print(f"[WORC WARNING] NaN found, patient {pnum}, label {fnum}. Replacing with zero.")
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
                  'CNB_alpha']

    for k in deletekeys:
        if k in para.keys():
            del para[k]

    return para
