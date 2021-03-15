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

import os
from scipy.stats import uniform
from WORC.classification import crossval as cv
from WORC.classification import construct_classifier as cc
from WORC.IOparser.file_io import load_features
import WORC.IOparser.config_io_classifier as config_io
from WORC.classification.AdvancedSampler import discrete_uniform, \
    log_uniform, boolean_uniform


def trainclassifier(feat_train, patientinfo_train, config,
                    output_hdf,
                    feat_test=None, patientinfo_test=None,
                    fixedsplits=None, verbose=True):
    """Train a classifier using machine learning from features.

    By default, if no
    split in training and test is supplied, a cross validation
    will be performed.

    Parameters
    ----------
    feat_train: string, mandatory
            contains the paths to all .hdf5 feature files used.
            modalityname1=file1,file2,file3,... modalityname2=file1,...
            Thus, modalities names are always between a space and a equal
            sign, files are split by commas. We assume that the lists of
            files for each modality has the same length. Files on the
            same position on each list should belong to the same patient.

    patientinfo: string, mandatory
            Contains the path referring to a .txt file containing the
            patient label(s) and value(s) to be used for learning. See
            the Github Wiki for the format.

    config: string, mandatory
            path referring to a .ini file containing the parameters
            used for feature extraction. See the Github Wiki for the possible
            fields and their description.

    output_hdf: string, mandatory
            path refering to a .hdf5 file to which the final classifier and
            it's properties will be written to.

    feat_test: string, optional
            When this argument is supplied, the machine learning will not be
            trained using a cross validation, but rather using a fixed training
            and text split. This field should contain paths of the test set
            feature files, similar to the feat_train argument.

    patientinfo_test: string, optional
            When feat_test is supplied, you can supply optionally a patient label
            file through which the performance will be evaluated.

    fixedsplits: string, optional
            By default, random split cross validation is used to train and
            evaluate the machine learning methods. Optionally, you can provide
            a .xlsx file containing fixed splits to be used. See the Github Wiki
            for the format.

    verbose: boolean, default True
            print final feature values and labels to command line or not.

    """
    # Convert inputs from lists to strings
    if type(patientinfo_train) is list:
        patientinfo_train = ''.join(patientinfo_train)

    if type(patientinfo_test) is list:
        patientinfo_test = ''.join(patientinfo_test)

    if type(config) is list:
        if len(config) == 1:
            config = ''.join(config)
        else:
            # FIXME
            print('[WORC Warning] You provided multiple configuration files: only the first one will be used!')
            config = config[0]

    if type(output_hdf) is list:
        if len(output_hdf) == 1:
            output_hdf = ''.join(output_hdf)
        else:
            # FIXME
            print('[WORC Warning] You provided multiple output hdf files: only the first one will be used!')
            output_hdf = output_hdf[0]

    if type(fixedsplits) is list:
        fixedsplits = ''.join(fixedsplits)

    # Load variables from the config file
    config = config_io.load_config(config)
    label_type = config['Labels']['label_names']
    modus = config['Labels']['modus']
    combine_features = config['FeatPreProcess']['Combine']
    combine_method = config['FeatPreProcess']['Combine_method']

    # Load the feature files and match to label data
    label_data_train, image_features_train =\
        load_features(feat_train, patientinfo_train, label_type,
                      combine_features, combine_method)

    if feat_test:
        label_data_test, image_features_test =\
            load_features(feat_test, patientinfo_test, label_type,
                          combine_features, combine_method)

    # Create tempdir name from patientinfo file name
    basename = os.path.basename(patientinfo_train)
    filename, _ = os.path.splitext(basename)
    path = patientinfo_train
    for i in range(4):
        # Use temp dir: result -> sample# -> parameters - > temppath
        path = os.path.dirname(path)

    _, path = os.path.split(path)
    path = os.path.join(path, 'trainclassifier', filename)

    # Construct the required classifier grid
    param_grid = cc.create_param_grid(config)

    # Add non-classifier parameters
    param_grid = add_parameters_to_grid(param_grid, config)

    # For N_iter, perform k-fold crossvalidation
    outputfolder = os.path.dirname(output_hdf)
    if feat_test is None:
        trained_classifier = cv.crossval(config, label_data_train,
                                         image_features_train,
                                         param_grid,
                                         modus=modus,
                                         use_fastr=config['Classification']['fastr'],
                                         fastr_plugin=config['Classification']['fastr_plugin'],
                                         fixedsplits=fixedsplits,
                                         ensemble=config['Ensemble'],
                                         outputfolder=outputfolder,
                                         tempsave=config['General']['tempsave'])
    else:
        trained_classifier = cv.nocrossval(config, label_data_train,
                                           label_data_test,
                                           image_features_train,
                                           image_features_test,
                                           param_grid,
                                           modus=modus,
                                           use_fastr=config['Classification']['fastr'],
                                           fastr_plugin=config['Classification']['fastr_plugin'],
                                           ensemble=config['Ensemble'])

    if not os.path.exists(os.path.dirname(output_hdf)):
        os.makedirs(os.path.dirname(output_hdf))

    trained_classifier.to_hdf(output_hdf, 'EstimatorData')

    print("Saved data!")


def add_parameters_to_grid(param_grid, config):
    """Add non-classifier parameters from config  to param grid."""
    # IF at least once groupwise search is turned on, add it to the param grid
    if 'True' in config['Featsel']['GroupwiseSearch']:
        param_grid['SelectGroups'] = config['Featsel']['GroupwiseSearch']
        for group in config['SelectFeatGroup'].keys():
            param_grid[group] = config['SelectFeatGroup'][group]

    # Add feature scaling parameters
    param_grid['FeatureScaling'] = config['FeatureScaling']['scaling_method']
    param_grid['FeatureScaling_skip_features'] =\
        [config['FeatureScaling']['skip_features']]

    # Add parameters for oversampling methods
    param_grid['Resampling_Use'] =\
        boolean_uniform(threshold=config['Resampling']['Use'])
    param_grid['Resampling_Method'] = config['Resampling']['Method']
    param_grid['Resampling_sampling_strategy'] =\
        config['Resampling']['sampling_strategy']
    param_grid['Resampling_n_neighbors'] =\
        discrete_uniform(loc=config['Resampling']['n_neighbors'][0],
                         scale=config['Resampling']['n_neighbors'][1])
    param_grid['Resampling_k_neighbors'] =\
        discrete_uniform(loc=config['Resampling']['k_neighbors'][0],
                         scale=config['Resampling']['k_neighbors'][1])
    param_grid['Resampling_threshold_cleaning'] =\
        uniform(loc=config['Resampling']['threshold_cleaning'][0],
                scale=config['Resampling']['threshold_cleaning'][1])

    param_grid['Resampling_n_cores'] = [config['General']['Joblib_ncores']]

    # Extract hyperparameter grid settings for SearchCV from config
    param_grid['FeatPreProcess'] = config['FeatPreProcess']['Use']
    param_grid['Featsel_Variance'] =\
        boolean_uniform(threshold=config['Featsel']['Variance'])

    param_grid['OneHotEncoding'] = config['OneHotEncoding']['Use']
    param_grid['OneHotEncoding_feature_labels_tofit'] =\
        [config['OneHotEncoding']['feature_labels_tofit']]

    param_grid['Imputation'] = config['Imputation']['use']
    param_grid['ImputationMethod'] = config['Imputation']['strategy']
    param_grid['ImputationNeighbours'] =\
        discrete_uniform(loc=config['Imputation']['n_neighbors'][0],
                         scale=config['Imputation']['n_neighbors'][1])

    param_grid['SelectFromModel'] =\
        boolean_uniform(threshold=config['Featsel']['SelectFromModel'])

    param_grid['SelectFromModel_lasso_alpha'] =\
        uniform(loc=config['Featsel']['SelectFromModel_lasso_alpha'][0],
                scale=config['Featsel']['SelectFromModel_lasso_alpha'][1])

    param_grid['SelectFromModel_estimator'] =\
        config['Featsel']['SelectFromModel_estimator']

    param_grid['SelectFromModel_n_trees'] =\
        discrete_uniform(loc=config['Featsel']['SelectFromModel_n_trees'][0],
                         scale=config['Featsel']['SelectFromModel_n_trees'][1])

    param_grid['UsePCA'] =\
        boolean_uniform(threshold=config['Featsel']['UsePCA'])
    param_grid['PCAType'] = config['Featsel']['PCAType']

    param_grid['StatisticalTestUse'] =\
        boolean_uniform(threshold=config['Featsel']['StatisticalTestUse'])

    param_grid['StatisticalTestMetric'] =\
        config['Featsel']['StatisticalTestMetric']
    param_grid['StatisticalTestThreshold'] =\
        log_uniform(loc=config['Featsel']['StatisticalTestThreshold'][0],
                    scale=config['Featsel']['StatisticalTestThreshold'][1])

    param_grid['ReliefUse'] =\
        boolean_uniform(threshold=config['Featsel']['ReliefUse'])

    param_grid['ReliefNN'] =\
        discrete_uniform(loc=config['Featsel']['ReliefNN'][0],
                         scale=config['Featsel']['ReliefNN'][1])

    param_grid['ReliefSampleSize'] =\
        uniform(loc=config['Featsel']['ReliefSampleSize'][0],
                scale=config['Featsel']['ReliefSampleSize'][1])

    param_grid['ReliefDistanceP'] =\
        discrete_uniform(loc=config['Featsel']['ReliefDistanceP'][0],
                         scale=config['Featsel']['ReliefDistanceP'][1])

    param_grid['ReliefNumFeatures'] =\
        discrete_uniform(loc=config['Featsel']['ReliefNumFeatures'][0],
                         scale=config['Featsel']['ReliefNumFeatures'][1])

    # Add a random seed, which is required for many methods
    param_grid['random_seed'] =\
        discrete_uniform(loc=0, scale=2**32 - 1)

    return param_grid
