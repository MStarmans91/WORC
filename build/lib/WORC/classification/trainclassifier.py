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

import json
import os
import sklearn

from WORC.classification import crossval as cv
from WORC.classification import construct_classifier as cc
from WORC.plotting.plot_SVM import plot_SVM
from WORC.plotting.plot_SVR import plot_single_SVR
import WORC.IOparser.file_io as file_io
import WORC.IOparser.config_io_classifier as config_io
from scipy.stats import uniform
from WORC.classification.AdvancedSampler import discrete_uniform


def trainclassifier(feat_train, patientinfo_train, config,
                    output_hdf, output_json,
                    feat_test=None, patientinfo_test=None,
                    fixedsplits=None, verbose=True):
    '''
    Train a classifier using machine learning from features. By default, if no
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

    output_json: string, mandatory
            path refering to a .json file to which the performance of the final
            classifier will be written to. This file is generated through one of
            the WORC plotting functions.

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

    '''

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

    if type(output_json) is list:
        if len(output_json) == 1:
            output_json = ''.join(output_json)
        else:
            # FIXME
            print('[WORC Warning] You provided multiple output json files: only the first one will be used!')
            output_json = output_json[0]

    # Load variables from the config file
    config = config_io.load_config(config)
    label_type = config['Labels']['label_names']
    modus = config['Labels']['modus']

    # Load the feature files and match to label data
    label_data_train, image_features_train =\
        load_features(feat_train, patientinfo_train, label_type)

    if feat_test:
        label_data_test, image_features_test =\
            load_features(feat_test, patientinfo_test, label_type)

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

    # IF at least once groupwise search is turned on, add it to the param grid
    if 'True'in config['Featsel']['GroupwiseSearch']:
        param_grid['SelectGroups'] = config['Featsel']['GroupwiseSearch']
        for group in config['SelectFeatGroup'].keys():
            param_grid[group] = config['SelectFeatGroup'][group]

    # If scaling is to be applied, add to parameters
    if config['FeatureScaling']['scale_features']:
        if type(config['FeatureScaling']['scaling_method']) is not list:
            param_grid['FeatureScaling'] = [config['FeatureScaling']['scaling_method']]
        else:
            param_grid['FeatureScaling'] = config['FeatureScaling']['scaling_method']

    # Add parameters for oversampling methods
    param_grid['SampleProcessing_SMOTE'] = config['SampleProcessing']['SMOTE']
    param_grid['SampleProcessing_SMOTE_ratio'] =\
        uniform(loc=config['SampleProcessing']['SMOTE_ratio'][0],
                scale=config['SampleProcessing']['SMOTE_ratio'][1])
    param_grid['SampleProcessing_SMOTE_neighbors'] =\
        discrete_uniform(loc=config['SampleProcessing']['SMOTE_neighbors'][0],
                         scale=config['SampleProcessing']['SMOTE_neighbors'][1])
    param_grid['SampleProcessing_SMOTE_n_cores'] = [config['General']['Joblib_ncores']]
    param_grid['SampleProcessing_Oversampling'] = config['SampleProcessing']['Oversampling']

    # Extract hyperparameter grid settings for SearchCV from config
    param_grid['Featsel_Variance'] = config['Featsel']['Variance']

    param_grid['Imputation'] = config['Imputation']['use']
    param_grid['ImputationMethod'] = config['Imputation']['strategy']
    param_grid['ImputationNeighbours'] =\
        discrete_uniform(loc=config['Imputation']['n_neighbors'][0],
                         scale=config['Imputation']['n_neighbors'][1])

    param_grid['SelectFromModel'] = config['Featsel']['SelectFromModel']

    param_grid['UsePCA'] = config['Featsel']['UsePCA']
    param_grid['PCAType'] = config['Featsel']['PCAType']

    param_grid['StatisticalTestUse'] =\
        config['Featsel']['StatisticalTestUse']
    param_grid['StatisticalTestMetric'] =\
        config['Featsel']['StatisticalTestMetric']
    param_grid['StatisticalTestThreshold'] =\
        uniform(loc=config['Featsel']['StatisticalTestThreshold'][0],
                scale=config['Featsel']['StatisticalTestThreshold'][1])

    param_grid['ReliefUse'] =\
        config['Featsel']['ReliefUse']

    param_grid['ReliefNN'] =\
        discrete_uniform(loc=config['Featsel']['ReliefNN'][0],
                         scale=config['Featsel']['ReliefNN'][1])

    param_grid['ReliefSampleSize'] =\
        discrete_uniform(loc=config['Featsel']['ReliefSampleSize'][0],
                         scale=config['Featsel']['ReliefSampleSize'][1])

    param_grid['ReliefDistanceP'] =\
        discrete_uniform(loc=config['Featsel']['ReliefDistanceP'][0],
                         scale=config['Featsel']['ReliefDistanceP'][1])

    param_grid['ReliefNumFeatures'] =\
        discrete_uniform(loc=config['Featsel']['ReliefNumFeatures'][0],
                         scale=config['Featsel']['ReliefNumFeatures'][1])

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

    trained_classifier.to_hdf(output_hdf, 'SVMdata')

    # Check whether we do regression or classification
    regressors = ['SVR', 'RFR', 'SGDR', 'Lasso', 'ElasticNet']
    isclassifier =\
        not any(clf in regressors for clf in config['Classification']['classifiers'])

    # Calculate statistics of performance
    if feat_test is None:
        if not isclassifier:
            statistics = plot_single_SVR(trained_classifier, label_data_train,
                                         label_type)
        else:
            statistics = plot_SVM(trained_classifier, label_data_train,
                                  label_type, modus=modus)
    else:
        if patientinfo_test is not None:
            if not isclassifier:
                statistics = plot_single_SVR(trained_classifier,
                                             label_data_test,
                                             label_type)
            else:
                statistics = plot_SVM(trained_classifier,
                                      label_data_test,
                                      label_type,
                                      modus=modus)
        else:
            statistics = None

    # Save output
    savedict = dict()
    savedict["Statistics"] = statistics

    if not os.path.exists(os.path.dirname(output_json)):
        os.makedirs(os.path.dirname(output_json))

    with open(output_json, 'w') as fp:
        json.dump(savedict, fp, indent=4)

    print("Saved data!")


def load_features(feat, patientinfo, label_type):
    ''' Read feature files and stack the features per patient in an array.
        Additionally, if a patient label file is supplied, the features from
        a patient will be matched to the labels.

        Parameters
        ----------
        featurefiles: list, mandatory
                List containing all paths to the .hdf5 feature files to be loaded.
                The argument should contain a list per modelity, e.g.
                [[features_mod1_patient1, features_mod1_patient2, ...],
                 [features_mod2_patient1, features_mod2_patient2, ...]].

        patientinfo: string, optional
                Path referring to the .txt file to be used to read patient
                labels from. See the Github Wiki for the format.

        label_names: list, optional
                List containing all the labels that should be extracted from
                the patientinfo file.

    '''
    # Split the feature files per modality
    feat_temp = list()
    modnames = list()
    for feat_mod in feat:
        feat_mod_temp = [str(item).strip() for item in feat_mod.split(',')]

        # The first item contains the name of the modality, followed by a = sign
        temp = [str(item).strip() for item in feat_mod_temp[0].split('=')]
        modnames.append(temp[0])
        feat_mod_temp[0] = temp[1]

        # Append the files to the main list
        feat_temp.append(feat_mod_temp)

    feat = feat_temp

    # Read the features and classification data
    label_data, image_features =\
        file_io.load_data(feat, patientinfo,
                          label_type, modnames)

    return label_data, image_features
