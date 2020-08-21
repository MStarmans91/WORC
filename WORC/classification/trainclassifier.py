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

import json
import os
import numpy as np
from scipy.stats import uniform
from WORC.classification import crossval as cv
from WORC.classification import construct_classifier as cc
from WORC.plotting.plot_estimator_performance import plot_estimator_performance
from WORC.IOparser.file_io import load_features
import WORC.IOparser.config_io_classifier as config_io
from WORC.classification.AdvancedSampler import discrete_uniform, \
    log_uniform, boolean_uniform


def trainclassifier(feat_train, patientinfo_train, config,
                    output_hdf, output_json,
                    feat_test=None, patientinfo_test=None,
                    fixedsplits=None, output_smac=None, verbose=True):
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

    if type(output_json) is list:
        if len(output_json) == 1:
            output_json = ''.join(output_json)
        else:
            # FIXME
            print('[WORC Warning] You provided multiple output json files: only the first one will be used!')
            output_json = output_json[0]

    if type(fixedsplits) is list:
        fixedsplits = ''.join(fixedsplits)

    if type(output_smac) is list:
        if len(output_smac) == 1:
            output_smac = ''.join(output_smac)
        else:
            # FIXME
            print('[WORC Warning] You provided multiple output json files: only the first one will be used!')
            output_smac = output_smac[0]

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

    param_grid['Imputation'] = config['Imputation']['use']
    param_grid['ImputationMethod'] = config['Imputation']['strategy']
    param_grid['ImputationNeighbours'] =\
        discrete_uniform(loc=config['Imputation']['n_neighbors'][0],
                         scale=config['Imputation']['n_neighbors'][1])

    param_grid['SelectFromModel'] =\
        boolean_uniform(threshold=config['Featsel']['SelectFromModel'])

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

    # For N_iter, perform k-fold crossvalidation
    outputfolder = os.path.dirname(output_hdf)
    smac_result_file = output_smac
    if feat_test is None:
        trained_classifier = cv.crossval(config, label_data_train,
                                         image_features_train,
                                         param_grid,
                                         modus=modus,
                                         use_fastr=config['Classification']['fastr'],
                                         use_SMAC=config['SMAC']['use'],
                                         fastr_plugin=config['Classification']['fastr_plugin'],
                                         fixedsplits=fixedsplits,
                                         ensemble=config['Ensemble'],
                                         outputfolder=outputfolder,
                                         smac_result_file=smac_result_file,
                                         tempsave=config['General']['tempsave'])
    else:
        trained_classifier = cv.nocrossval(config, label_data_train,
                                           label_data_test,
                                           image_features_train,
                                           image_features_test,
                                           param_grid,
                                           modus=modus,
                                           use_fastr=config['Classification']['fastr'],
                                           use_SMAC=config['SMAC']['use'],
                                           fastr_plugin=config['Classification']['fastr_plugin'],
                                           ensemble=config['Ensemble'])

    if not os.path.exists(os.path.dirname(output_hdf)):
        os.makedirs(os.path.dirname(output_hdf))

    trained_classifier.to_hdf(output_hdf, 'EstimatorData')

    # Check whether we do regression or classification
    regressors = ['SVR', 'RFR', 'SGDR', 'Lasso', 'ElasticNet']
    isclassifier =\
        not any(clf in regressors for clf in config['Classification']['classifiers'])


    if config['SMAC']['use']:

        with open(smac_result_file, 'r') as jsonfile:
            smac_result_dict = json.load(jsonfile)

        # Gather the statistics of all cross-validation summaries
        all_best_scores = []
        all_average_scores = []
        all_inc_wallclock_times = []
        all_inc_evaluations = []
        all_inc_changed = []
        for cv_iteration in smac_result_dict:
            all_best_scores.append(smac_result_dict[cv_iteration]['cv-summary']['best_score'])
            all_average_scores.append(smac_result_dict[cv_iteration]['cv-summary']['average_score'])
            all_inc_wallclock_times.append(smac_result_dict[cv_iteration]['cv-summary']['best_inc_wallclock_time'])
            all_inc_evaluations.append(smac_result_dict[cv_iteration]['cv-summary']['best_inc_evals'])
            all_inc_changed.append(smac_result_dict[cv_iteration]['cv-summary']['best_inc_changed'])
        overall_results = {'overall results': {'avg_best_score': np.mean(all_best_scores),
                                               'std_best_score': np.std(all_best_scores),
                                               'avg_average_score': np.mean(all_average_scores),
                                               'std_average_score': np.std(all_average_scores),
                                               'avg_inc_wallclock_time': np.mean(all_inc_wallclock_times),
                                               'std_inc_wallclock_time': np.std(all_inc_wallclock_times),
                                               'avg_inc_evaluations': np.mean(all_inc_evaluations),
                                               'std_inc_evaluations': np.std(all_inc_evaluations),
                                               'avg_inc_changed': np.mean(all_inc_changed),
                                               'std_inc_changed': np.std(all_inc_changed)}
                           }
        smac_result_dict.update(overall_results)
        with open(smac_result_file, 'w') as jsonfile:
            json.dump(smac_result_dict, jsonfile, indent=4)

    '''
    ## ----------------------------------------- ##
    # Process the statistics of the SMAC optimization
    # ! Perhaps move this to a better location in the future
    if config['SMAC']['use']:

        with open(smac_result_file, 'r') as jsonfile:
            smac_result_dict = json.load(jsonfile)

        # Create a dictionary with the averages
        totals = dict()
        metric_names = ['ta_runs', 'std_ta_runs', 'n_configs', 'std_n_configs',
                        'wallclock_time_used', 'std_wallclock_time_used',
                        'ta_time_used', 'std_ta_time_used', 'inc_changed',
                        'std_inc_changed', 'wallclock_time_best',
                        'std_wallclock_time_best', 'evaluation_best',
                        'std_evaluation_best', 'cost_best', 'std_cost_best']
        for metric_name in metric_names[0::2]:
            totals[metric_name] = []

        best_instances = []
        for cv_iteration in smac_result_dict:
            all_val_scores = []
            for instance in cv_iteration:
                nr_of_incumbent_updates = instance['inc_changed']
                all_val_scores.append(instance['inc_costs'][nr_of_incumbent_updates - 1])
            best_score_index = all_val_scores.index(np.max(all_val_scores))
            best_instances.append(str(best_score_index))

        instance_index_count = 0
        for cv_iteration in smac_result_dict:
            # Only run this code for the best instance in this cv
            instance = cv_iteration[best_instances[instance_index_count]]

            nr_of_incumbent_updates = instance['inc_changed']
            # Extract the details of the last (best) incumbent
            totals['wallclock_time_best'].append(
                instance['inc_wallclock_times'][nr_of_incumbent_updates - 1])
            totals['evaluation_best'].append(
                instance['inc_evaluations'][nr_of_incumbent_updates - 1])
            totals['cost_best'].append(
                instance['inc_costs'][nr_of_incumbent_updates - 1])
            for metric in instance:
                if metric in metric_names:
                    totals[metric].append(instance[metric])
            instance_index_count += 1

        averages = dict()
        list_position_count = 1
        for metric_name in totals:
            averages[metric_name] = np.mean(totals[metric_name])
            averages[metric_names[list_position_count]] = np.std(totals[metric_name])
            list_position_count += 2

        smac_result_dict['averages'] = averages

        with open(smac_result_file, 'w') as jsonfile:
            json.dump(smac_result_dict, jsonfile, indent=4)
    '''
    ## --------------------------------------------- ##


    # Calculate statistics of performance
    overfit_scaler = config['Evaluation']['OverfitScaler']
    if feat_test is None:
        if not isclassifier:
            statistics =\
                plot_estimator_performance(trained_classifier,
                                           label_data_train,
                                           label_type,
                                           ensemble=config['Ensemble']['Use'],
                                           bootstrap=config['Bootstrap']['Use'],
                                           bootstrap_N=config['Bootstrap']['N_iterations'],
                                           overfit_scaler=overfit_scaler)
        else:
            statistics =\
                plot_estimator_performance(trained_classifier,
                                           label_data_train,
                                           label_type, modus=modus,
                                           ensemble=config['Ensemble']['Use'],
                                           bootstrap=config['Bootstrap']['Use'],
                                           bootstrap_N=config['Bootstrap']['N_iterations'],
                                           overfit_scaler=overfit_scaler)
    else:
        if patientinfo_test is not None:
            if not isclassifier:
                statistics =\
                    plot_estimator_performance(trained_classifier,
                                               label_data_test,
                                               label_type,
                                               ensemble=config['Ensemble']['Use'],
                                               bootstrap=config['Bootstrap']['Use'],
                                               bootstrap_N=config['Bootstrap']['N_iterations'],
                                               overfit_scaler=overfit_scaler)
            else:
                statistics =\
                    plot_estimator_performance(trained_classifier,
                                               label_data_test,
                                               label_type,
                                               modus=modus,
                                               ensemble=config['Ensemble']['Use'],
                                               bootstrap=config['Bootstrap']['Use'],
                                               bootstrap_N=config['Bootstrap']['N_iterations'],
                                               overfit_scaler=overfit_scaler)
        else:
            statistics = None

    # Save output

    if not os.path.exists(os.path.dirname(output_json)):
        os.makedirs(os.path.dirname(output_json))

    with open(output_json, 'w') as fp:
        json.dump(statistics, fp, sort_keys=True, indent=4)

    print("Saved data!")
