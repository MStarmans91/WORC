#!/usr/bin/env python

import WORC
import os

# These packages are only used in analysing the results
import pandas as pd
import json
import fastr
import time
import sys
import glob

# If you don't want to use your own data, we use the following example set,
# see also the next code block in this example.
from WORC.exampledata.datadownloader import download_HeadAndNeck

local = False
dataset = 'DM'  # [DM, BLT]
# if sys.argv[7] == '0':
#    no_sex_and_age = False
# else:
#    no_sex_and_age = True
no_sex_and_age = True

# Define the folder this script is in, so we can easily find the example data
if local:
    script_path = os.path.dirname(os.path.abspath(__file__))
else:
    script_path = os.path.abspath('/home/mdeen/code/WORC')

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
# The minimal inputs to WORC are:
#   - Images
#   - Segmentations
#   - Labels
#
# In SimpleWORC, we assume you have a folder "datadir", in which there is a
# folder for each patient, where in each folder there is a image.nii.gz and a mask.nii.gz:
#           Datadir
#               Patient_001
#                   image.nii.gz
#                   mask.nii.gz
#               Patient_002
#                   image.nii.gz
#                   mask.nii.gz
#               ...
#
#
# You can skip this part if you use your own data.
# In the example, We will use open source data from the online XNAT platform
# at https://xnat.bmia.nl/data/archive/projects/stwstrategyhn1. This dataset
# consists of CT scans of patients with Head and Neck tumors. We will download
# a subset of 20 patients in this folder. You can change this settings if you
# like

nsubjects = 20  # use "all" to download all patients

if local:
    data_path = os.path.join(script_path, 'Experiments', 'Data')
else:
    data_path = os.path.abspath('archive/mdeen')
# download_HeadAndNeck(datafolder=data_path, nsubjects=nsubjects)

# Identify our data structure: change the fields below accordingly
# if you use your own data.
# imagedatadir = os.path.join(data_path, 'stwstrategyhn1')
# image_file_name = 'image.nii.gz'
# segmentation_file_name = 'mask.nii.gz'

if local:
    featuredatadir = os.path.join(data_path, 'Features')
    feature_file_name = 'features_0.hdf5'
elif dataset == 'BLT':
    # Make the feature dictionary
    if no_sex_and_age:
        pathnames = glob.glob(
            '/archive/mdeen/features/predict_features_erasmus_no_sex_and_age/features_predict_MR_0_BLTRadiomics-*_0.hdf5')
    else:
        pathnames = glob.glob(
            '/archive/mdeen/features/predict_features_erasmus/features_predict_MR_0_BLTRadiomics-*_0.hdf5')
    features = dict()
    for i in range(len(pathnames)):

        feature_path = pathnames[i]
        if no_sex_and_age:
            patient_nr = feature_path[99: 102]
        else:
            patient_nr = feature_path[84:87]

        feature_name = 'BLTRadiomics-' + patient_nr

        '''
        # Read in the hdf5 file and remove 4 features
        pandas_data = pd.read_hdf(feature_path)
        index = 0
        features_to_remove = ['PREDICT_original_pf_age', 'PREDICT_original_pf_sex', 'PREDICT_original_semf_Age', 'PREDICT_original_semf_Gender']
        for wrong_feature in features_to_remove:
            index = pandas_data.feature_labels.index(wrong_feature)
            del pandas_data.feature_values[index]
            del pandas_data.feature_labels[index]

        # Write the new features back to file
        new_feature_path = '/archive/mdeen/features/predict_features_erasmus_no_sex_and_age/features_predict_MR_0_BLTRadiomics-' \
                           + patient_nr + '_0.hdf5'
        pandas_data.to_hdf(new_feature_path, 'Image features')
        '''

        features[feature_name] = feature_path

        i += 1

    feature_input = []
    feature_input.append(features)

elif dataset == 'DM':
    '''
    # Make the feature dictionary
    pathnames = glob.glob(
        '/archive/mdeen/features/DM/*.hdf5')
    features = dict()

    for i in range(len(pathnames)):

        feature_path = pathnames[i]

        features[feature_path] = feature_path

        i += 1

    feature_input = []
    feature_input.append(features)
    print(feature_input)
    '''

    features_predict = dict()
    pathnames_predict_DESM = glob.glob('/archive/mdeen/features/DM/features_predict_MR_0_DESM*.hdf5')
    for i in range(len(pathnames_predict_DESM)):
        if i <= 2:
            nr = i + 100
        else:
            nr = i + 102
        key = 'DESM-' + str(nr)
        features_predict[key] = pathnames_predict_DESM[i]

    pathnames_predict = glob.glob('/archive/mdeen/features/DM/features_predict_MR_0_DMR*.hdf5')
    for i in range(len(pathnames_predict)):
        nr = pathnames_predict[i][61:64]
        key = 'DMRadiomics-' + str(nr)
        features_predict[key] = pathnames_predict[i]

    features_pyradiomics = dict()
    pathnames_pyradiomics_DESM = glob.glob('/archive/mdeen/features/DM/features_pyradiomics_MR_0_DESM*.hdf5')
    for i in range(len(pathnames_pyradiomics_DESM)):
        if i <= 2:
            nr = i + 100
        else:
            nr = i + 102
        key = 'DESM-' + str(nr)
        features_pyradiomics[key] = pathnames_pyradiomics_DESM[i]

    pathnames_pyradiomics = glob.glob('/archive/mdeen/features/DM/features_pyradiomics_MR_0_DMR*.hdf5')
    for i in range(len(pathnames_pyradiomics)):
        nr = pathnames_pyradiomics[i][65:68]
        key = 'DMRadiomics-' + str(nr)
        features_pyradiomics[key] = pathnames_pyradiomics[i]

    feature_input = []
    feature_input.append(features_predict)
    feature_input.append(features_pyradiomics)

# File in which the labels (i.e. outcome you want to predict) is stated
# Again, change this accordingly if you use your own data.
if local:
    label_file = os.path.join(data_path, 'Examplefiles', 'pinfo_HN.csv')
elif dataset == 'BLT':
    label_file = os.path.abspath('/archive/mdeen/labels/pinfo_BLT_all_ComBat.csv')
elif dataset == 'DM':
    label_file = os.path.abspath('/archive/mdeen/labels/pinfo_DM_v2.csv')

# Name of the label you want to predict
if local:
    label_name = 'imaginary_label_1'
elif dataset == 'BLT':
    label_name = 'Malignant'
elif dataset == 'DM':
    label_name = 'DM'

# Determine whether we want to do a coarse quick experiment, or a full lengthy
# one. Again, change this accordingly if you use your own data.
coarse = False

# Give your experiment a name
if local:
    experiment_name_base = 'Tutorial_dataset_'
elif dataset == 'BLT':
    experiment_name_base = 'BLT_dataset_'
elif dataset == 'DM':
    experiment_name_base = 'DM_dataset_'
experiment_name_input = input('Experiment name: ')
experiment_name = experiment_name_base + experiment_name_input

# Instead of the default tempdir, let's but the temporary output in a subfolder
# in the same folder as this script

if local:
    tmpdir = os.path.join(script_path, 'Experiments', 'Tempfiles', 'WORC_', experiment_name)
else:
    tmpdir = os.path.abspath('/scratch/mdeen/tempfiles/WORC_' + experiment_name)

# ---------------------------------------------------------------------------
# The actual experiment
# ---------------------------------------------------------------------------

# Create a Simple WORC object
experiment = WORC.BasicWORC(experiment_name)

# Set the input data according to the variables we defined earlier
# experiment.images_from_this_directory(imagedatadir,
#                                      image_file_name=image_file_name)
# experiment.segmentations_from_this_directory(imagedatadir,
#                                             segmentation_file_name=segmentation_file_name)
if local:
    experiment.features_from_this_directory(featuredatadir, feature_file_name=feature_file_name)
else:
    experiment.features_train = feature_input

experiment.labels_from_this_file(label_file)
experiment.predict_labels([label_name])

# Use the standard workflow for binary classification
experiment.binary_classification(coarse=coarse, scoring_method='f1_weighted')

# Set the temporary directory
experiment.set_tmpdir(tmpdir)

# Set the settings relevant for an experiment
input_parameters = sys.argv
if input_parameters[1] == '1':
    use_smac = 'True'
else:
    use_smac = 'False'
cv_iter = input_parameters[2]
cv_splits = input_parameters[3]
opt_iter = input_parameters[4]
jobs_per_core = input_parameters[5]
n_ensembles = input_parameters[6]

experiment.add_config_overrides({'CrossValidation': {'N_iterations': cv_iter},
                                 'Classification': {'LRpenalty': 'l2'},
                                 'HyperOptimization': {'n_splits': cv_splits},
                                 'Ensemble': {'Use': n_ensembles}})

if dataset == 'BLT':
    experiment.add_config_overrides({'SelectFeatGroup': {'toolbox': 'PREDICT'}})

if use_smac == 'True':
    budget_type = input_parameters[7]
    init_method = input_parameters[8]
    init_budget = input_parameters[9]
    experiment.add_config_overrides({'SMAC': {'use': 'True',
                                              'n_smac_cores': jobs_per_core,
                                              'budget_type': budget_type,
                                              'budget': opt_iter,
                                              'init_method': init_method,
                                              'init_budget': init_budget}})
else:
    experiment.add_config_overrides({'HyperOptimization': {'N_iterations': opt_iter,
                                                           'n_jobspercore': jobs_per_core}})

# Turn off the groupwise selection
# experiment.add_config_overrides({'Featsel': {'GroupwiseSearch': 'False'}})

# Turn off statistical test feature selection
# experiment.add_config_overrides({'Featsel': {'StatisticalTestUse': '0.0'}})

# Until the implementation is fully tested, disable resampling
# experiment.add_config_overrides({'Resampling': {'Use': '0.0'}})

# Use a fixed seed for the train/test split
experiment.add_config_overrides({'CrossValidation': {'fixed_seed': 'True'}})

# Set the ensemble method to Caruana
#experiment.add_config_overrides({'Ensemble': {'Method': 'top_N'}})

# Set up wall clock time
start_time = time.time()

# Run the experiment!
experiment.execute()

# NOTE:  Precomputed features can be used instead of images and masks
# by instead using ``experiment.features_from_this_directory(featuresdatadir)`` in a similar fashion.


# ---------------------------------------------------------------------------
# Analysis of results
# ---------------------------------------------------------------------------

# There are two main outputs: the features for each patient/object, and the overall
# performance. These are stored as .hdf5 and .json files, respectively. By
# default, they are saved in the so-called "fastr output mount", in a subfolder
# named after your experiment name.

# Locate output folder
outputfolder = fastr.config.mounts['output']
experiment_folder = os.path.join(outputfolder, 'WORC_' + experiment_name)

print(f"Your output is stored in {experiment_folder}.")

# Read the features for the first patient
# NOTE: we use the glob package for scanning a folder to find specific files
# feature_files = glob.glob(os.path.join(experiment_folder,
#                                       'Features',
#                                       'features_*.hdf5'))

# if len(feature_files) == 0:
#    raise ValueError('No feature files found: your network has failed.')

# featurefile_p1 = feature_files[0]
# features_p1 = pd.read_hdf(featurefile_p1)

# Read the overall peformance
performance_file = os.path.join(experiment_folder, 'performance_all_0.json')
if not os.path.exists(performance_file):
    raise ValueError('No performance file found: your network has failed.')

with open(performance_file, 'r') as fp:
    performance = json.load(fp)

# Print the feature values and names
# print("Feature values:")
# for v, l in zip(features_p1.feature_values, features_p1.feature_labels):
#    print(f"\t {l} : {v}.")

# Print the output performance
print("\n Performance:")
stats = performance['Statistics']
for k, v in stats.items():
    print(f"\t {k} {v}.")

end_time = time.time()
runtime = end_time - start_time

print('Total execution time: ', runtime)