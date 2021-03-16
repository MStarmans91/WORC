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


import pandas as pd
import WORC.processing.label_processing as lp
import WORC.addexceptions as WORCexceptions
import numpy as np
import os


def load_data(featurefiles, patientinfo=None, label_names=None, modnames=[],
              combine_features=False, combine_method='mean'):
    """Read feature files and stack the features per patient in an array.

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

        combine_features: boolean, default False
                Determines whether to combine the features from all samples
                of the same patient or not.

        combine_methods: string, mean or max
                If features per patient should be combined, determine how.
    """
    # Read out all feature values and labels
    image_features_temp = list()
    feature_labels_all = list()
    pids = list()
    for i_patient in range(0, len(featurefiles[0])):
        feature_values_temp = list()
        feature_labels_temp = list()
        for i_mod in range(0, len(featurefiles)):
            feat_temp = pd.read_hdf(featurefiles[i_mod][i_patient])
            feature_values_temp += feat_temp.feature_values
            if not modnames:
                # Create artificial names
                feature_labels_temp += [f + '_M' + str(i_mod) for f in feat_temp.feature_labels]
            else:
                # Use the provides modality names
                feature_labels_temp += [f + '_' + str(modnames[i_mod]) for f in feat_temp.feature_labels]

        image_features_temp.append((feature_values_temp, feature_labels_temp))

        # Also make a list of all unique label names
        feature_labels_all = feature_labels_all + list(set(feature_labels_temp) - set(feature_labels_all))

        # If PID in feature file, use those
        if 'patient' in list(feat_temp.keys()):
            pids.append(feat_temp.patient)

    # Check when we found patient ID's, if we did for all objects
    if pids:
        if len(pids) != len(image_features_temp):
            raise WORCexceptions.WORCValueError(f'Length of pids {len(pids)}' +
                                                'does not match' +
                                                'number of objects ' +
                                                str(len(image_features_temp)) +
                                                f'Found {pids}.')

    # If some objects miss certain features, we will identify these with NaN values
    feature_labels_all.sort()
    image_features = list()
    for patient in image_features_temp:
        feat_temp = patient[0]
        label_temp = patient[1]

        feat = list()
        for f in feature_labels_all:
            if f in label_temp:
                index = label_temp.index(f)
                fv = feat_temp[index]
            else:
                fv = np.NaN
            feat.append(fv)

        image_features.append((feat, feature_labels_all))

    # Get the labels and patient IDs
    if patientinfo is not None:
        # We use the feature files of the first modality to match to patient name
        pfiles = featurefiles[0]
        try:
            if pids:
                label_data, image_features =\
                    lp.findlabeldata(patientinfo,
                                     label_names,
                                     pids=pids,
                                     objects=image_features)
            else:
                label_data, image_features =\
                    lp.findlabeldata(patientinfo,
                                     label_names,
                                     filenames=pfiles,
                                     objects=image_features)
        except ValueError as e:
            message = str(e) + '. Please take a look at your labels' +\
                ' file and make sure it is formatted correctly. ' +\
                r'See also https://worc.readthedocs.io/en/latest/static/configuration.html#config-labels.'
            raise WORCexceptions.WORCValueError(message)

        if len(label_names) == 1:
            print("Labels:")
            print(label_data['label'])
            print('Total of ' + str(label_data['patient_IDs'].shape[0]) +
                  ' patients')
            pos = np.sum(label_data['label'])
            neg = label_data['patient_IDs'].shape[0] - pos
            print(('{} positives, {} negatives').format(pos, neg))
    else:
        # Use filenames as patient ID s
        patient_IDs = list()
        for i in featurefiles[0]:
            patient_IDs.append(os.path.basename(i))
        label_data = dict()
        label_data['patient_IDs'] = patient_IDs

    # Optionally, combine features of same patient
    if combine_features:
        print('Combining features of the same patient.')
        feature_labels = image_features[0][1]
        label_name = label_data['label_name']
        new_label_data = list()
        new_pids = list()
        new_features = list()
        pid_length = len(label_data['patient_IDs'])
        print(f'\tOriginal number of samples / patients: {pid_length}.')

        already_processed = list()
        for pnum, pid in enumerate(label_data['patient_IDs']):
            if pid not in already_processed:
                # NOTE: should check whether we have already processed this patient
                occurrences = list(label_data['patient_IDs']).count(pid)

                # NOTE: Assume all object from one patient have the same label
                label = label_data['label'][0][pnum]
                new_label_data.append(label)
                new_pids.append(pid)

                # Only process patients which occur multiple times
                if occurrences > 1:
                    print(f'\tFound {occurrences} occurrences for {pid}.')
                    indices = [i for i, x in enumerate(label_data['patient_IDs']) if x == pid]
                    feature_values_thispatient = np.asarray([image_features[i][0] for i in indices])
                    if combine_method == 'mean':
                        feature_values_thispatient = np.nanmean(feature_values_thispatient, axis=0).tolist()
                    else:
                        raise WORCexceptions.KeyError(f'{combine_method} is not a valid combination method, should be mean or max.')
                    features = (feature_values_thispatient, feature_labels)

                    # And add the new one
                    new_features.append(features)
                else:
                    new_features.append(image_features[pnum])

                already_processed.append(pid)

        # Adjust the labels and features for further processing
        label_data = dict()
        label_data['patient_IDs'] = np.asarray(new_pids)
        label_data['label'] = np.asarray([new_label_data])
        label_data['label_name'] = label_name

        image_features = new_features

        pid_length = len(label_data['patient_IDs'])
        print(f'\tNumber of samples / patients after combining: {pid_length}.')

    return label_data, image_features


def load_features(feat, patientinfo, label_type, combine_features=False,
                  combine_method='mean'):
    """Read feature files and stack the features per patient in an array.

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

        combine_features: boolean, default False
                Determines whether to combine the features from all samples
                of the same patient or not.

        combine_methods: string, mean or max
                If features per patient should be combined, determine how.

    """
    # Check if features is a simple list, or just one string
    if '=' not in feat[0]:
        feat = ['Mod0=' + ','.join(feat)]

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
        load_data(feat, patientinfo,
                  label_type, modnames,
                  combine_features,
                  combine_method)

    return label_data, image_features


def convert_config_pyradiomics(config):
    """Convert WORC to PyRadiomics config.

    Convert fields from WORC confiparser object to a PyRadiomics
    compatible dictionary.
    """
    # Creatae main config structure
    outputconfig = dict()
    outputconfig['imageType'] = dict()
    outputconfig['setting'] = dict()
    outputconfig['featureClass'] = dict()

    # Take out the specific PyRadiomics values
    outputconfig['setting']['geometryTolerance'] =\
        float(config['PyRadiomics']['geometryTolerance'])

    if config['PyRadiomics']['normalize'] == 'True':
        outputconfig['setting']['normalize'] = True
    else:
        outputconfig['setting']['normalize'] = False

    outputconfig['setting']['normalizeScale'] =\
        int(config['PyRadiomics']['normalizeScale'])

    outputconfig['setting']['interpolator'] =\
        config['PyRadiomics']['interpolator']

    if config['PyRadiomics']['preCrop'] == 'True':
        outputconfig['setting']['preCrop'] = True
    else:
        outputconfig['setting']['preCrop'] = False

    outputconfig['setting']['label'] =\
        int(config['PyRadiomics']['label'])

    if config['PyRadiomics']['force2D'] == 'True':
        outputconfig['setting']['force2D'] = True
    else:
        outputconfig['setting']['force2D'] = False

    outputconfig['setting']['force2Ddimension'] =\
        int(config['PyRadiomics']['force2Ddimension'])

    outputconfig['setting']['voxelArrayShift'] =\
        int(config['PyRadiomics']['voxelArrayShift'])

    if config['PyRadiomics']['binCount'] == 'None':
        outputconfig['setting']['binCount'] = None
    else:
        outputconfig['setting']['binCount'] =\
            int(config['PyRadiomics']['binCount'])

    if config['PyRadiomics']['binWidth'] == 'None':
        outputconfig['setting']['binWidth'] = None
    else:
        outputconfig['setting']['binWidth'] =\
            float(config['PyRadiomics']['binWidth'])

    if config['PyRadiomics']['resampledPixelSpacing'] == 'None':
        outputconfig['setting']['resampledPixelSpacing'] = None
    else:
        outputconfig['setting']['resampledPixelSpacing'] =\
            [float(i) for i in config['PyRadiomics']['resampledPixelSpacing'].split(',')]
        if len(outputconfig['setting']['resampledPixelSpacing']) != 3:
            length = len(outputconfig['setting']['resampledPixelSpacing'])
            raise WORCexceptions.WORCValueError(f'Length PyRadiomics resampledPixelSpacing should be 3, got {length}.')

    # Extract several general values as well
    # Convert strings with values to list of ints
    distances = config['ImageFeatures']['GLCM_distances']
    distances = distances.split(',')
    distances = [int(s) for s in distances]
    outputconfig['setting']['distances'] = distances

    # Check if we need to apply transforms to the image
    if config['PyRadiomics']['Original'] == 'True':
        outputconfig['imageType']['Original'] = dict()

    if config['PyRadiomics']['Wavelet'] == 'True':
        outputconfig['imageType']['Wavelet'] = dict()

    if config['PyRadiomics']['LoG'] == 'True':
        outputconfig['imageType']['LoG'] = dict()
        sigmas = config['ImageFeatures']['log_sigma']
        sigmas = sigmas.split(',')
        sigmas = [int(s) for s in sigmas]
        outputconfig['imageType']['LoG']['sigma'] = sigmas

    # Determine which features to extract:
    if config['PyRadiomics']['extract_firstorder'] == 'True':
        outputconfig['featureClass']['firstorder'] = None
    if config['PyRadiomics']['extract_shape'] == 'True':
        outputconfig['featureClass']['shape'] = None
    if config['PyRadiomics']['texture_GLCM'] == 'True':
        outputconfig['featureClass']['glcm'] = None
    if config['PyRadiomics']['texture_GLRLM'] == 'True':
        outputconfig['featureClass']['glrlm'] = None
    if config['PyRadiomics']['texture_GLSZM'] == 'True':
        outputconfig['featureClass']['glszm'] = None
    if config['PyRadiomics']['texture_GLDM'] == 'True':
        outputconfig['featureClass']['gldm'] = None
    if config['PyRadiomics']['texture_NGTDM'] == 'True':
        outputconfig['featureClass']['ngtdm'] = None

    return outputconfig
