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


import pandas as pd
import WORC.processing.label_processing as lp
import WORC.addexceptions as WORCexceptions
import numpy as np


def load_data(featurefiles, patientinfo=None, label_names=None, modnames=[]):
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

    # Read out all feature values and labels
    image_features_temp = list()
    feature_labels_all = list()
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
            label_data, image_features =\
                lp.findlabeldata(patientinfo,
                                 label_names,
                                 pfiles,
                                 image_features)
        except ValueError as e:
            message = e.message + '. Please take a look at your labels' +\
                ' file and make sure it is formatted correctly. ' +\
                'See also https://github.com/MStarmans91/WORC/wiki/The-WORC-configuration#genetics.'
            raise WORCexceptions.WORCValueError(message)

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
        for i in featurefiles:
            patient_IDs.append(os.path.basename(i))
        label_data = dict()
        label_data['patient_IDs'] = patient_IDs

    return label_data, image_features
