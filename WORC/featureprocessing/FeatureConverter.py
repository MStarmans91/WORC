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

import pandas as pd
import collections
import WORC.addexceptions as WORCexceptions

# General labels for feature file
panda_labels = ['image_type', 'parameters', 'feature_values',
                'feature_labels']


def convert_pyradiomics_featurevector(featureVector):
    """
    Convert a PyRadiomics feature vector to WORC compatible features.
    """
    # Split center of mass features in the three dimensions
    # Save as orientation features
    COM_index = featureVector['diagnostics_Mask-original_CenterOfMassIndex']
    featureVector['of_original_COM_Index_x'] = COM_index[0]
    featureVector['of_original_COM_Index_y'] = COM_index[1]
    featureVector['of_original_COM_Index_z'] = COM_index[2]

    COM = featureVector['diagnostics_Mask-original_CenterOfMass']
    featureVector['of_original_COM_x'] = COM[0]
    featureVector['of_original_COM_y'] = COM[1]
    featureVector['of_original_COM_z'] = COM[2]

    # Delete all diagnostics features:
    for k in featureVector.keys():
        if 'diagnostics' in k:
            del featureVector[k]

    # Change label to be similar to PREDICT
    new_featureVector = collections.OrderedDict()
    texture_features = ['_glcm_', '_gldm_', '_glrlm_', '_glszm_', '_ngtdm']
    for k in featureVector.keys():
        if any(t in k for t in texture_features):
            kn = 'tf_' + k
        elif '_shape_' in k:
            kn = 'sf_' + k
        elif '_firstorder_' in k:
            kn = 'hf_' + k
        elif '_of_' in k:
            # COM
            kn = k
        else:
            message = ('Key {} is unknown!').format(k)
            raise ValueError(message)

        # Add PyRadiomics to the key
        kn = 'PyRadiomics_' + kn

        # Add to new feature Vector
        new_featureVector[kn] = featureVector[k]

    return new_featureVector


def convert_PREDICT(features, config, feat_out):
    """
    Convert features from PREDICT toolbox to WORC compatible format.

    As PREDICT is the WORC default toolbox, we only need to add the name of
    the toolbox.
    """
    # Read input features
    data = pd.read_hdf(features)

    # Add name of toolbox to the labels
    feature_labels = data.feature_labels
    feature_labels = ['PREDICT_' + l for l in feature_labels]

    # Convert to pandas Series and save as hdf5
    panda_data = pd.Series([data.image_type, data.parameters,
                            data.feature_values, feature_labels],
                           index=panda_labels,
                           name='Image features'
                           )

    print(f'Saving image features to {feat_out}.')
    panda_data.to_hdf(feat_out, 'image_features')


def convert_pyradiomics(features, config, feat_out):
    """
    Convert features from PyRadiomics toolbox to WORC compatible format.

    Description:
    """
    # Read the csv file and split in objects
    # TODO: Make sure we can read multiple output types
    data = pd.read_csv(features)
    keys = data.keys()
    values = data.values

    # Remove the non-feature elements
    omitted = ['Image', 'Mask', 'diagnostics']
    featureVector = dict()
    for k, v in zip(keys, values):
        if not any(k.startswith(om) for om in omitted):
            featureVector[k] = v

    featureVector = convert_pyradiomics_featurevector(featureVector)

    # Extract the PyRadiomics parameters
    parameters = data['diagnostics_Configuration_Settings'][0]

    # Convert to pandas Series and save as hdf5
    image_type = 'None'
    panda_data = pd.Series([image_type, parameters, featureVector.values,
                            featureVector.keys()],
                           index=panda_labels,
                           name='Image features'
                           )

    print(f'Saving image features to {feat_out}.')
    panda_data.to_hdf(feat_out, 'image_features')


def FeatureConverter(feat_in, toolbox, config, feat_out):
    """
    Convert features as extracted by a third-party toolbox to WORC format.

    Parameters
    ----------
    feat_in: string
        Path to input feature file as outputted by the feature extraction
        toolbox.

    toolbox: string
        Name of toolbox from which features are extracted.

    config: string
        Path to .ini file containing the configuration for this function.

    feat_out: string
        Path to .hdf5 file to which converted features should be saved

    """
    # Convert input features
    print(toolbox)
    if toolbox == 'PREDICT':
        convert_PREDICT(feat_in, config, feat_out)
    elif toolbox == 'PyRadiomics':
        convert_pyradiomics(feat_in, config, feat_out)
    else:
        raise WORCexceptions.WORCKeyError(f'Toolbox {toolbox} not recognized.')
