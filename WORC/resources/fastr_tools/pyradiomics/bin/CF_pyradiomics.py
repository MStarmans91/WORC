#!/usr/bin/env python

# Copyright 2017-2018 Biomedical Imaging Group Rotterdam, Departments of
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

import logging
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import collections


def convertconfig(parameters):
    kwargs = dict()
    kwargs['binWidth'] = 25
    kwargs['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
    kwargs['interpolator'] = sitk.sitkBSpline
    kwargs['verbose'] = True

    # Specific MR Settings: see https://github.com/Radiomics/pyradiomics/blob/master/examples/exampleSettings/exampleMR_NoResampling.yaml
    kwargs['normalize'] = True
    kwargs['normalizeScale'] = 100
    kwargs['preCrop'] = True
    kwargs['force2D'] = True
    kwargs['force2Ddimension'] = 0  # axial slices, for coronal slices, use dimension 1 and for sagittal, dimension 2.
    kwargs['binWidth'] = 5
    kwargs['voxelArrayShift'] = 300
    kwargs['label'] = 1

    # NOTE: A little more tolerance may be required on matching the dimensions
    kwargs['geometryTolerance'] = 1E-3

    return kwargs


def AllFeatures(image, mask, parameters=None):
    if parameters is None:
        # Default settings for signature calculation from PyRadiomics
        kwargs = {}

        # These are currently set equal to the respective default values
        kwargs['binWidth'] = 25
        kwargs['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
        kwargs['interpolator'] = sitk.sitkBSpline
        kwargs['verbose'] = True

        # Specific MR Settings: see https://github.com/Radiomics/pyradiomics/blob/master/examples/exampleSettings/exampleMR_NoResampling.yaml
        kwargs['normalize'] = True
        kwargs['normalizeScale'] = 100
        kwargs['preCrop'] = True
        kwargs['force2D'] = True
        kwargs['force2Ddimension'] = 0  # axial slices, for coronal slices, use dimension 1 and for sagittal, dimension 2.
        kwargs['binWidth'] = 5
        kwargs['voxelArrayShift'] = 300
        kwargs['label'] = 1

        # NOTE: A little more tolerance may be required on matching the dimensions
        kwargs['geometryTolerance'] = 1E-3
    else:
        # Extract fields of parameters dict to right kwargs arguments
        kwargs = convertconfig(parameters)

    # Initialize wrapperClass to generate signature
    extractor = featureextractor.RadiomicsFeaturesExtractor(**kwargs)

    # Disable all classes except firstorder
    extractor.enableAllFeatures()

    # Prevent radiomics logger from printing out log entries with level < WARNING to the console
    logger = logging.getLogger('radiomics')
    logger.handlers[0].setLevel(logging.WARNING)
    logger.propagate = False  # Do not pass log messages on to root logger

    # Write out all log entries to a file
    handler = logging.FileHandler(filename='testLog.txt', mode='w')
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    print("Calculating features")

    featureVector = extractor.execute(image, mask)

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

    featureVector = new_featureVector

    # Print the values and keys
    nfeat = len(featureVector.keys())
    print(('Total of {} feature computed:').format(str(nfeat)))
    for featureName in featureVector.keys():
        print("Computed %s: %s" % (featureName, featureVector[featureName]))

    return featureVector, kwargs
