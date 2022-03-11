#!/usr/bin/env python

# Copyright 2016-2022 Biomedical Imaging Group Rotterdam, Departments of
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

import numpy as np
import configparser
import SimpleITK as sitk
from WORC.addexceptions import WORCKeyError, WORCValueError
from WORC.processing.label_processing import findlabeldata

quantitative_modalities = ['CT', 'PET', 'Thermography', 'ADC', 'MG']
qualitative_modalities = ['MRI', 'MR', 'DWI', 'US']
all_modalities = quantitative_modalities + qualitative_modalities


class Fingerprinter(object):
    """Fingerprinting object for WORC configuration."""

    def __init__(self):
        """Initialize object."""
        self.images = None
        self.segmentations = None
        self.features = None
        self.labels = None
        self.configuration = None
        self.type = None

    def execute(self):
        """Determine fingerprint of dataset.

        Parameters
        ----------
        worcobject: WORC object
            WORC object to fingerprint

        """
        # Read the config file
        config = configparser.ConfigParser()
        config.read(self.configuration)

        if self.type == 'classification':
            # Check class balance
            label_type = config['Labels']['label_names']
            if len(label_type) != 1:
                # multiclass pass
                pass
            else:
                if self.images:
                    label_data, objects_out =\
                        findlabeldata(patientinfo=self.labels,
                                      label_type=label_type,
                                      filenames=self.images)

                elif self.features:
                    label_data, objects_out =\
                        findlabeldata(patientinfo=self.labels,
                                      label_type=label_type,
                                      filenames=self.features)
                else:
                    raise WORCValueError('Either need images or features for classification fingerprinting, neither provided.')

                labels = label_data['label'][0].tolist()
                if len(np.unique(labels)) == 2:
                    # Binary classification, check class balance
                    total = float(len(labels))
                    positives = float(np.sum(labels))
                    negatives = total - positives
                    print(positives, negatives, positives/total, negatives/total)
                    if negatives/total > 0.60 or positives/total > 0.60:
                        # Class imbalance, keep resampling
                        print('Class imbalance, keep default Resampling usage of 0.20.')
                        config['Resampling']['Use'] = '0.20'
                    else:
                        # No class imbalance, turn resampling off
                        print('No class imbalance, setting Resampling usage to 0.0.')
                        config['Resampling']['Use'] = '0.00'

        elif self.type == 'images':
            # Determine modality
            modality = config['ImageFeatures']['image_type']

            if modality in qualitative_modalities:
                # Apply image normalization and use fixed bin count
                print('Qualitative modality: Apply image normalization and use fixed bin count.')
                config['Preprocessing']['Normalize'] = 'True'
                config['PyRadiomics']['binCount'] = config['ImageFeatures']['GLCM_levels']
                config['PyRadiomics']['binWidth'] = 'None'

            elif modality in quantitative_modalities:
                # Apply no image normalization and use fixed bin width
                print('Quantitative modality: Apply no image normalization and use fixed bin width.')
                config['Preprocessing']['Normalize'] = 'False'
                config['PyRadiomics']['binCount'] = 'None'
                config['PyRadiomics']['binWidth'] = '25'  # Default of PyRadiomics

            else:
                raise WORCKeyError(f'{modality} is not a known modality, should be one of {all_modalities}.')

            # Determine if we are dealing with 2D, 2.5D, or 3D images and segmentations
            spacings_x = list()
            spacings_y = list()
            spacings_z = list()

            max_num_images = int(config['Fingerprinting']['max_num_image'])
            if len(self.images) > max_num_images:
                self.images = self.images[0:max_num_images]
                self.segmentations = self.segmentations[0:max_num_images]

            for imagefile in self.images:
                image = sitk.ReadImage(imagefile)
                spacings_x.append(image.GetSpacing()[0])
                spacings_y.append(image.GetSpacing()[1])
                spacings_z.append(image.GetSpacing()[2])

            mean_spacings = [np.mean(spacings_x), np.mean(spacings_y), np.mean(spacings_z)]

            # Assume x and y spacing are the same, so we just compare x with z
            if mean_spacings[2] > 2.0*mean_spacings[0]:
                # 2.5D images, use standard feature extraction
                print(f'Mean spacing {mean_spacings}, thus 2.5D images, use standard feature extraction.')
                config['ImageFeatures']['extraction_mode'] = '2.5D'
                pass
            else:
                # 3D images, thus compute 3D shape features
                print(f'Mean spacing {mean_spacings}, thus 3D images, use 3D feature extraction.')
                config['ImageFeatures']['extraction_mode'] = '3D'

                # We keep all the default 2.5D features, as most of them are only defined in 2D
                config['ImageFeatures']['texture_Gabor'] = 'True'
                config['ImageFeatures']['texture_LBP'] = 'True'
                config['ImageFeatures']['texture_GLCM'] = 'True'
                config['ImageFeatures']['texture_GLCMMS'] = 'True'
                config['ImageFeatures']['vessel'] = 'True'
                config['ImageFeatures']['log'] = 'True'
                config['ImageFeatures']['phase'] = 'True'

            # Check if segmentations are 2D or 3D
            num_masked_slices_all = list()
            for segmentationfile in self.segmentations:
                segmentation = sitk.GetArrayFromImage(sitk.ReadImage(segmentationfile))
                segmentation = segmentation.astype(np.bool)
                num_masked_slices = len(np.flatnonzero(np.any(segmentation, axis=(1, 2))))
                num_masked_slices_all.append(num_masked_slices)

            if all(elem == 1 for elem in num_masked_slices_all):
                print('All masks only contain one slice, so turn of 3D features.')
                # NOTE: PREDICT will mostly switch itself between these features by looking at the masks
                config['ImageFeatures']['extraction_mode'] = '2D'

                config['ImageFeatures']['orientation'] = 'False'
                config['ImageFeatures']['texture_GLCMMS'] = 'False'

                # For PyRadiomics, only this parameter needs to be changed
                config['PyRadiomics']['force2D'] = 'True'

        else:
            raise WORCValueError(f'Type {type} is not valid for fingeprinting. Should be one of ["classification", "images"].')

        return config
