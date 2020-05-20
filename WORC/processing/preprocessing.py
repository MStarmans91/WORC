#!/usr/bin/env python

# Copyright 2017-2020 Biomedical Imaging Group Rotterdam, Departments of
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

import SimpleITK as sitk
import pydicom
import WORC.IOparser.config_preprocessing as config_io
import os
from WORC.processing.segmentix import dilate_contour
import numpy as np


def preprocess(imagefile, config, metadata=None, mask=None):
    '''
    Apply preprocessing to an image to prepare it for feture extration
    '''
    # Read the config, image and if given masks and metadata
    config = config_io.load_config(config)
    image = sitk.ReadImage(imagefile)

    if metadata is not None:
        metadata = pydicom.read_file(metadata)

    if mask is not None:
        mask = sitk.ReadImage(mask)

    # Convert image to Hounsfield units if type is CT
    image_type = config['ImageFeatures']['image_type']
    # NOTE: We only do this if the input is a DICOM folder
    if 'CT' in image_type and not os.path.isfile(imagefile):
        print('Converting intensity to Hounsfield units.')
        image = image*metadata.RescaleSlope +\
            metadata.RescaleIntercept

    # Apply the preprocessing
    if config['Normalize']['ROI'] == 'Full':
        print('Apply z-scoring on full image.')
        image = sitk.Normalize(image)
    elif config['Normalize']['ROI'] == 'True':
        print('Apply scaling of image based on a Region Of Interest.')

        # Dilate the mask if required
        if config['Normalize']['ROIdilate'] == 'True':
            radius = config['Normalize']['ROIdilateradius']
            print(f"Dilating ROI with radius {radius}.")
            mask = sitk.GetArrayFromImage(mask)
            mask = dilate_contour(mask, radius)
            mask = mask.astype(np.uint8)
            mask = sitk.GetImageFromArray(mask)

        if mask is None:
            if config['Normalize']['ROIDetermine'] == 'Provided':
                raise IOError('Mask input required for ROI normalization.')
            elif config['Normalize']['ROIDetermine'] == 'Otsu':
                mask = 1 - sitk.OtsuThreshold(image)
            else:
                raise IOError(f"{config['Normalize']['ROIDetermine']} is not a valid method!")
        else:
            if config['Normalize']['Method'] == 'z_score':
                print('Apply scaling using z-scoring based on the ROI')

                # Cast to float to allow proper processing
                image = sitk.Cast(image, 9)
                mask = sitk.Cast(mask, 0)

                LabelFilter = sitk.LabelStatisticsImageFilter()
                try:
                    LabelFilter.Execute(image, mask)
                except RuntimeError as e:
                    if config['General']['AssumeSameImageAndMaskMetadata']:
                        print(f'[WORC Warning] error: {e}.')
                        print(f'[WORC Warning] Assuming image and mask have same metadata.')
                        mask.CopyInformation(image)
                        LabelFilter.Execute(image, mask)
                    else:
                        raise RuntimeError(e)

                ROI_mean = LabelFilter.GetMean(1)
                ROI_std = LabelFilter.GetSigma(1)

                image = sitk.ShiftScale(image,
                                        shift=-ROI_mean,
                                        scale=1.0/ROI_std)
            elif config['Normalize']['Method'] == 'minmed':
                print('Apply scaling using the minimum and median of the ROI')
                image = sitk.Cast(image, 9)
                mask = sitk.Cast(mask, 0)

                LabelFilter = sitk.LabelStatisticsImageFilter()
                try:
                    LabelFilter.Execute(image, mask)
                except RuntimeError as e:
                    if config['General']['AssumeSameImageAndMaskMetadata']:
                        print(f'[WORC Warning] error: {e}.')
                        print(f'[WORC Warning] Assuming image and mask have same metadata.')
                        mask.CopyInformation(image)
                        LabelFilter.Execute(image, mask)
                    else:
                        raise RuntimeError(e)

                ROI_median = LabelFilter.GetMedian(1)
                ROI_minimum = LabelFilter.GetMinimum(1)

                image = sitk.ShiftScale(image,
                                        shift=-ROI_minimum,
                                        scale=0.5/ROI_median)
    else:
        print('No preprocessing was applied.')

    return image
