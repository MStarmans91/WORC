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
    """Apply preprocessing to an image to prepare it for feture extration."""
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
    if config['Preprocessing']['Normalize']:
        method = config['Preprocessing']['Method']
        normalize = config['Preprocessing']['Normalize_ROI']
        dilate = config['Preprocessing']['ROIdilate']
        radius = config['Preprocessing']['ROIdilateradius']
        ROIDetermine = config['Preprocessing']['ROIDetermine']
        metadata = config['General']['AssumeSameImageAndMaskMetadata']
        image = normalize_image(image=image,
                                mask=mask,
                                method=method,
                                Normalize_ROI=normalize,
                                Dilate_ROI=dilate,
                                ROI_dilate_radius=radius,
                                ROIDetermine=ROIDetermine,
                                AssumeSameImageAndMaskMetadata=metadata)
    else:
        print('No preprocessing was applied.')

    return image


def normalize_image(image, mask=None, method='z_score', Normalize_ROI='Full',
                    Dilate_ROI='True',
                    ROI_dilate_radius=5, ROIDetermine='Provided',
                    AssumeSameImageAndMaskMetadata=True):
    """Apply normalization to an image.

    Parameters
    ----------
    image: ITK Image
        Image to normalize
    mask: None or ITK Image
        Optional: mask to be used for normalization
    method: string, default z_score
        Method to be used for normalization.
    Normalize_ROI: string, default Full
        ROI to use for normalization
    Dilate_ROI: string, default True
        Whether to dilate the ROI or not
    ROI_dilate_radius: int, default 5
        Radius to use for ROI dilation.
    ROIDetermine: string, default provided
        Whether mask is used as ROI or it is determined, e.g. through Otsu.
    AssumeSameImageAndMaskMetadata: boolean
        If True, copy metadata from image to mask.

    """
    if Normalize_ROI == 'Full':
        print('Apply z-scoring on full image.')
        image = sitk.Normalize(image)
    elif Normalize_ROI == 'True':
        print('Apply scaling of image based on a Region Of Interest.')

        # Dilate the mask if required
        if Dilate_ROI == 'True':
            radius = ROI_dilate_radius
            print(f"Dilating ROI with radius {radius}.")
            mask = sitk.GetArrayFromImage(mask)
            mask = dilate_contour(mask, radius)
            mask = mask.astype(np.uint8)
            mask = sitk.GetImageFromArray(mask)

        if mask is None:
            if ROIDetermine:
                raise IOError('Mask input required for ROI normalization.')
            elif ROIDetermine == 'Otsu':
                mask = 1 - sitk.OtsuThreshold(image)
            else:
                raise IOError(f"{ROIDetermine} is not a valid method!")
        else:
            if method == 'z_score':
                print('Apply scaling using z-scoring based on the ROI')

                # Cast to float to allow proper processing
                image = sitk.Cast(image, 9)
                mask = sitk.Cast(mask, 0)

                LabelFilter = sitk.LabelStatisticsImageFilter()
                try:
                    LabelFilter.Execute(image, mask)
                except RuntimeError as e:
                    if AssumeSameImageAndMaskMetadata:
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
            elif method == 'minmed':
                print('Apply scaling using the minimum and median of the ROI')
                image = sitk.Cast(image, 9)
                mask = sitk.Cast(mask, 0)

                LabelFilter = sitk.LabelStatisticsImageFilter()
                try:
                    LabelFilter.Execute(image, mask)
                except RuntimeError as e:
                    if AssumeSameImageAndMaskMetadata:
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

    return image
