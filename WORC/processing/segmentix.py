#!/usr/bin/env python

# Copyright 2017-2021 Biomedical Imaging Group Rotterdam, Departments of
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

from WORC.processing.ExtractNLargestBlobsn import ExtractNLargestBlobsn
import SimpleITK as sitk
from skimage import morphology
import scipy.ndimage as nd
import numpy as np
import WORC.IOparser.config_segmentix as config_io
from WORC.processing import helpers as h
import pydicom
import WORC.addexceptions as ae


def get_ring(contour, radius=5):
    """Get a ring on the boundary of the contour.

    PARAMETERS
    ----------
    contour: numpy array
            Array containing the contour

    radius: int, default 5
            Radius of ring to be extracted.

    """
    contour = contour.astype(bool)
    radius = int(radius)
    disk = morphology.disk(radius)

    # Dilation with radius in axial direction
    for ind in range(contour.shape[0]):
        contour_d = morphology.binary_dilation(contour[ind, :, :], disk)
        contour_e = morphology.binary_erosion(contour[ind, :, :], disk)
        contour[ind, :, :] = np.bitwise_xor(contour_d, contour_e)

    return contour


def dilate_contour(contour, radius=5):
    """Dilate the contour.

    PARAMETERS
    ----------
    contour: numpy array
            Array containing the contour

    radius: int, default 5
            Radius of ring to be extracted.

    """
    contour = contour.astype(bool)
    radius = int(radius)
    disk = morphology.disk(radius)

    # Dilation with radius in axial direction
    for ind in range(contour.shape[0]):
        contour[ind, :, :] = morphology.binary_dilation(contour[ind, :, :],
                                                        disk)
    return contour


def mask_contour(contour, mask, method='multiply'):
    """Apply a mask to a contour.

    PARAMETERS
    ----------
    contour: numpy array
            Array containing the contour

    mask: string
            path referring to the mask used for the final segmentation.
            Should be a format compatible with ITK, e.g. .nii, .nii.gz, .mhd,
            .raw, .tiff, .nrrd.

    method: string
            How masking is applied: can be subtract or pairwise

    """
    mask = sitk.ReadImage(mask)
    mask = sitk.GetArrayFromImage(mask)
    mask = nd.binary_fill_holes(mask)
    mask = mask.astype(bool)

    if method == 'subtract':
        contour = np.bitwise_xor(contour, mask)
    elif method == "multiply":
        contour = np.multiply(contour, mask)

    return contour


def segmentix(parameters, image=None, segmentation=None,
              output=None, metadata_file=None, mask=None):
    """Alter a segmentation.

    Segmentix is a mixture of processing methods that can be applied to
    agument a segmentation. Examples include selecting only the largest blob
    and the application of morphological operations.

    Parameters
    ----------
    parameters: string, mandatory
            Contains the path referring to a .ini file in which the
            parameters to be used are specified. See the Github Wiki
            for more details on the format and content.

    image: string, optional
            Note implemented yet! Image to be used for automatic segmentation.

    segmentation: string, currently mandatory
            path referring to the input segmentation file. Should be a
            format compatible with ITK, e.g. .nii, .nii.gz, .mhd,
            .raw, .tiff, .nrrd.

    output: string, mandatory
            path referring to the output segmentation file. Should be a
            format compatible with ITK, e.g. .nii, .nii.gz, .mhd,
            .raw, .tiff, .nrrd.

    metadata_file: string, optional
            Note implemented yet! Path referring to the .dcm from which
            fields can be used as metadata for segmentation.

    mask: string, optional
            path referring to the mask used for the final segmentation.
            Should be a format compatible with ITK, e.g. .nii, .nii.gz, .mhd,
            .raw, .tiff, .nrrd.

    """
    # Load variables from the confilg file
    config = config_io.load_config(parameters)

    # Load segmentation and perform routines
    if type(segmentation) is list:
        segmentation = ''.join(segmentation)

    if type(image) is list:
        image = ''.join(image)

    # Load the image onvert to array
    contour_original = sitk.ReadImage(segmentation)
    contour = sitk.GetArrayFromImage(contour_original)

    if config['Segmentix']['fillholes']:
        print('[Segmentix] Filling holes.')
        contour = nd.binary_fill_holes(contour)

    if config['Segmentix']['remove_small_objects']:
        print('[Segmentix] Removing small objects.')
        min_size = config['Segmentix']['min_object_size']
        contour = morphology.remove_small_objects(contour, min_size=min_size,
                                                  connectivity=2,
                                                  in_place=False)

    if config['Segmentix']['N_blobs'] != 0:
        print('[Segmentix] Extracting largest blob.')
        contour = contour.astype(bool)
        contour = ExtractNLargestBlobsn(contour, 1)

    # Expand contour depending on settings
    # TODO: this is a workaround for 3-D morphology
    if config['Segmentix']['type'] == 'Ring':
        print('[Segmentix] Converting contour to ring around existing contour.')
        radius = config['Segmentix']['radius']
        contour = get_ring(contour, radius)

    elif config['Segmentix']['type'] == 'Dilate':
        print('[Segmentix] Dilating contour.')
        radius = config['Segmentix']['radius']
        contour = dilate_contour(contour, radius)

    # Mask the segmentation if necessary
    if mask is not None:
        method = config['Segmentix']['mask']
        print('[Segmentix] Masking contour.')
        if type(mask) is list:
            mask = ''.join(mask)
        new_contour = mask_contour(contour, mask, method)
        if np.sum(new_contour) == 0:
            print('\t [WARNING] Contour after masking sums to zero: not applying masking.')
        else:
            contour = new_contour

    # Convert back to ITK Image with correct meta information
    contour = contour.astype(np.uint8)
    contour = sitk.GetImageFromArray(contour)

    if config['Segmentix']['AssumeSameImageAndMaskMetadata']:
        print('[Segmentix] Copy metadata information from image to mask.')
        image = sitk.ReadImage(image)
        contour.CopyInformation(image)
    else:
        contour.CopyInformation(contour_original)

    # Detect incorrect spacings
    if config['Preprocessing']['CheckSpacing']:
        if metadata_file is not None:
            metadata = pydicom.read_file(metadata_file)
        else:
            raise ae.WORCValueError('When correcting for spacing, you need to input metadata.')

        if image.GetSpacing() == (1, 1, 1):
            print('Detected 1x1x1 spacing, overwriting with DICOM metadata.')
            if [0x18, 0x50] in list(metadata.keys()):
                slice_thickness = metadata[0x18, 0x50].value
            elif [0x18, 0x88] in list(metadata.keys()):
                # Take spacing between slices
                slice_thickness = metadata[0x18, 0x88].value
            else:
                slice_thickness = 1.0

            if [0x28, 0x30] in list(metadata.keys()):
                pixel_spacing = metadata[0x28, 0x30].value
            else:
                pixel_spacing = [1.0, 1.0]

            spacing = (float(pixel_spacing[0]),
                       float(pixel_spacing[1]),
                       float(slice_thickness))
            contour.SetSpacing(spacing)

    # Apply re-orientation of the image
    if config['Preprocessing']['CheckOrientation']:
        primary_axis = config['Preprocessing']['OrientationPrimaryAxis']
        print(f'Apply re-orientation of image to {primary_axis} if required.')
        contour = h.transpose_image(image=contour, primary_axis=primary_axis)
    else:
        print('No re-orientation was applied.')

    # Apply resampling if required
    if config['Preprocessing']['Resampling']:
        new_spacing = config['Preprocessing']['Resampling_spacing']
        print(f'[Segmentix] Apply resampling of image to spacing {new_spacing}.')
        contour = h.resample_image(image=contour, new_spacing=new_spacing,
                                   interpolator=sitk.sitkNearestNeighbor)
    else:
        print('No resampling was applied.')

    # If required, output contour
    if output is not None:
        print(f'[Segmentix] Writing image to {output}.')
        sitk.WriteImage(contour, output)

    return contour
