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

from WORC.processing.ExtractNLargestBlobsn import ExtractNLargestBlobsn
import SimpleITK as sitk
from skimage import morphology
import scipy.ndimage as nd
import numpy as np
import WORC.IOparser.config_segmentix as config_io


def segmentix(parameters=None, image=None, segmentation=None,
              output=None, metadata_file=None, mask=None):
        '''
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

        '''
        if parameters is None:
            raise IOError("You must provide a parameter file!")
        # Load variables from the confilg file
        config = config_io.load_config(parameters)

        if segmentation is not None:
            # Load segmentation and perform routines

            if type(segmentation) is list:
                segmentation = ''.join(segmentation)

            # Convert to binary image and clean up small errors/areas
            contour_original = sitk.ReadImage(segmentation)
            contour = sitk.GetArrayFromImage(contour_original)

            # BUG: remove first and last slice, fault in liver segmentations
            contour[:, :, -1] = np.zeros([contour.shape[0], contour.shape[1]])
            contour[:, :, 0] = np.zeros([contour.shape[0], contour.shape[1]])

            if config['Segmentix']['fillholes']:
                contour = nd.binary_fill_holes(contour)

            # contour = morphology.remove_small_objects(contour, min_size=2, connectivity=2, in_place=False)
            if config['Segmentix']['N_blobs'] != 0:
                contour = contour.astype(bool)
                contour = ExtractNLargestBlobsn(contour, 1)

            # Expand contour depending on settings
            # TODO: this is a workaround for 3-D morphology
            if config['Segmentix']['type'] == 'Ring':
                contour = contour.astype(bool)
                radius = int(config['Segmentix']['radius'])
                disk = morphology.disk(radius)

                # Dilation with radius in axial direction
                for ind in range(contour.shape[0]):
                    contour_d = morphology.binary_dilation(contour[ind, :, :], disk)
                    contour_e = morphology.binary_erosion(contour[ind, :, :], disk)
                    contour[ind, :, :] = np.bitwise_xor(contour_d, contour_e)

            # Mask the segmentation if necessary
            if mask is not None:
                if type(mask) is list:
                    mask = ''.join(mask)

                mask = sitk.ReadImage(mask)
                mask = sitk.GetArrayFromImage(mask)
                mask = nd.binary_fill_holes(mask)
                mask = mask.astype(bool)
                method = config['Segmentix']['mask']
                if method == 'subtract':
                    contour = np.bitwise_xor(contour, mask)
                elif method == "multiply":
                    contour = np.multiply(contour, mask)

            # Output contour
            contour = contour.astype(np.uint8)
            contour = sitk.GetImageFromArray(contour)
            contour.CopyInformation(contour_original)
            if output is not None:
                sitk.WriteImage(contour, output)
        else:
            # TODO: perform segmentation routine or transform segmentation
            pass

        return contour
