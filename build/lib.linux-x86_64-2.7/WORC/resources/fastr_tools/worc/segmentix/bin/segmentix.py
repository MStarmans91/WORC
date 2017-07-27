#!/usr/bin/env python

# Copyright 2011-2017 Biomedical Imaging Group Rotterdam, Departments of
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
        if parameters is None:
            raise IOError("You must provide a parameter file!")
        # Load variables from the confilg file
        config = config_io.load_config(parameters)

        if segmentation is not None:
            # Load segmentation and perform routines

            if type(segmentation) is list:
                segmentation = ''.join(segmentation)

            # Convert to binary image and clean up small errors/areas
            # TODO: More robust is to do this with labeling and select largest blob
            contour = sitk.ReadImage(segmentation)
            contour = sitk.GetArrayFromImage(contour)

            # BUG: remove first and last slice, fault in liver segmentations
            contour[:,:,-1] = np.zeros([contour.shape[0], contour.shape[1]])
            contour[:,:, 0] = np.zeros([contour.shape[0], contour.shape[1]])

            contour = nd.binary_fill_holes(contour)
            contour = contour.astype(bool)
            # contour = morphology.remove_small_objects(contour, min_size=2, connectivity=2, in_place=False)
            contour = ExtractNLargestBlobsn(contour, 1)

            # Expand contour depending on settings
            # TODO: this is a workaround for 3-D morphology
            if config['Segmentix']['type'] == 'Ring':
                radius = int(config['Segmentix']['radius'])
                disk = morphology.disk(radius)

                # Dilation with radius
                for ind in range(contour.shape[2]):
                    contour_d = morphology.binary_dilation(contour[:, :, ind], disk)
                    contour_e = morphology.binary_erosion(contour[:, :, ind], disk)
                    contour[:, :, ind] = np.subtract(contour_d, contour_e)

            # Mask if necessary
            if mask is not None:
                if type(mask) is list:
                    mask = ''.join(mask)

                mask = sitk.ReadImage(mask)
                mask = sitk.GetArrayFromImage(mask)
                mask = nd.binary_fill_holes(mask)
                mask = mask.astype(bool)
                method = config['Segmentix']['mask']
                if method == 'subtract':
                    contour = contour - mask
                elif method == "multiply":
                    contour = np.multiply(contour, mask)


            # Use slicer if necessary
            # if config['Segmentix']['slicer']:
            #     # Every 2-D Axial slice is a separate sample

            # Output contour
            contour = contour.astype(np.uint8)
            contour = sitk.GetImageFromArray(contour)
            if output is not None:
                sitk.WriteImage(contour, output)
        else:
            # TODO: perform segmentation routine or transform segmentation
            pass

        return contour
