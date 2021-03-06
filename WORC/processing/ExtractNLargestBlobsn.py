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


from skimage.measure import label, regionprops
import numpy as np


def ExtractNLargestBlobsn(binaryImage, numberToExtract=1):
    """Extract N largest blobs from binary image.

    Arguments:
        binaryImage: boolean numpy array one or several contours.
        numberToExtract: number of blobs to extract (integer).

    Returns:
        binaryImage: boolean numpy are containing only the N
                     extracted blobs.
    """

    # Get all the blob properties.
    connectivity = binaryImage.ndim
    
    if connectivity == 3 and binaryImage.shape[2] == 1:
        # Oh no! Its a 2D image disguised as a 3D image! We must change connectivity to 2
        connectivity = 2
    
    labeledImage = label(binaryImage, connectivity=connectivity)
    blobMeasurements = regionprops(labeledImage)

    if len(blobMeasurements) == 1:
        # Single blob, return input
        binaryImage = binaryImage
    else:
        # Get all the areas
        allAreas = list()
        allCoords = list()
        for blob in blobMeasurements:
            allAreas.append(blob.area)
            allCoords.append(blob.coords)

        allAreas = np.asarray(allAreas)
        if numberToExtract > 0:
            # For positive numbers, sort in order of largest to smallest.
            # Sort them.
            indices = np.argsort(-allAreas)
        elif numberToExtract < 0:
            # For negative numbers, sort in order of smallest to largest.
            # Sort them.
            indices = np.argsort(allAreas)
        else:
            raise ValueError("Number of blobs to extract should not be zero!")

        binaryImage = np.zeros(binaryImage.shape)
        # NOTE: There must be a more efficient way to do this
        for nblob in range(0, numberToExtract):
            nblob = abs(nblob)
            coords = allCoords[indices[nblob]]
            for coord in coords:
                binaryImage[coord[0], coord[1], coord[2]] = 1

    return binaryImage
