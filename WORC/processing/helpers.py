#!/usr/bin/env python

# Copyright 2020-2020 Biomedical Imaging Group Rotterdam, Departments of
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
import numpy as np


def resample_image(image, new_spacing, interpolator=sitk.sitkBSpline):
    """Resample an image to another spacing.

    Parameters
    ----------
    image : ITK Image
        Input image.
    new_spacing : list
        Spacing to resample image to

    Returns
    -------
    resampled_image : ITK Image
        Output image.
    """
    # Get original settings
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    # ITK can only do 3D images
    if len(original_size) == 2:
        original_size = original_size + (1, )
    if len(original_spacing) == 2:
        original_spacing = original_spacing + (1.0, )

    # Compute output size
    new_size = [int(original_size[0]*original_spacing[0]/new_spacing[0]),
                int(original_size[1]*original_spacing[1]/new_spacing[1]),
                int(original_size[2]*original_spacing[2]/new_spacing[2])]

    # Set and execute the filter
    ResampleFilter = sitk.ResampleImageFilter()
    ResampleFilter.SetInterpolator(interpolator)
    ResampleFilter.SetOutputSpacing(new_spacing)
    ResampleFilter.SetSize(new_size)
    ResampleFilter.SetOutputDirection(image.GetDirection())
    ResampleFilter.SetOutputOrigin(image.GetOrigin())
    ResampleFilter.SetOutputPixelType(image.GetPixelID())
    ResampleFilter.SetTransform(sitk.Transform())
    try:
        resampled_image = ResampleFilter.Execute(image)
    except RuntimeError:
        # Assume the error is due to the direction determinant being 0
        # Crude solution: simply set a correct direction
        print('[Segmentix Warning] Bad output direction in resampling, resetting direction.')
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        ResampleFilter.SetOutputDirection(direction)
        image.SetDirection(direction)
        resampled_image = ResampleFilter.Execute(image)

    return resampled_image


def check_image_orientation(image):
    """Check the orientation of an ITK image."""
    direction_of_cosines = image.GetDirection()

    X_vector = np.abs(np.array(direction_of_cosines[0:3]))
    Y_vector = np.abs(np.array(direction_of_cosines[3:6]))

    X_index = np.argmax(X_vector)
    Y_index = np.argmax(Y_vector)

    if X_index == 0 and Y_index == 1:
        # Axial
        orientation = 1
    elif X_index == 0 and Y_index == 2:
        # Coronal
        orientation = 2
    elif X_index == 1 and Y_index == 2:
        # Sagital
        orientation = 3
    else:
        # Don't know what this is
        orientation = 4

    return orientation


def transpose_image(image, primary_axis='axial'):
    """Transpose an ITK image, while keeping the metadata."""
    orientation = check_image_orientation(image)

    if primary_axis == 'axial':
        if orientation == 1:
            print('Already in axial orientation, skipping.')
            return image

        elif orientation == 2:
            # From coronal to axial
            transform = (1, 0, 2)
            flip = (False, True, False)
            direction_of_cosines = [0, 1, 2, 3, 5, 4, 6, 7, 8]

        else:
            raise ValueError(f'Orientation {orientation} not supported for {primary_axis}.')
    else:
        raise ValueError(f'Primary axis {primary_axis} not supported.')

    # Transform image
    array = sitk.GetArrayFromImage(image)
    array = np.transpose(array, transform)
    new_image = sitk.GetImageFromArray(array)
    new_image = sitk.Flip(new_image, flip)

    # Also Transform metadata
    new_image.SetSpacing([image.GetSpacing()[t] for t in transform])
    new_image.SetOrigin([image.GetOrigin()[t] for t in transform])
    try:
        new_image.SetDirection([image.GetDirection()[d] for d in direction_of_cosines])
    except RuntimeError:
        # Bad determinant, just assume we have the correct direction already
        pass

    return new_image
