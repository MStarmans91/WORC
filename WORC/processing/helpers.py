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
    resampled_image = ResampleFilter.Execute(image)
    return resampled_image
