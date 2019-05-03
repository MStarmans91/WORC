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

import SimpleITK as sitk
import argparse
import pydicom
import WORC.IOparser.config_preprocessing as config_io
import os


def main():
    parser = argparse.ArgumentParser(description='Feature extraction')
    parser.add_argument('-im', '--im', metavar='image', nargs='+',
                        dest='im', type=str, required=False,
                        help='Images to calculate features on')
    parser.add_argument('-md', '--md', metavar='metadata', dest='md',
                        type=str, required=False, nargs='+',
                        help='Clinical data on patient (DICOM)')
    parser.add_argument('-mask', '--mask', metavar='mask', dest='mask',
                        type=str, required=False, nargs='+',
                        help='Mask that can be used in normalization')
    parser.add_argument('-para', '--para', metavar='Parameters', nargs='+',
                        dest='para', type=str, required=True,
                        help='Parameters')
    parser.add_argument('-out', '--out', metavar='Features',
                        dest='out', type=str, required=False,
                        help='Image output (ITK Image)')
    args = parser.parse_args()

    # Convert list inputs to strings
    if type(args.im) is list:
        args.im = ''.join(args.im)

    if type(args.md) is list:
        args.md = ''.join(args.md)

    if type(args.mask) is list:
        args.mask = ''.join(args.mask)

    if type(args.para) is list:
        args.para = ''.join(args.para)

    if type(args.out) is list:
        args.out = ''.join(args.out)

    # Read the config, image and if given masks and metadata
    config = config_io.load_config(args.para)
    image = sitk.ReadImage(args.im)

    if args.md is not None:
        metadata = pydicom.read_file(args.md)

    if args.mask is not None:
        mask = sitk.ReadImage(args.mask)

    # Convert image to Hounsfield units if type is CT
    image_type = config['ImageFeatures']['image_type']
    # NOTE: We only do this if the input is a DICOM folder
    if 'CT' in image_type and not os.path.isfile(args.im):
        print('Converting intensity to Hounsfield units.')
        image = image*metadata.RescaleSlope +\
            metadata.RescaleIntercept

    # Apply the preprocessing
    if config['Normalize']['ROI'] == 'Full':
        print('Apply z-scoring on full image.')
        image = sitk.Normalize(image)
    elif config['Normalize']['ROI'] == 'True':
        print('Apply scaling of image based on a Region Of Interest.')
        if args.mask is None:
            raise IOError('Mask input required for ROI normalization.')
        else:
            if config['Normalize']['Method'] == 'z_score':
                print('Apply scaling using z-scoring based on the ROI')

                # Cast to float to allow proper processing
                image = sitk.Cast(image, 9)

                LabelFilter = sitk.LabelStatisticsImageFilter()
                LabelFilter.Execute(image, mask)
                ROI_mean = LabelFilter.GetMean(1)
                ROI_std = LabelFilter.GetSigma(1)

                image = sitk.ShiftScale(image,
                                        shift=-ROI_mean,
                                        scale=1.0/ROI_std)
            elif config['Normalize']['Method'] == 'minmed':
                print('Apply scaling using the minimum and mean of the ROI')
                image = sitk.Cast(image, 9)

                LabelFilter = sitk.LabelStatisticsImageFilter()
                LabelFilter.Execute(image, mask)
                ROI_median = LabelFilter.GetMedian(1)
                ROI_minimum = LabelFilter.GetMinimum(1)

                image = sitk.ShiftScale(image,
                                        shift=-ROI_minimum,
                                        scale=0.5/ROI_median)
    else:
        print('No preprocessing was applied.')

    # Save the output
    sitk.WriteImage(image, args.out)


if __name__ == '__main__':
    main()
