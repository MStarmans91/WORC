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
import argparse
from WORC.processing.preprocessing import preprocess


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

    # Apply preprocessing
    image = preprocess(imagefile=args.im, config=args.para, metadata=args.md,
                       mask=args.mask)

    # Save the output
    sitk.WriteImage(image, args.out)


if __name__ == '__main__':
    main()
