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

import argparse
import SimpleITK as sitk


def main():
    parser = argparse.ArgumentParser(description='Feature extraction')
    parser.add_argument('-source', '--source', metavar='source', nargs='+',
                        dest='source', type=str, required=True,
                        help='Source image')
    parser.add_argument('-dest', '--dest', metavar='dest', nargs='+',
                        dest='dest', type=str, required=True,
                        help='Destination image to copy metadata to')
    parser.add_argument('-out', '--out', metavar='out',
                        dest='out', type=str, required=True,
                        help='Output file (ITK Image)')
    args = parser.parse_args()

    # Convet input lists to strings
    if type(args.dest) is list:
        args.dest = ''.join(args.dest)

    if type(args.source) is list:
        args.source = ''.join(args.source)

    if type(args.out) is list:
        args.out = ''.join(args.out)

    # Read images and copy metadata
    dest = sitk.ReadImage(args.dest)
    source = sitk.ReadImage(args.source)
    dest.CopyInformation(source)

    # Write the output image
    sitk.WriteImage(dest, args.out)


if __name__ == '__main__':
    main()
