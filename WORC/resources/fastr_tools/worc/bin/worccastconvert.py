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
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description='Image conversion')
    parser.add_argument('-in', '--in', metavar='input', nargs='+',
                        dest='im', type=str, required=False,
                        help='Image in (ITK Image)')
    parser.add_argument('-out', '--out', metavar='output',
                        dest='out', type=str, required=False,
                        help='Image out (ITK Image)')
    args = parser.parse_args()

    if type(args.im) is list:
        args.im = ''.join(args.im)

    if type(args.out) is list:
        args.out = ''.join(args.out)

    # Check if input and output have the same format
    extension_input = os.path.splitext(args.im)[-1]
    extension_output = os.path.splitext(args.out)[-1]
    if extension_input == extension_output:
        # Make link
        try:
            os.symlink(args.im, args.out)
        except (OSError, AttributeError):
            print('[WORC WARNING] Cannot symlink, fallback to copying.')
            shutil.copy2(args.im, args.out)
    else:
        # Read and Write file using SimpleITK
        image = sitk.ReadImage(args.im)
        sitk.WriteImage(image, args.out)


if __name__ == '__main__':
    main()
