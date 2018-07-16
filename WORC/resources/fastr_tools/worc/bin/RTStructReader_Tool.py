#!/usr/bin/env python

# Copyright 2011-2018 Biomedical Imaging Group Rotterdam, Departments of
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
from WORC.processing.RTStructReader import RTStructReader


def main():
    parser = argparse.ArgumentParser(description='RTStruct Reader')
    parser.add_argument('-rt', '--rt', metavar='rt', nargs='+',
                        dest='rt', type=str, required=False,
                        help='RT Struct file to read (DICOM)')
    parser.add_argument('-roiname', '--roiname', metavar='roiname',
                        dest='roiname',
                        type=str, required=False, nargs='+',
                        help='Name of ROI to extract (string)')
    parser.add_argument('-out', '--out', metavar='out',
                        dest='out', type=str, required=False,
                        help='Image output (ITK Image)')
    args = parser.parse_args()

    if type(args.rt) is list:
        args.rt = ''.join(args.rt)

    if type(args.roiname) is list:
        args.roiname = ''.join(args.roiname)

    if type(args.out) is list:
        args.out = ''.join(args.out)

    RTStructReader(args.rt, args.roiname)


if __name__ == '__main__':
    main()
