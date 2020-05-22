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

import argparse
from WORC.featureprocessing.FeatureConverter import FeatureConverter


def main():
    parser = argparse.ArgumentParser(description='Radiomics classification')
    parser.add_argument('-feat_in', '--feat_in', metavar='feat_in',
                        nargs='+', dest='feat_in', type=str, required=True,
                        help='Patient features input of first modality (HDF)')
    parser.add_argument('-toolbox', '--toolbox', metavar='toolbox', nargs='+',
                        dest='toolbox', type=str, required=True,
                        help='Toolbox used for feature calculation')
    parser.add_argument('-cf', '--conf', metavar='config', nargs='+',
                        dest='cf', type=str, required=True,
                        help='Configuration')
    parser.add_argument('-feat_out', '--feat_out', metavar='feat_out',
                        nargs='+', dest='feat_out', type=str, required=True,
                        default=None,
                        help='Patient features input of second modality (HDF)')
    args = parser.parse_args()

    # Convert several input arguments from lists to strings
    if type(args.feat_in) is list:
        args.feat_in = ''.join(args.feat_in)

    if type(args.toolbox) is list:
        args.toolbox = ''.join(args.toolbox)

    if type(args.cf) is list:
        args.cf = ''.join(args.cf)

    if type(args.feat_out) is list:
        args.feat_out = ''.join(args.feat_out)
    # Run converter
    FeatureConverter(feat_in=args.feat_in,
                     toolbox=args.toolbox,
                     config=args.cf,
                     feat_out=args.feat_out)


if __name__ == '__main__':
    main()
