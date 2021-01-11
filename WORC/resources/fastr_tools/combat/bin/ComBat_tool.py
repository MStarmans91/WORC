#!/usr/bin/env python

# Copyright 2016-2020 Biomedical Imaging Group Rotterdam, Departments of
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
from WORC.featureprocessing.ComBat import ComBat


def main():
    parser = argparse.ArgumentParser(description='ComBat Feature Harmonization')
    parser.add_argument('-feat_train', '--feat_train', metavar='feat_train',
                        nargs='+', dest='feat_train', type=str, required=True,
                        help='Patient features input (HDF)')
    parser.add_argument('-feat_test', '--feat_test', metavar='feat_test',
                        nargs='+', dest='feat_test', type=str, required=False,
                        default=None,
                        help='Patient features input (HDF)')
    parser.add_argument('-pc_train', '--pc_train', metavar='pc_train',
                        dest='pc_train',
                        type=str, required=True, nargs='+',
                        help='Classification of patient')
    parser.add_argument('-pc_test', '--pc_test', metavar='pc_test',
                        dest='pc_test',
                        type=str, required=False, nargs='+',
                        help='Classification of patient')
    parser.add_argument('-config', '--config', metavar='config', nargs='+',
                        dest='config', type=str, required=True,
                        help='Configuration')
    parser.add_argument('-feat_train_out', '--feat_train_out', metavar='feat_train_out',
                        nargs='+', dest='feat_train_out', type=str, required=True,
                        help='Patient features output (HDF)')
    parser.add_argument('-feat_test_out', '--feat_test_out', metavar='feat_test_out',
                        nargs='+', dest='feat_test_out', type=str, required=False,
                        default=None,
                        help='Patient features output (HDF)')
    args = parser.parse_args()

    # Convert input arguments that should be strings
    if type(args.pc_train) is list:
        args.pc_train = ''.join(args.pc_train)

    if type(args.pc_test) is list:
        args.pc_test = ''.join(args.pc_test)

    if type(args.config) is list:
        args.config = ''.join(args.config)

    ComBat(features_train_in=args.feat_train,
           labels_train=args.pc_train,
           config=args.config,
           features_train_out=args.feat_train_out,
           features_test_in=args.feat_test,
           labels_test=args.pc_test,
           features_test_out=args.feat_test_out)


if __name__ == '__main__':
    main()
