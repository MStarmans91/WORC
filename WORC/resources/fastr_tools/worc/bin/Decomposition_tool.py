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
from WORC.featureprocessing.Decomposition import Decomposition


def main():
    parser = argparse.ArgumentParser(description='Radiomics classification')
    parser.add_argument('-feat', '--feat', metavar='features',
                        nargs='+', dest='feat', type=str, required=True,
                        help='Patient features input of first modality (HDF)')
    parser.add_argument('-pc', '--pc', metavar='Patientinfo',
                        dest='pc',
                        type=str, required=True, nargs='+',
                        help='Classification of patient')
    parser.add_argument('-cf', '--conf', metavar='config', nargs='+',
                        dest='cf', type=str, required=True,
                        help='Configuration')
    parser.add_argument('-output', '--output', metavar='output',
                        dest='output', type=str, required=True, nargs='+',
                        help='Output (PNG)')
    args = parser.parse_args()

    # Convert inputs to lists
    if type(args.pc) is list:
        args.pc = ''.join(args.pc)

    if type(args.cf) is list:
        args.cf = ''.join(args.cf)

    if type(args.output) is list:
        args.output = ''.join(args.output)

    Decomposition(features=args.feat,
                  patientinfo=args.pc,
                  config=args.cf,
                  output=args.output,
                  verbose=False)


if __name__ == '__main__':
    main()
