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
from WORC.featureprocessing.StatisticalTestFeatures import StatisticalTestFeatures


def main():
    parser = argparse.ArgumentParser(description='Radiomics classification')
    parser.add_argument('-feat', '--feat', metavar='features',
                        nargs='+', dest='feat', type=str, required=True,
                        help='Patient features input of first modality (HDF)')
    parser.add_argument('-pc', '--pc', metavar='Patientinfo', dest='pc',
                        type=str, required=True, nargs='+',
                        help='Classification of patient')
    parser.add_argument('-cf', '--conf', metavar='config', nargs='+',
                        dest='cf', type=str, required=True,
                        help='Configuration')
    parser.add_argument('-output_csv', '--output_csv', metavar='output_csv',
                        dest='output_csv', type=str, required=True, nargs='+',
                        help='P-values of statistical tests (CSV)')
    parser.add_argument('-output_png', '--output_png', metavar='output_png',
                        dest='output_png', type=str, required=False, nargs='+',
                        help='P-values of statistical tests (PNG)')
    parser.add_argument('-output_tex', '--output_tex', metavar='output_tex',
                        dest='output_tex', type=str, required=False, nargs='+',
                        help='P-values of statistical tests (Tex)')
    args = parser.parse_args()

    if type(args.pc) is list:
        args.pc = ''.join(args.pc)

    if type(args.cf) is list:
        args.cf = ''.join(args.cf)

    StatisticalTestFeatures(features=args.feat,
                            patientinfo=args.pc,
                            config=args.cf,
                            output_csv=args.output_csv,
                            output_png=args.output_png,
                            output_tex=args.output_tex,
                            verbose=False)


if __name__ == '__main__':
    main()
