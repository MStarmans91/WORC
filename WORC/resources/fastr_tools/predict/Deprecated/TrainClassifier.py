#!/usr/bin/env python

# Copyright 2011-2017 Biomedical Imaging Group Rotterdam, Departments of
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
from PREDICT.trainclassifier import trainclassifier


def main():
    parser = argparse.ArgumentParser(description='Radiomics classification')
    parser.add_argument('-feat_m1', '--feat_m1', metavar='features_m1',
                        nargs='+', dest='feat_m1', type=str, required=True,
                        help='Patient features input of first modality (HDF)')
    parser.add_argument('-feat_m2', '--feat_m2', metavar='features_m2',
                        nargs='+', dest='feat_m2', type=str, required=False,
                        default=None,
                        help='Patient features input of second modality (HDF)')
    parser.add_argument('-feat_m3', '--feat_m3', metavar='features_m3',
                        nargs='+', dest='feat_m3', type=str, required=False,
                        default=None,
                        help='Patient features input of third modality (HDF)')
    parser.add_argument('-pc', '--pc', metavar='Patientinfo', dest='pc',
                        type=str, required=True, nargs='+',
                        help='Classification of patient')
    parser.add_argument('-cf', '--conf', metavar='config', nargs='+',
                        dest='cf', type=str, required=True,
                        help='Configuration')
    parser.add_argument('-c', '--class', metavar='classification',
                        dest='svm', type=str, required=True, nargs='+',
                        help='Classification (HDF)')
    parser.add_argument('-perf', '--perf', metavar='performance',
                        dest='perf', type=str, required=True, nargs='+',
                        help='Performance (JSON)')
    parser.add_argument('-fs', '--fs', metavar='fixedsplits',
                        dest='fs', type=str, required=False, nargs='+',
                        help='File containing fixed splits for iterations (XLSX)')
    args = parser.parse_args()

    trainclassifier(feat_m1=args.feat_m1, feat_m2=args.feat_m2,
                    feat_m3=args.feat_m3,
                    config=args.cf, patientinfo=args.pc,
                    output_svm=args.svm, output_json=args.perf, verbose=False,
                    fixedsplits=args.fs)


if __name__ == '__main__':
    main()
