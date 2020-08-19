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
from WORC.classification.trainclassifier import trainclassifier


def main():
    parser = argparse.ArgumentParser(description='Radiomics classification')
    parser.add_argument('-feat_train', '--feat_train', metavar='features_train',
                        nargs='+', dest='feat_train', type=str, required=True,
                        help='Patient features input of first modality (HDF)')
    parser.add_argument('-feat_test', '--feat_test', metavar='features_test',
                        nargs='+', dest='feat_test', type=str, required=False,
                        default=None,
                        help='Patient features input of second modality (HDF)')
    parser.add_argument('-pc_train', '--pc_train', metavar='Patientinfo',
                        dest='pc_train',
                        type=str, required=True, nargs='+',
                        help='Classification of patient')
    parser.add_argument('-pc_test', '--pc_test', metavar='Patientinfo',
                        dest='pc_test',
                        type=str, required=False, nargs='+',
                        help='Classification of patient')
    parser.add_argument('-cf', '--conf', metavar='config', nargs='+',
                        dest='cf', type=str, required=True,
                        help='Configuration')
    parser.add_argument('-c', '--class', metavar='classification',
                        dest='hdf', type=str, required=True, nargs='+',
                        help='Classification (HDF)')
    parser.add_argument('-fs', '--fs', metavar='fixedsplits',
                        dest='fs', type=str, required=False, nargs='+',
                        help='File containing fixed splits for iterations (XLSX)')
    args = parser.parse_args()

    trainclassifier(feat_train=args.feat_train,
                    patientinfo_train=args.pc_train,
                    config=args.cf,
                    output_hdf=args.hdf,
                    feat_test=args.feat_test,
                    patientinfo_test=args.pc_test,
                    verbose=False,
                    fixedsplits=args.fs)


if __name__ == '__main__':
    main()
