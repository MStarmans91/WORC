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
import PREDICT.classifiers.SVM.parameter_optimization as po
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Radiomics classification')
    parser.add_argument('-data', '--data', metavar='data',
                        dest='data', type=str, required=True,
                        help='Data in (HDF)')
    parser.add_argument('-svm', '--svm', metavar='svm',
                        dest='svm', type=str, required=True,
                        help='SVM specifications (HDF)')
    args = parser.parse_args()

    data = pd.read_hdf(args.data)

    X_train = data['X_train']
    Y_train = data['Y_train']
    config = data['config']

    # Find best hyperparameters and construct svm
    svm = po.random_search_parameters(X_train, Y_train,
                                      **config['HyperOptimization'])

    source_labels = ['svm', 'X_train', 'X_test', 'Y_train', 'Y_test',
                     'config', 'patient_ID_train', 'patient_ID_test',
                     'random_seed', 'scaler']

    source_data =\
        pd.Series([svm, X_train, data['X_test'], Y_train,
                   data['Y_test'], config, data['patient_ID_train'],
                   data['patient_ID_test'], data['random_seed'], data['scaler']],
                  index=source_labels,
                  name='Source Data')
    source_data.to_hdf(args.svm, 'SVM')


if __name__ == '__main__':
    main()
