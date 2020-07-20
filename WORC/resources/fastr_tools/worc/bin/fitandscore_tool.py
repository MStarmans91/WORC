#!/usr/bin/env python

# Copyright 2016-2019 Biomedical Imaging Group Rotterdam, Departments of
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
import json
import pandas as pd
from joblib import Parallel, delayed
from WORC.classification.fitandscore import fit_and_score


def main():
    parser = argparse.ArgumentParser(description='Radiomics classification')
    parser.add_argument('-ed', '--ed', metavar='ed',
                        dest='ed', type=str, required=True,
                        help='Estimator data in (HDF)')
    parser.add_argument('-tt', '--tt', metavar='tt',
                        dest='tt', type=str, required=True,
                        help='Train- and testdata in (HDF)')
    parser.add_argument('-para', '--para', metavar='para',
                        dest='para', type=str, required=True,
                        help='Parameters (JSON)')
    parser.add_argument('-out', '--out', metavar='out',
                        dest='out', type=str, required=True,
                        help='Output: fitted estimator (HDF)')
    args = parser.parse_args()

    # Convert lists into strings
    if type(args.ed) is list:
        args.ed = ''.join(args.ed)
    if type(args.tt) is list:
        args.tt = ''.join(args.tt)
    if type(args.para) is list:
        args.para = ''.join(args.para)
    if type(args.out) is list:
        args.out = ''.join(args.out)

    # Read the data
    data = pd.read_hdf(args.ed)
    traintest = pd.read_hdf(args.tt)
    with open(args.para, 'rb') as fp:
        para = json.load(fp)

    n_cores = 1
    ret = Parallel(
        n_jobs=n_cores, verbose=data['verbose'],
        pre_dispatch=2*n_cores
    )(delayed(fit_and_score)(X=data['X'], y=data['y'],
                             scoring=data['scoring'],
                             train=traintest['train'],
                             test=traintest['test'], verbose=data['verbose'],
                             para=parameters, fit_params=data['fit_params'],
                             return_train_score=data['return_train_score'],
                             return_parameters=data['return_parameters'],
                             return_n_test_samples=data['return_n_test_samples'],
                             return_times=data['return_times'],
                             error_score=data['error_score'],
                             return_all=False)
      for parameters in para.values())

    source_labels = ['RET']

    source_data =\
        pd.Series([ret],
                  index=source_labels,
                  name='Fit and Score Output')
    source_data.to_hdf(args.out, 'RET')


if __name__ == '__main__':
    main()
