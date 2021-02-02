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
import json
import pandas as pd
from joblib import Parallel, delayed
from WORC.classification.fitandscore import fit_and_score
import WORC.addexceptions as ae


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
    parser.add_argument('-verbose', '--verbose', metavar='verbose',
                        nargs='+', dest='verbose', type=str, required=False,
                        default=None, help='verbose')
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
    if type(args.verbose) is list:
        args.verbose = ''.join(args.verbose)

    # Read the data
    data = pd.read_hdf(args.ed)
    traintest = pd.read_hdf(args.tt)
    with open(args.para, 'rb') as fp:
        para = json.load(fp)

    # Check whether verbose is given or not
    if args.verbose is None:
        args.verbose = False
    elif args.verbose == 'False':
        args.verbose = False
    elif args.verbose == 'True':
        args.verbose = True
    else:
        raise ae.WORCKeyError(f'{args.verbose} is not a valid verbose option!')

    # Run the tool
    n_cores = 1
    if data['return_all']:
        # Return not only the performance, but all fitted objects
        (ret, GroupSel, VarSel, SelectModel, _, scaler,
            encoder, imputer, pca, StatisticalSel, ReliefSel,
            Sampler) = Parallel(
            n_jobs=n_cores, verbose=args.verbose,
            pre_dispatch=2*n_cores
        )(delayed(fit_and_score)(X=data['X'], y=data['y'],
                                 scoring=data['scoring'],
                                 train=traintest['train'],
                                 test=traintest['test'], verbose=args.verbose,
                                 parameters=parameters,
                                 fit_params=data['fit_params'],
                                 return_train_score=data['return_train_score'],
                                 return_parameters=data['return_parameters'],
                                 return_n_test_samples=data['return_n_test_samples'],
                                 return_times=data['return_times'],
                                 return_estimator=data['return_estimator'],
                                 error_score=data['error_score'],
                                 return_all=data['return_all'],
                                 refit_workflows=data['refit_workflows'])
          for parameters in para.values())

        source_labels = ['RET', 'GroupSel', 'VarSel', 'SelectModel',
                         'scaler', 'encoder', 'imputer',
                         'pca', 'StatisticalSel', 'ReliefSel', 'Sampler']

        source_data =\
            pd.Series([ret, GroupSel, VarSel, SelectModel, scaler,
                       encoder, imputer, pca, StatisticalSel, ReliefSel,
                       Sampler],
                      index=source_labels,
                      name='Fit and Score Output')
        source_data.to_hdf(args.out, 'RET')

    else:
        ret = Parallel(
            n_jobs=n_cores, verbose=args.verbose,
            pre_dispatch=2*n_cores
        )(delayed(fit_and_score)(X=data['X'], y=data['y'],
                                 scoring=data['scoring'],
                                 train=traintest['train'],
                                 test=traintest['test'], verbose=args.verbose,
                                 parameters=parameters,
                                 fit_params=data['fit_params'],
                                 return_train_score=data['return_train_score'],
                                 return_parameters=data['return_parameters'],
                                 return_n_test_samples=data['return_n_test_samples'],
                                 return_times=data['return_times'],
                                 return_estimator=data['return_estimator'],
                                 error_score=data['error_score'],
                                 return_all=data['return_all'],
                                 refit_workflows=data['refit_workflows'])
          for parameters in para.values())

        source_labels = ['RET']

        source_data =\
            pd.Series([ret],
                      index=source_labels,
                      name='Fit and Score Output')
        source_data.to_hdf(args.out, 'RET')


if __name__ == '__main__':
    main()
