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
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
from datetime import datetime
import random
import numpy as np
import csv


def main():
    parser = argparse.ArgumentParser(description='Radiomics classification')
    parser.add_argument('-ed', '--ed', metavar='ed',
                        dest='ed', type=str, required=True,
                        help='Estimator data in (HDF)')
    #parser.add_argument('-tt', '--tt', metavar='tt',
    #                    dest='tt', type=str, required=True,
    #                    help='Train- and testdata in (HDF)')
    #parser.add_argument('-para', '--para', metavar='para',
    #                    dest='para', type=str, required=True,
    #                    help='Parameters (JSON)')
    parser.add_argument('-out', '--out', metavar='out',
                        dest='out', type=str, required=True,
                        help='Output: fitted estimator (HDF)')
    args = parser.parse_args()

    # Convert lists into strings
    if type(args.ed) is list:
        args.ed = ''.join(args.ed)
    #if type(args.tt) is list:
    #    args.tt = ''.join(args.tt)
    #if type(args.para) is list:
    #    args.para = ''.join(args.para)
    if type(args.out) is list:
        args.out = ''.join(args.out)

    # Read the data
    data = pd.read_hdf(args.ed)
    #traintest = pd.read_hdf(args.tt)
    #with open(args.para, 'rb') as fp:
    #    para = json.load(fp)


    # Run the smac optimization
    current_date_time = datetime.now()
    run_name = current_date_time.strftime('smac-run_' + '%m-%d_%H-%M-%S')

    # Create the output storage
    with open('/scratch/mdeen/tested_configs/' + run_name + '.csv', 'w') as file:
        csvwriter = csv.writer(file, delimiter=',',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['train_score', 'test_score', 'test_sample_counts',
                            'fit_time', 'score_time', 'para_estimator', 'para'])

    scenario = Scenario({"run_obj": "quality",  # optimize for solution quality
                         "runcount-limit": data['n_iter'],  # max. number of function evaluations;
                         "cs": data['search_space'],
                         "deterministic": "true",
                         "output_dir": "/scratch/mdeen/SMAC_output/" + run_name
                         })

    def score_cfg(cfg):
        # Construct a new dictionary with parameters from the input configuration
        parameters = convert_cfg(cfg.get_dictionary())

        # Read the data from the smac_tool
        nonlocal data
        nonlocal run_name

        # Fit the classifier and store the result
        all_scores = []
        for train, test in data['cv_iter']:
            ret = fit_and_score(data['X'], data['y'], data['scoring'],
                                train, test, parameters,
                                fit_params=data['fit_params'],
                                return_train_score=data['return_train_score'],
                                return_n_test_samples=data['return_n_test_samples'],
                                return_times=data['return_times'],
                                return_parameters=data['return_parameters'],
                                error_score=data['error_score'],
                                verbose=data['verbose'],
                                return_all=False)
            all_scores.append(ret)

        # Process the results:
        # Return the average score over all cross-validation folds
        print('all_scores: ' + str(all_scores) + '\n')
        df = pd.DataFrame(all_scores, columns=['train_score', 'test_score',
                                             'test_sample_counts', 'fit_time',
                                             'score_time', 'para_estimator', 'para'])

        with open('/scratch/mdeen/tested_configs/' + run_name + '.csv', 'a') as file:
            csvwriter = csv.writer(file, delimiter=',')
            csvwriter.writerow([float(df['train_score'].mean()), df['test_score'].mean(),
                                df['test_sample_counts'][0], df['fit_time'].mean(),
                                df['score_time'].mean(), df['para_estimator'][0],
                                df['para'][0]])

        score = 1 - df['test_score'].mean()  # We minimize so take the inverse

        return score

    run_id = random.randint(0, 2 ** 32 - 1)
    smac = SMAC4HPO(scenario=scenario,
                    tae_runner=score_cfg, run_id=run_id)
    opt_config = smac.optimize()

    '''
    # Load in the runhistory data
    runhistory_file = open('/scratch/mdeen/SMAC_output/' + run_name + '/run_' + str(run_id) +
                           '/runhistory.json')
    runhistory = json.load(runhistory_file)

    best_configs = []
    # Loop over all evaluated configurations
    for i in range(len(runhistory['configs'])):
        # We want the highest priority (low number) to be associated
        # with the worst scores so take the inverse again
        score = 1 - runhistory['data'][i][1][0]
        config = runhistory['configs'][str(i + 1)]
        parameters = convert_cfg(config)
        # If the list is shorter than the maximum
        # length, add the configuration
        if len(best_configs) < self.maxlen:
            heapq.heappush(best_configs, (score, i, parameters))
        # Otherwise, check if this config outperforms the worst one in the list
        # We use i to break ties between scores
        elif best_configs[0][0] < score:
            heapq.heapreplace(best_configs, (score, i, parameters))
'''

    source_labels = ['RET']

    with open('/scratch/mdeen/tested_configs/' + run_name + '.csv', newline='') as file:
        csvreader = csv.reader(file)
        output = list(csvreader)

    with open('/scratch/mdeen/ret-smac.txt', 'a') as retfile:
        retfile.write(str([output]))

    print(str([output]))

    source_data =\
        pd.Series([output],
                  index=source_labels,
                  name='SMAC Output')
    source_data.to_hdf(args.out, 'RET')


def convert_cfg(cfg):
    parameters = cfg
    # Add some parameters that are used for fitting, but are not part of the optimization
    parameters['random_seed'] = 42
    parameters['max_iter'] = 10000

    # No featureScaling flag is accepted in fit_and_score,
    # so remove it
    parameters.pop('use_featureScaling')
    # fit_and_score requires a flag but only if it is true
    if parameters['Imputation'] == 'False':
        parameters.pop('Imputation')
    # Delete four more flags from the config if they are false
    if parameters['StatisticalTestUse'] == 'False':
        parameters.pop('StatisticalTestUse')
    if parameters['SampleProcessing_SMOTE'] == 'False':
        parameters.pop('SampleProcessing_SMOTE')
    #if parameters['SampleProcessing_Oversampling'] == 'False':
    #    parameters.pop('SampleProcessing_Oversampling')
    #if parameters['ReliefUse'] == 'False':
    #    parameters.pop('ReliefUse')
    # 'PCAType' is either '95variance' or an int
    if parameters['UsePCA'] == 'True' and \
            parameters['PCAType'] == 'n_components':
        parameters['PCAType'] = parameters.pop('n_components')

    return parameters


if __name__ == '__main__':
    main()
