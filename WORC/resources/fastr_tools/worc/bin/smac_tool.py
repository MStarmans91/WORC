#!/usr/bin/env python

# Copyright 2016-2022 Biomedical Imaging Group Rotterdam, Departments of
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

import pandas as pd
import numpy as np
import WORC.addexceptions as WORCexceptions
from WORC.classification.fitandscore import fit_and_score
try:
    from smac.scenario.scenario import Scenario
    from smac.facade.smac_hpo_facade import SMAC4HPO
    from smac.initial_design.random_configuration_design import RandomConfigurations
    from smac.initial_design.sobol_design import SobolDesign
except:
    print("SMAC functionality currently not available. Please see https://worc.readthedocs.io/en/latest/static/additionalfunctionality.html.")
import ast
import os
import json


def main():
    parser = argparse.ArgumentParser(description='Radiomics classification')
    parser.add_argument('-ed', '--ed', metavar='ed',
                        dest='ed', type=str, required=True,
                        help='Estimator data in (HDF)')
    parser.add_argument('-id', '--id', metavar='id',
                        dest='id', type=str, required=True,
                        help='Parallel run data in (HDF)')
    parser.add_argument('-out', '--out', metavar='out',
                        dest='out', type=str, required=True,
                        help='Output: fitted estimator (HDF)')
    args = parser.parse_args()

    # Convert lists into strings
    if type(args.ed) is list:
        args.ed = ''.join(args.ed)
    if type(args.id) is list:
        args.id = ''.join(args.id)
    if type(args.out) is list:
        args.out = ''.join(args.out)

    # Read the data
    data = pd.read_hdf(args.ed)
    run_info = pd.read_hdf(args.id)

    output_filename = os.path.join(run_info['tempfolder'], 'SMAC_output', run_info['run_name'])

    scenario_settings = {'run_obj': 'quality',  # optimize for solution quality
                         'cs': data['search_space'],
                         'deterministic': 'true',
                         'output_dir': output_filename,
                         'shared_model': False,
                         'abort_on_first_run_crash': 'false',
                         }

    # modify the budget of the optimization according to the settings
    if data['budget_type'] == 'evals':
        scenario_settings['runcount-limit'] = data['budget']
    elif data['budget_type'] == 'time':
        scenario_settings['wallclock-limit'] = data['budget']
    else:
        message = f'No valid smac budget_type specified. Should be evals or time, but you gave {data["budget_type"]}.'
        raise WORCexceptions.WORCValueError(message)

    scenario = Scenario(scenario_settings)

    def score_cfg(cfg):
        # Construct a new dictionary with parameters from the input configuration
        if cfg is not None:
            parameters = convert_cfg(cfg.get_dictionary())
        else:
            return float(15000000)

        # Read the data from the smac_tool
        nonlocal data

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
                                return_all=False, use_smac=True)

            # If the run failed because no features were left, report a score of 0
            # (SMAC will use the max integer value internally to steer away from crashing runs)
            if np.isnan(ret[0]['score']):
                ret[0] = 0
                ret[1] = 0
            else:
                ret[0] = ret[0]['score']
                ret[1] = ret[1]['score']
            all_scores.append(ret)

        # Process the results:
        # Return the average score over all cross-validation folds
        df = pd.DataFrame(all_scores, columns=['train_score', 'test_score',
                                             'test_sample_counts', 'fit_time',
                                             'score_time', 'para_estimator', 'para'])

        fname = os.path.join(run_info['tempfolder'],
                             'tested_configs',
                             run_info['run_name'],
                             str(run_info['run_id']) + '.csv')

        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        with open(fname, 'a') as f:
            df.to_csv(f, mode='a', header=f.tell()==0, index=False)

        score = 1 - df['test_score'].mean()  # We minimize so take the inverse

        return score

    # Prepare the SMAC instance
    if data['init_method'] == 'sobol':
        initial_design = SobolDesign  # Apparently does not work for more than 40 dimensions
    elif data['init_method'] == 'random':
        initial_design = RandomConfigurations
    else:
        message = 'No valid smac initialization method specified'
        raise WORCexceptions.WORCValueError(message)

    initial_design_kwargs = {'init_budget': data['init_budget']}

    smac = SMAC4HPO(scenario=scenario, rng=run_info['run_rng'],
                    tae_runner=score_cfg, run_id=run_info['run_id'],
                    initial_design=initial_design,
                    initial_design_kwargs=initial_design_kwargs)

    smac.optimize()

    # Read in the stats from the SMAC output
    stats_file_location = os.path.join(run_info['tempfolder'],
                                       'SMAC_output', run_info['run_name'],
                                       'run_' + str(run_info['run_id']),
                                       'stats.json')

    with open(stats_file_location, 'r') as statsfile:
        smac_stats = json.load(statsfile)

    # Read in the history of the incumbents from the SMAC output
    # and append some trajectory info to the stats
    traj_file_location = os.path.join(run_info['tempfolder'],
                                      'SMAC_output', run_info['run_name'],
                                      'run_' + str(run_info['run_id']),
                                      'traj.json')
    wallclock_times = []
    evaluations = []
    costs = []
    configs = []
    with open(traj_file_location, 'r') as trajfile:
        for line in trajfile:
            incumbent_update_info = ast.literal_eval(line)
            wallclock_times.append(incumbent_update_info['wallclock_time'])
            evaluations.append(incumbent_update_info['evaluations'])
            costs.append(incumbent_update_info['cost'])
            configs.append(incumbent_update_info['incumbent'])

    # Remove the first line, as it does not represent an improvement in the incumbent
    del wallclock_times[0]
    del evaluations[0]
    del costs[0]
    del configs[0]
    smac_stats['inc_wallclock_times'] = wallclock_times
    smac_stats['inc_evaluations'] = evaluations
    smac_stats['inc_costs'] = costs
    smac_stats['inc_configs'] = configs

    result_file_name = os.path.join(run_info['tempfolder'],
                                    'tested_configs',
                                    run_info['run_name'],
                                    'smac_stats_' + str(run_info['run_id']) + '.json')

    if not os.path.exists(os.path.dirname(result_file_name)):
        os.makedirs(os.path.dirname(result_file_name))

    with open(result_file_name, 'a') as file:
        smac_results = {run_info['run_id']: smac_stats}
        json.dump(smac_results, file, indent=4)

    source_labels = ['RET']

    output_df = pd.read_csv(os.path.join(run_info['tempfolder'],
                                         'tested_configs',
                                         run_info['run_name'],
                                         str(run_info['run_id']) + '.csv'))
    output = output_df.values.tolist()

    # Convert strings and floats to dict:
    for ret in output:
        ret[0] = {'score': ret[0]}
        ret[1] = {'score': ret[1]}
        ret[5] = {}
        ret[6] = ast.literal_eval(ret[6])

    source_data =\
        pd.Series([output],
                  index=source_labels,
                  name='SMAC Output')
    source_data.to_hdf(args.out, 'RET')


def convert_cfg(cfg):
    '''If a component is not used at all, pop the main parameter from the cfg.'''
    parameters = cfg

    # fit_and_score requires a flag but only if it is true
    if parameters['Imputation'] == 'False':
        parameters.pop('Imputation')

    if parameters['StatisticalTestUse'] == 'False':
        parameters.pop('StatisticalTestUse')

    if parameters['ReliefUse'] == 'False':
        parameters.pop('ReliefUse')

    if parameters['SelectFromModel'] == 'False':
        parameters.pop('SelectFromModel')

    if parameters['Resampling_Use'] == 'False':
        parameters.pop('Resampling_Use')

    # 'PCAType' is either '95variance' or an int
    if parameters['UsePCA'] == 'True' and \
            parameters['PCAType'] == 'n_components':
        parameters['PCAType'] = parameters.pop('n_components')

    # Add a dummy value for lr_l1_ratio when it is not active;
    # fit_and_score expects it
    if parameters['classifiers'] == 'LR':
        if parameters['LRpenalty'] != 'elasticnet':
            parameters['LR_l1_ratio'] = 0

    return parameters


if __name__ == '__main__':
    main()
