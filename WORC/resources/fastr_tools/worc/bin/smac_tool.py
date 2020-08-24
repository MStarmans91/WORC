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
import pandas as pd
import numpy as np
import WORC.addexceptions as WORCexceptions
from WORC.classification.fitandscore import fit_and_score
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.initial_design.initial_design import InitialDesign
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.initial_design.sobol_design import SobolDesign
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
    if type(args.id) is list:
        args.id = ''.join(args.id)
    #if type(args.tt) is list:
    #    args.tt = ''.join(args.tt)
    #if type(args.para) is list:
    #    args.para = ''.join(args.para)
    if type(args.out) is list:
        args.out = ''.join(args.out)

    # Read the data
    data = pd.read_hdf(args.ed)
    run_info = pd.read_hdf(args.id)
    #traintest = pd.read_hdf(args.tt)
    #with open(args.para, 'rb') as fp:
    #    para = json.load(fp)

    #init_design = InitialDesign(cs=data['search_space'],
    #                            init_budget=0.1*data['n_iter'],
    #                            rng=run_info['run_rng'],
    #                            traj_logger=None,
    #                            ta_run_limit=100)

    scenario_settings = {'run_obj': 'quality',  # optimize for solution quality
                         'cs': data['search_space'],
                         'deterministic': 'true',
                         'output_dir': '/scratch/mdeen/SMAC_output/' + run_info['run_name'],
                         'shared_model': False,
                         'input_psmac_dirs': '/scratch/mdeen/SMAC_output/' + run_info['run_name'],
                         'abort_on_first_run_crash': 'false',
                         }

    # modify the budget of the optimization according to the settings
    if data['budget_type'] == 'evals':
        scenario_settings['runcount-limit'] = data['budget']
    elif data['budget_type'] == 'time':
        scenario_settings['wallclock-limit'] = data['budget']
    else:
        message = 'No valid smac budget_type specified'
        raise WORCexceptions.WORCValueError(message)

    scenario = Scenario(scenario_settings)

    # "runcount-limit": data['n_iter'],  # max. number of function evaluations;
    # "wallclock-limit": data['n_iter'],

    def score_cfg(cfg):
        # Construct a new dictionary with parameters from the input configuration

        # ! THIS POTENTIALLY HAS A BUG BUT PROBABLY NOT ACTUALLY ! #
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
        fname = '/scratch/mdeen/tested_configs/' + run_info['run_name'] + '/' + \
                str(run_info['run_id']) + '.csv'
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        with open(fname, 'a') as f:
            df.to_csv(f, mode='a', header=f.tell()==0, index=False)

        score = 1 - df['test_score'].mean()  # We minimize so take the inverse

        return score

    # Prepare the SMAC instance
    if data['init_method'] == 'sobol':
        initial_design = SobolDesign
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
    opt_config = smac.optimize()

    # Read in the stats from the SMAC output
    stats_file_location = '/scratch/mdeen/SMAC_output/' + run_info['run_name'] + \
                          '/run_' + str(run_info['run_id']) + '/stats.json'
    with open(stats_file_location, 'r') as statsfile:
        smac_stats = json.load(statsfile)

    # Read in the history of the incumbents from the SMAC output
    # and append some trajectory info to the stats
    traj_file_location = '/scratch/mdeen/SMAC_output/' + run_info['run_name'] + \
                         '/run_' + str(run_info['run_id']) + '/traj.json'
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

    # Update the result file of the optimization
    # ! This is inaccurate for multiple instances
    result_file = data['smac_result_file']

    result_file_name = '/scratch/mdeen/tested_configs/' + run_info['run_name'] +\
                       '/smac_stats_' + str(run_info['run_id']) + '.json'
    if not os.path.exists(os.path.dirname(result_file_name)):
        os.makedirs(os.path.dirname(result_file_name))
    with open(result_file_name, 'a') as file:
        smac_results = {run_info['run_id']: smac_stats}
        json.dump(smac_results, file, indent=4)

    '''
    if os.path.exists(result_file):
        with open(result_file, 'r') as jsonfile:
            smac_results = json.load(jsonfile)
        if smac_results.has_key(run_info['run_name']):
            smac_results[run_info['run_name']][run_info['run_id']] = smac_stats
        else:
            smac_results[run_info['run_name']] = {run_info['run_id']: smac_stats}
        with open(result_file, 'w') as jsonfile:
            json.dump(smac_results, jsonfile, indent=4)
    else:
        with open(result_file, 'a') as jsonfile:
            smac_results = {run_info['run_name']: {run_info['run_id']: smac_stats}}
            json.dump(smac_results, jsonfile, indent=4)

    
    if os.path.exists(result_file):
        with open(result_file, 'r') as jsonfile:
            smac_results = json.load(jsonfile)
        cv_iteration = len(smac_results)
        last_instance_name = smac_results['cv-' + str(cv_iteration-1)]['0']['instance_name']
        with open('/scratch/mdeen/debug.txt', 'a') as debugfile:
            debugfile.write('last instance name: ' + last_instance_name)
            debugfile.write('run_id: ' + run_info['run_id'])
        if last_instance_name == run_info['run_name']:

            # This means we are still in the same outer cv
            smac_results['cv-' + str(cv_iteration-1)][run_info['run_id']] = smac_stats
        else:
            smac_results['cv-' + str(cv_iteration)][run_info['run_id']] = smac_stats
        with open(result_file, 'w') as jsonfile:
            json.dump(smac_results, jsonfile, indent=4)
    else:
        with open(result_file, 'a') as jsonfile:
            with open('/scratch/mdeen/debug.txt', 'a') as debugfile:
                debugfile.write('outer else statement called')
                debugfile.write('run_id: ' + run_info['run_id'])
            smac_results = {'cv-0': {run_info['run_id']: smac_stats}}
            json.dump(smac_results, jsonfile, indent=4)
    '''

    source_labels = ['RET']

    output_df = pd.read_csv('/scratch/mdeen/tested_configs/' + run_info['run_name'] + '/' +
                            str(run_info['run_id']) + '.csv')
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
    parameters = cfg

    # fit_and_score requires a flag but only if it is true
    if parameters['Imputation'] == 'False':
        parameters.pop('Imputation')
    # Delete four more flags from the config if they are false
    if parameters['StatisticalTestUse'] == 'False':
        parameters.pop('StatisticalTestUse')
    #if parameters['SampleProcessing_SMOTE'] == 'False':
    #    parameters.pop('SampleProcessing_SMOTE')
    #if parameters['SampleProcessing_Oversampling'] == 'False':
    #    parameters.pop('SampleProcessing_Oversampling')
    if parameters['ReliefUse'] == 'False':
        parameters.pop('ReliefUse')
    # 'PCAType' is either '95variance' or an int
    if parameters['UsePCA'] == 'True' and \
            parameters['PCAType'] == 'n_components':
        parameters['PCAType'] = parameters.pop('n_components')

    return parameters


if __name__ == '__main__':
    main()
