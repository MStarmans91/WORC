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
import os
import subprocess
import scipy.io as sio
import WORC.IOparser.file_io as wio
import WORC.IOparser.config_io_combat as cio
import numpy as np
import pandas as pd
from WORC.addexceptions import WORCValueError
import tempfile
from sys import platform


def main():
    parser = argparse.ArgumentParser(description='ComBat Feature Harmonization')
    parser.add_argument('-feat_train', '--feat_train', metavar='features_train',
                        nargs='+', dest='feat_train', type=str, required=True,
                        help='Patient features input (HDF)')
    parser.add_argument('-feat_test', '--feat_test', metavar='features_test',
                        nargs='+', dest='feat_test', type=str, required=False,
                        default=None,
                        help='Patient features input (HDF)')
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
    parser.add_argument('-feat_train_out', '--feat_train_out', metavar='features_train_out',
                        nargs='+', dest='feat_train_out', type=str, required=True,
                        help='Patient features output (HDF)')
    parser.add_argument('-feat_test_out', '--feat_test_out', metavar='features_test_out',
                        nargs='+', dest='feat_test_out', type=str, required=False,
                        default=None,
                        help='Patient features output (HDF)')
    args = parser.parse_args()

    # Convert input arguments that should be strings
    if type(args.pc_train) is list:
        args.pc_train = ''.join(args.pc_train)

    if type(args.pc_test) is list:
        args.pc_test = ''.join(args.pc_test)

    if type(args.cf) is list:
        args.cf = ''.join(args.cf)

    ComBat(features_train_in=args.feat_train,
           labels_train=args.pc_train,
           config=args.cf,
           features_train_out=args.feat_train_out,
           features_test_in=args.feat_test,
           labels_test=args.pc_test,
           features_test_out=args.feat_test_out)


def ComBat(features_train_in, labels_train, config, features_train_out,
           features_test_in=None, labels_test=None, features_test_out=None):
    '''
    Apply ComBat feature harmonization. Based on: https://github.com/Jfortin1/ComBatHarmonization
    '''
    # Load the config
    config = cio.load_config(config)

    # If mod, than also load moderating labels
    if config['ComBat']['mod'][0] == '[]':
        label_names = config['ComBat']['batch']
    else:
        label_names = config['ComBat']['batch'] + config['ComBat']['mod']

    # Load the features for both training and testing, match with batch and mod parameters
    label_data_train, image_features_train =\
        wio.load_features(features_train_in, patientinfo=labels_train,
                          label_type=label_names)

    feature_labels = image_features_train[0][1]
    image_features_train = [i[0] for i in image_features_train]

    if features_test_in:
        label_data_test, image_features_test =\
            wio.load_features(features_test_in, patientinfo=labels_test,
                              label_type=label_names)

        image_features_test = [i[0] for i in image_features_test]
        all_features = image_features_train + image_features_test
        all_labels = list()
        for i in range(label_data_train['label'].shape[0]):
            all_labels.append(label_data_train['label'][i, :, 0].tolist() + label_data_test['label'][i, :, 0].tolist())
        all_labels = np.asarray(all_labels)
    else:
        all_features = image_features_train
        all_labels = label_data_train['label']

    # Convert data to a single array
    all_features_matrix = np.asarray(all_features)

    # Convert all_labels to dictionary
    all_labels = np.squeeze(all_labels)
    all_labels = {k: v for k, v in zip(label_data_train['label_name'], all_labels)}

    # Split labels in batch and moderation labels
    batch = [all_labels[l] for l in all_labels.keys() if l in config['ComBat']['batch']]
    if config['ComBat']['mod'][0] == '[]':
        mod = []
    else:
        mod = [all_labels[l] for l in all_labels.keys() if l in config['ComBat']['mod']]

    # Convert all inputs to arrays with right shape
    all_features_matrix = np.transpose(all_features_matrix)
    mod = np.transpose(np.asarray(mod))

    # Run ComBatin Matlab
    data_harmonized = ComBatMatlab(dat=all_features_matrix,
                                   batch=batch,
                                   mod=mod,
                                   par=config['ComBat']['par'])

    # Convert again to train hdf5 files
    parameters = {'batch': config['ComBat']['batch'],
                  'mod': config['ComBat']['mod'],
                  'par': config['ComBat']['par']}
    name = 'Image features: ComBat corrected'
    panda_labels = ['parameters',
                    'feature_values',
                    'feature_labels']

    feature_values_train_combat = [data_harmonized[i] for i in range(len(image_features_train))]
    for fnum, i_feat in enumerate(feature_values_train_combat):
        # Convert to pandas Series and save as hdf5
        panda_data = pd.Series([parameters, i_feat,
                                feature_labels],
                               index=panda_labels,
                               name=name
                               )

        print(f'Saving image features to: {features_train_out[fnum]}.')
        panda_data.to_hdf(features_train_out[fnum], 'image_features')

    # Repeat for testing if required
    if features_test_in:
        feature_values_test_combat = [d for d in data_harmonized[len(image_features_train):]]
        for fnum, i_feat in enumerate(feature_values_test_combat):
            # Convert to pandas Series and save as hdf5
            panda_data = pd.Series([parameters, i_feat,
                                    feature_labels],
                                   index=panda_labels,
                                   name=name
                                   )

            print(f'Saving image features to: {features_test_out[fnum]}.')
            panda_data.to_hdf(features_test_out[fnum], 'image_features')


def ComBatMatlab(dat, batch, mod=None, par=0):
    '''
    Run the ComBat Function Matlab script.

    par = 0 is non-parametric.
    '''

    # Mod: default argument is empty list
    if mod is None:
        mod = []

    # Save the features in a .mat MatLab Compatible format
    # NOTE: Should change this_folder to a proper temporary directory
    this_folder = os.path.dirname(os.path.realpath(__file__))
    tempdir = tempfile.gettempdir()
    tempfile_in = os.path.join(tempdir, 'combat_input.mat')
    tempfile_out = os.path.join(tempdir, 'combat_output.mat')
    ComBatFolder = os.path.join(os.path.dirname(this_folder),
                                'external',
                                'ComBat',
                                'Matlab',
                                'scripts')

    dict = {'output': tempfile_out,
            'ComBatFolder': ComBatFolder,
            'datvar': dat,
            'batchvar': batch,
            'modvar': mod,
            'parvar': par
            }

    sio.savemat(tempfile_in, dict)

    # Make sure there is no tempfile out from the previous run
    if os.path.exists(tempfile_out):
        os.remove(tempfile_out)

    # Run ComBat
    currentdir = os.getcwd()
    if platform == "linux" or platform == "linux2":
        commandseparator = ' ; '
    elif platform == "win32":
        commandseparator = ' & '

    regcommand = ('cd ' + this_folder + commandseparator +
                  '"C:\\Program Files\\MATLAB\\R2015b\\bin\\matlab.exe" -nodesktop -nosplash -nojvm -r "combatmatlab(' + "'" + str(tempfile_in) + "'" + ')"' +
                  commandseparator +
                  'cd ' + currentdir)
    proc = subprocess.Popen(regcommand,
                            shell=True,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            )
    proc.wait()
    stdout_value, stderr_value = proc.communicate()

    # BUG: Waiting does not work, just wait for output to arrive, either with
    # the actual output or an error message
    succes = False
    while succes is False:
        if os.path.exists(tempfile_out):
            try:
                mat_dict = sio.loadmat(tempfile_out)
                try:
                    data_harmonized = mat_dict['data_harmonized']
                    succes = True
                except KeyError:
                    try:
                        message = mat_dict['message']
                        raise WORCValueError(f'Error in Matlab ComBat execution: {message}.')
                    except KeyError:
                        pass
            except sio.matlab.miobase.MatReadError:
                pass

    # Check if expected output file exists
    if not os.path.exists(tempfile_out):
        raise WORCValueError(f'Error in Matlab ComBat execution: command: {regcommand}, stdout: {stdout_value}, stderr: {stderr_value}')

    # Read the output from ComBat
    mat_dict = sio.loadmat(tempfile_out)
    data_harmonized = mat_dict['data_harmonized']
    data_harmonized = np.transpose(data_harmonized)

    # Remove temporary files
    os.remove(tempfile_out)
    os.remove(tempfile_in)

    return data_harmonized


if __name__ == '__main__':
    main()
