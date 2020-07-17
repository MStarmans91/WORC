#!/usr/bin/env python

# Copyright 2020 Biomedical Imaging Group Rotterdam, Departments of
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

import os
import subprocess
import scipy.io as sio
import WORC.IOparser.file_io as wio
import WORC.IOparser.config_io_combat as cio
import numpy as np
import random
import pandas as pd
from WORC.addexceptions import WORCValueError, WORCKeyError
import tempfile
from sys import platform
from WORC.featureprocessing.VarianceThreshold import selfeat_variance
from sklearn.preprocessing import StandardScaler
from neuroCombat import neuroCombat
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from WORC.featureprocessing.Imputer import Imputer


def ComBat(features_train_in, labels_train, config, features_train_out,
           features_test_in=None, labels_test=None, features_test_out=None,
           VarianceThreshold=True, scaler=False, logarithmic=False):
    """
    Apply ComBat feature harmonization.

    Based on: https://github.com/Jfortin1/ComBatHarmonization
    """
    # Load the config
    print('############################################################')
    print('#                    Initializing ComBat.                  #')
    print('############################################################\n')
    config = cio.load_config(config)
    excluded_features = config['ComBat']['excluded_features']

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
    label_data_train['patient_IDs'] = list(label_data_train['patient_IDs'])

    # Exclude features
    if excluded_features:
        print(f'\t Excluding features containing: {excluded_features}')
        # Determine indices of excluded features
        included_feature_indices = []
        excluded_feature_indices = []
        for fnum, i in enumerate(feature_labels):
            if not any(e in i for e in excluded_features):
                included_feature_indices.append(fnum)
            else:
                excluded_feature_indices.append(fnum)

        # Actually exclude the features
        image_features_train_combat = [np.asarray(i)[included_feature_indices].tolist() for i in image_features_train]
        feature_labels_combat = np.asarray(feature_labels)[included_feature_indices].tolist()

        image_features_train_noncombat = [np.asarray(i)[excluded_feature_indices].tolist() for i in image_features_train]
        feature_labels_noncombat = np.asarray(feature_labels)[excluded_feature_indices].tolist()

    else:
        image_features_train_combat = image_features_train
        feature_labels_combat = feature_labels.tolist()

        image_features_train_noncombat = []
        feature_labels_noncombat = []

    # Detect NaNs, otherwise first feature imputation is required
    if any(np.isnan(a) for a in np.asarray(image_features_train_combat).flatten()):
        print('\t [WARNING] NaNs detected, applying median imputation')
        imputer = Imputer(missing_values=np.nan, strategy='median')
        imputer.fit(image_features_train_combat)
        image_features_train_combat = imputer.transform(image_features_train_combat)
    else:
        imputer = None

    # Apply a scaler to the features
    if scaler:
        print('\t Fitting scaler on dataset.')
        scaler = StandardScaler().fit(image_features_train_combat)
        image_features_train_combat = scaler.transform(image_features_train_combat)

    # Remove features with a constant value
    if VarianceThreshold:
        print(f'\t Applying variance threshold on dataset.')
        image_features_train_combat, feature_labels_combat, VarSel =\
            selfeat_variance(image_features_train_combat, np.asarray([feature_labels_combat]))
        feature_labels_combat = feature_labels_combat[0].tolist()

    if features_test_in:
        label_data_test, image_features_test =\
            wio.load_features(features_test_in, patientinfo=labels_test,
                              label_type=label_names)

        image_features_test = [i[0] for i in image_features_test]
        label_data_test['patient_IDs'] = list(label_data_test['patient_IDs'])

        if excluded_features:
            image_features_test_combat = [np.asarray(i)[included_feature_indices].tolist() for i in image_features_test]
            image_features_test_noncombat = [np.asarray(i)[excluded_feature_indices].tolist() for i in image_features_test]
        else:
            image_features_test_combat = image_features_test
            image_features_test_noncombat = []

        # Apply imputation if required
        if imputer is not None:
            image_features_test_combat = imputer.transform(image_features_test_combat)

        # Apply a scaler to the features
        if scaler:
            image_features_test_combat = scaler.transform(image_features_test_combat)

        # Remove features with a constant value
        if VarianceThreshold:
            image_features_test_combat = VarSel.transform(image_features_test_combat)

        all_features = image_features_train_combat.tolist() + image_features_test_combat.tolist()
        all_labels = list()
        for i in range(label_data_train['label'].shape[0]):
            all_labels.append(label_data_train['label'][i, :, 0].tolist() + label_data_test['label'][i, :, 0].tolist())
        all_labels = np.asarray(all_labels)
    else:
        all_features = image_features_train_combat.tolist()
        all_labels = label_data_train['label']

    # Convert data to a single array
    all_features_matrix = np.asarray(all_features)
    all_labels = np.squeeze(all_labels)

    # Apply logarithm if required
    if logarithmic:
        print('\t Taking log10 of features before applying ComBat.')
        all_features_matrix = np.log10(all_features_matrix)

    # Convert all_labels to dictionary
    if len(all_labels.shape) == 1:
        # No mod variables
        all_labels = {label_data_train['label_name'][0]: all_labels}
    else:
        all_labels = {k: v for k, v in zip(label_data_train['label_name'], all_labels)}

    # Split labels in batch and moderation labels
    bat = config['ComBat']['batch']
    mod = config['ComBat']['mod']
    print(f'\t Using batch variable {bat}, mod variables {mod}.')
    batch = [all_labels[l] for l in all_labels.keys() if l in config['ComBat']['batch']]
    batch = batch[0]
    if config['ComBat']['mod'][0] == '[]':
        mod = None
    else:
        mod = [all_labels[l] for l in all_labels.keys() if l in config['ComBat']['mod']]

    # Set parameters for output files
    parameters = {'batch': config['ComBat']['batch'],
                  'mod': config['ComBat']['mod'],
                  'par': config['ComBat']['par']}
    name = 'Image features: ComBat corrected'
    panda_labels = ['parameters',
                    'patient',
                    'feature_values',
                    'feature_labels']
    feature_labels = feature_labels_combat + feature_labels_noncombat

    # Convert all inputs to arrays with right shape
    all_features_matrix = np.transpose(all_features_matrix)
    if mod is not None:
        mod = np.transpose(np.asarray(mod))

    # Patients identified with batch -1.0 should be skipped
    skipname = 'Image features: ComBat skipped'
    ntrain = len(image_features_train_combat)
    ndel = 0
    print(features_test_out)
    for bnum, b in enumerate(batch):
        bnum -= ndel
        if b == -1.0:
            if bnum < ntrain - ndel:
                # Training patient
                print('train')
                pid = label_data_train['patient_IDs'][bnum]
                out = features_train_out[bnum]

                # Combine ComBat and non-ComBat features
                feature_values_temp = list(all_features_matrix[:, bnum]) + list(image_features_train_noncombat[bnum])

                # Delete patient for later processing
                del label_data_train['patient_IDs'][bnum]
                del image_features_train_noncombat[bnum]
                del features_train_out[bnum]
                image_features_train_combat = np.delete(image_features_train_combat, bnum, 0)

            else:
                # Test patient
                print('test')
                pid = label_data_test['patient_IDs'][bnum - ntrain]
                out = features_test_out[bnum - ntrain]

                # Combine ComBat and non-ComBat features
                feature_values_temp = list(all_features_matrix[:, bnum]) + list(image_features_test_noncombat[bnum - ntrain])

                # Delete patient for later processing
                del label_data_test['patient_IDs'][bnum - ntrain]
                del image_features_test_noncombat[bnum - ntrain]
                del features_test_out[bnum - ntrain]
                image_features_test_combat = np.delete(image_features_test_combat, bnum - ntrain, 0)

            # Delete some other variables for later processing
            all_features_matrix = np.delete(all_features_matrix, bnum, 1)
            if mod is not None:
                mod = np.delete(mod, bnum, 0)
            batch = np.delete(batch, bnum, 0)

            # Notify user
            print(f'[WARNING] Skipping patient {pid} as batch variable is -1.0.')

            # Sort based on feature label
            feature_labels_temp, feature_values_temp =\
                zip(*sorted(zip(feature_labels, feature_values_temp)))

            # Convert to pandas Series and save as hdf5
            panda_data = pd.Series([parameters, pid, feature_values_temp,
                                    feature_labels_temp],
                                   index=panda_labels,
                                   name=skipname
                                   )

            print(f'\t Saving image features to: {out}.')
            panda_data.to_hdf(out, 'image_features')

            ndel += 1

    print(features_test_out)
    # Run ComBat in Matlab
    if config['ComBat']['language'] == 'matlab':
        print('\t Executing ComBat through Matlab')
        data_harmonized = ComBatMatlab(dat=all_features_matrix,
                                       batch=batch,
                                       command=config['ComBat']['matlab'],
                                       mod=mod,
                                       par=config['ComBat']['par'],
                                       per_feature=config['ComBat']['per_feature'])

    elif config['ComBat']['language'] == 'python':
        print('\t Executing ComBat through neuroComBat in Python')
        data_harmonized = ComBatPython(dat=all_features_matrix,
                                       batch=batch,
                                       mod=mod,
                                       eb=config['ComBat']['eb'],
                                       par=config['ComBat']['par'],
                                       per_feature=config['ComBat']['per_feature'])
    else:
        raise WORCKeyError(f"Language {config['ComBat']['language']} unknown.")

    # Convert values back if logarithm was used
    if logarithmic:
        data_harmonized = 10 ** data_harmonized

    # Convert again to train hdf5 files
    feature_values_train_combat = [data_harmonized[:, i] for i in range(len(image_features_train_combat))]
    for fnum, i_feat in enumerate(feature_values_train_combat):
        # Combine ComBat and non-ComBat features
        feature_values_temp = i_feat.tolist() + image_features_train_noncombat[fnum]

        # Sort based on feature label
        feature_labels_temp, feature_values_temp =\
            zip(*sorted(zip(feature_labels, feature_values_temp)))

        # Convert to pandas Series and save as hdf5
        pid = label_data_train['patient_IDs'][fnum]
        panda_data = pd.Series([parameters, pid, feature_values_temp,
                                feature_labels_temp],
                               index=panda_labels,
                               name=name
                               )

        print(f'Saving image features to: {features_train_out[fnum]}.')
        panda_data.to_hdf(features_train_out[fnum], 'image_features')

    # Repeat for testing if required
    if features_test_in:
        print(len(image_features_test_combat))
        print(data_harmonized.shape[1])
        feature_values_test_combat = [data_harmonized[:, i] for i in range(data_harmonized.shape[1] - len(image_features_test_combat), data_harmonized.shape[1])]
        for fnum, i_feat in enumerate(feature_values_test_combat):
            print(fnum)
            # Combine ComBat and non-ComBat features
            feature_values_temp = i_feat.tolist() + image_features_test_noncombat[fnum]

            # Sort based on feature label
            feature_labels_temp, feature_values_temp =\
                zip(*sorted(zip(feature_labels, feature_values_temp)))

            # Convert to pandas Series and save as hdf5
            pid = label_data_test['patient_IDs'][fnum]
            panda_data = pd.Series([parameters, pid, feature_values_temp,
                                    feature_labels_temp],
                                   index=panda_labels,
                                   name=name
                                   )

            print(f'Saving image features to: {features_test_out[fnum]}.')
            panda_data.to_hdf(features_test_out[fnum], 'image_features')


def ComBatPython(dat, batch, mod=None, par=1,
                 eb=1, per_feature=False, plotting=False):
    """
    Run the ComBat Function python script.

    par = 0 is non-parametric.
    """
    # convert inputs to neuroCombat format.
    covars = dict()
    categorical_cols = list()
    covars['batch'] = batch
    if mod is not None:
        for i_mod in range(mod.shape[1]):
            label = f'mod_{i_mod}'
            covars[label] = [m for m in mod[:, i_mod]]
            categorical_cols.append(label)

    covars = pd.DataFrame(covars)
    batch_col = 'batch'
    if par == 0:
        parametric = False
    elif par == 1:
        parametric = True
    else:
        raise WORCValueError(f'Par should be 0 or 1, now {par}.')

    if eb == 0:
        eb = False
    elif eb == 1:
        eb = True
    else:
        raise WORCValueError(f'eb should be 0 or 1, now {eb}.')

    if per_feature == 0:
        per_feature = False
    elif per_feature == 1:
        per_feature = True
    else:
        raise WORCValueError(f'per_feature should be 0 or 1, now {per_feature}.')

    # execute ComBat
    if not per_feature:
        data_harmonized = neuroCombat(dat=dat, covars=covars, batch_col=batch_col,
                                      categorical_cols=categorical_cols,
                                      eb=eb, parametric=parametric)
    elif per_feature:
        print('\t Executing ComBat per feature.')
        data_harmonized = np.zeros(dat.shape)
        # Shape: (features, samples)
        for i in range(dat.shape[0]):
            if eb:
                # Copy feature + random noise
                random_feature = np.random.rand(dat[i, :].shape[0])
                feat_temp = np.asarray([dat[i, :], dat[i, :] + random_feature])
            else:
                # Just use the single feature
                feat_temp = np.asarray([dat[i, :]])

            feat_temp = neuroCombat(dat=feat_temp, covars=covars,
                                    batch_col=batch_col,
                                    categorical_cols=categorical_cols,
                                    eb=eb, parametric=parametric)
            data_harmonized[i, :] = feat_temp[0, :]

            if plotting:
                feat1 = dat[i, :]
                feat1_harm = data_harmonized[i, :]
                print(len(feat1))

                feat1_b1 = [f for f, b in zip(feat1, batch[0]) if b == 1.0]
                feat1_b2 = [f for f, b in zip(feat1, batch[0]) if b == 2.0]
                print(len(feat1_b1))
                print(len(feat1_b2))

                feat1_harm_b1 = [f for f, b in zip(feat1_harm, batch[0]) if b == 1.0]
                feat1_harm_b2 = [f for f, b in zip(feat1_harm, batch[0]) if b == 2.0]

                plt.figure()
                ax = plt.subplot(2, 1, 1)
                ax.scatter(np.ones((len(feat1_b1))), feat1_b1, color='red')
                ax.scatter(np.ones((len(feat1_b2))) + 1, feat1_b2, color='blue')
                plt.title('Before Combat')

                ax = plt.subplot(2, 1, 2)
                ax.scatter(np.ones((len(feat1_b1))), feat1_harm_b1, color='red')
                ax.scatter(np.ones((len(feat1_b2))) + 1, feat1_harm_b2, color='blue')
                plt.title('After Combat')

                plt.show()

    else:
        raise WORCValueError(f'per_feature should be False or True, now {per_feature}.')

    return data_harmonized


def Synthetictest(n_patients=50, n_features=10, par=1, eb=1,
                  per_feature=False, difscale=False, logarithmic=False,
                  oddpatient=True, oddfeat=True, samefeat=True):
    """Test for ComBat with Synthetic data."""
    features = np.zeros((n_features, n_patients))
    batch = list()

    # First batch: Gaussian with loc 0, scale 1
    for i in range(0, int(n_patients/2)):
        feat_temp = [np.random.normal(loc=0.0, scale=1.0) for i in range(n_features)]
        if i == 1 and oddpatient:
            feat_temp = [np.random.normal(loc=10.0, scale=1.0) for i in range(n_features)]
        elif oddfeat:
            feat_temp = [np.random.normal(loc=0.0, scale=1.0) for i in range(n_features - 1)] + [np.random.normal(loc=10000.0, scale=1.0)]

        if samefeat:
            feat_temp[-1] = 1

        features[:, i] = feat_temp
        batch.append(1)

    # Get directions for features
    directions = list()
    for i in range(n_features):
        direction = random.random()
        if direction > 0.5:
            directions.append(1.0)
        else:
            directions.append(-1.0)

    # First batch: Gaussian with loc 5, scale 1
    for i in range(int(n_patients/2), n_patients):
        feat_temp = [np.random.normal(loc=direction*5.0, scale=1.0) for i in range(n_features)]
        if oddfeat:
            feat_temp = [np.random.normal(loc=5.0, scale=1.0) for i in range(n_features - 1)] + [np.random.normal(loc=10000.0, scale=1.0)]

        if difscale:
            feat_temp = [f + 1000 for f in feat_temp]

        feat_temp = np.multiply(feat_temp, directions)
        if samefeat:
            feat_temp[-1] = 1

        features[:, i] = feat_temp
        batch.append(2)

    # Create mod var
    mod = [[np.random.randint(30, 100) for i in range(n_patients)]]

    # Apply ComBat
    batch = np.asarray([batch])
    mod = np.transpose(np.asarray(mod))
    if logarithmic:
        minfeat = np.min(features)
        features = np.log10(features + np.abs(minfeat) + 1E-100)

    data_harmonized = ComBatPython(dat=features, batch=batch, mod=mod, par=par,
                                   eb=eb, per_feature=per_feature)

    if logarithmic:
        data_harmonized = 10 ** data_harmonized - np.abs(minfeat)

    for i in range(n_features):
        f = plt.figure()
        ax = plt.subplot(2, 1, 1)
        ax.scatter(np.ones((int(n_patients/2))), features[i, 0:int(n_patients/2)], color='red')
        ax.scatter(np.ones((n_patients - int(n_patients/2))) + 1, features[i, int(n_patients/2):], color='blue')
        plt.title('Before Combat')

        ax = plt.subplot(2, 1, 2)
        ax.scatter(np.ones((int(n_patients/2))), data_harmonized[i, 0:int(n_patients/2)], color='red')
        ax.scatter(np.ones((n_patients - int(n_patients/2))) + 1, data_harmonized[i, int(n_patients/2):], color='blue')
        plt.title('After Combat')

        plt.show()
        f.savefig(f'combat_par{par}_eb{eb}_perfeat{per_feature}_feat{i}.png')

    # Logarithmic: not useful, as we have negative numbers, and (almost) zeros.
    # so combat gives unuseful results.
    # Same feature twice with eb and par: nans


def ComBatMatlab(dat, batch, command, mod=None, par=1, per_feature='true'):
    """
    Run the ComBat Function Matlab script.

    par = 0 is non-parametric.
    """
    # Mod: default argument is empty list
    if mod is None:
        mod = []

    # TODO: Add check whether matlab executable is found

    # Save the features in a .mat MatLab Compatible format
    # NOTE: Should change this_folder to a proper temporary directory
    this_folder = os.path.dirname(os.path.realpath(__file__))
    tempdir = tempfile.gettempdir()
    tempfile_in = os.path.join(tempdir, 'combat_input.mat')
    tempfile_out = os.path.join(tempdir, 'combat_output.mat')
    ComBatFolder = os.path.join(os.path.dirname(this_folder),
                                'external',
                                'ComBatHarmonization',
                                'Matlab',
                                'scripts')

    dict = {'output': tempfile_out,
            'ComBatFolder': ComBatFolder,
            'datvar': dat,
            'batchvar': batch,
            'modvar': mod,
            'parvar': par,
            'per_feature': per_feature
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

    # BIGR Cluster: /cm/shared/apps/matlab/R2015b/bin/matlab
    regcommand = ('cd "' + this_folder + '"' + commandseparator +
                  '"' + command + '" -nodesktop -nosplash -nojvm -r "combatmatlab(' + "'" + str(tempfile_in) + "'" + ')"' +
                  commandseparator +
                  'cd "' + currentdir + '"')
    print(f'Executing ComBat in Matlab through command: {regcommand}.')
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
            except (sio.matlab.miobase.MatReadError, ValueError):
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
