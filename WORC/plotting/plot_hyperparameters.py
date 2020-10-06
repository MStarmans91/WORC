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

import os
import pandas as pd
import WORC.addexceptions as ae


def plot_hyperparameters(prediction, label_type=None, estsize=50,
                         output=None, removeconstants=False, verbose=False):
    """Gather which hyperparameters have been used in the best workflows.

    Parameters
    ----------
    prediction: pandas dataframe or string, mandatory
        output of trainclassifier function, either a pandas dataframe
        or a HDF5 file

    estsize: integer, default 50
        Number of estimators that should be taken into account.

    output: filename of csv, default None
        Output file to write to. If None, not output is written, but just
        returned as a variable.

    removeconstants: boolean, default False
        Determine whether to remove any hyperparameters which have the same
        value in all workflows.

    verbose: boolean, default False
        Whether to show print messages or not.

    """
    # Load the prediction file
    if type(prediction) is not pd.core.frame.DataFrame:
        if os.path.isfile(prediction):
            prediction = pd.read_hdf(prediction)
        else:
            raise ae.WORCIOError(f'{prediction} is not an existing file!')

    # Select the estimator from the pandas dataframe to use
    keys = prediction.keys()
    if label_type is None:
        label_type = keys[0]
    prediction = prediction[label_type]

    # Loop over classifiers
    total = len(prediction.classifiers)
    for cnum, cls in enumerate(prediction.classifiers):
        if verbose:
            print(f'Extracting hyperparameters for iteration {cnum + 1} / {total}.')
        # Get parameters and select only a set number
        parameters = cls.cv_results_['params']
        if len(parameters) > estsize:
            parameters = parameters[0:estsize]

        # Add which (cross-validation) iteration is used and the rank
        for i in range(0, estsize):
            parameters[i]['Iteration'] = cnum + 1
            parameters[i]['Rank'] = i + 1

        # Intialize data object if this is the first iteration
        if cnum == 0:
            data = {k: list() for k in parameters[i]}

        # Add to general data object
        for p in parameters:
            for k in p.keys():
                data[k].append(p[k])

    # Optionally, remove any hyperparameters which have the same
    # value in all workflows.
    n_parameters = len(list(data.keys()))
    if removeconstants:
        if verbose:
            print('Removing parameters with constant values.')

        keys = list(data.keys())
        for k in keys:
            # First convert all values to strings so we can use set
            tempdata = [str(i) for i in data[k]]

            # Count unique values, and if only one, delete
            n_unique = len(list(set(tempdata)))
            if n_unique == 1:
                if verbose:
                    print(f'\t Removing parameter {k}.')
                del data[k]

    # Write to csv if output name is provided
    if output is not None:
        if verbose:
            print(f'Writing output to {output}.')

        # First, specify order of columns for easy reading
        columns = list(data.keys())
        columns.remove('Iteration')
        columns.remove('Rank')
        columns = ['Iteration', 'Rank'] + columns

        # Write to dataframe
        df = pd.DataFrame(data)
        df.to_csv(output, index=False, columns=columns)

    # Display some information
    if verbose:
        print(f'Number of hyperparameters: {n_parameters}.')
        n_parameters_unique = len(list(data.keys()))
        print(f'Number of hyperparameters with unique values: {n_parameters_unique}.')

    return data
