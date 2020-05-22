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

import glob
import os
from WORC.tests import test_helpers as th
from WORC.addexceptions import WORCValueError
from WORC.featureprocessing.ComBat import ComBat, Synthetictest

# TODO: Matlab and Python currently do not give the same results!


def test_combat():
    """Test ComBat feature harmonization."""
    # Check if example data directory exists
    example_data_dir = th.find_exampledatadir()

    # Check if example data required exists
    features = glob.glob(os.path.join(example_data_dir, 'examplefeatures_Patient*.hdf5'))
    if len(features) < 6:
        message = 'Too few example features for ComBat testing not found!' +\
            'Run the create_example_data script from the WORC exampledata ' +\
            'directory!'
        raise WORCValueError(message)
    elif len(features) > 6:
        message = 'Too many example features for ComBat testing not found!' +\
            'Run the create_example_data script from the WORC exampledata ' +\
            'directory!'
        raise WORCValueError(message)

    objectlabels = os.path.join(example_data_dir, 'objectlabels.csv')

    # Python
    config = os.path.join(example_data_dir, 'ComBatConfig_python.ini')
    features_train_out = [f.replace('examplefeatures_', 'examplefeatures_ComBat_python_') for f in features]

    # First run synthetic test
    # Synthetictest()

    # # Run the Combat function: only for training
    ComBat(features_train_in=features,
           labels_train=objectlabels,
           config=config,
           features_train_out=features_train_out)

    # # Run the Combat function: now for train + testing
    # ComBat(features_train_in=features[0:4],
    #        labels_train=objectlabels,
    #        config=config,
    #        features_train_out=features_train_out[0:4],
    #        features_test_in=features[4:],
    #        labels_test=objectlabels,
    #        features_test_out=features_train_out[4:])

    # # Matlab
    # config = os.path.join(example_data_dir, 'ComBatConfig_matlab.ini')
    # features_train_out = [f.replace('examplefeatures_', 'examplefeatures_ComBat_matlab_') for f in features]
    #
    # # # Run the Combat function: only for training
    # ComBat(features_train_in=features,
    #        labels_train=objectlabels,
    #        config=config,
    #        features_train_out=features_train_out)
    #
    # # Run the Combat function: now for train + testing
    # ComBat(features_train_in=features[0:4],
    #        labels_train=objectlabels,
    #        config=config,
    #        features_train_out=features_train_out[0:4],
    #        features_test_in=features[4:],
    #        labels_test=objectlabels,
    #        features_test_out=features_train_out[4:])

    # Remove the feature files
    for i in glob.glob(os.path.join(example_data_dir, '*features_ComBat*.hdf5')):
        os.remove(i)


if __name__ == "__main__":
    test_combat()
