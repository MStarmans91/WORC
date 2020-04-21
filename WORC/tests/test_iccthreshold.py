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
from WORC.featureprocessing.ICCThreshold import convert_features_ICC_threshold
from random import shuffle


def test_iccthreshold():
    '''
    Test ICC Thresholding statistical testing and computation.
    '''
    # Check if example data directory exists
    example_data_dir = th.find_exampledatadir()

    # Check if example data required exists
    features = glob.glob(os.path.join(example_data_dir, 'examplefeatures_Patient*.hdf5'))
    if len(features) < 6:
        message = 'Too few example features for ICC testing not found!' +\
            'Run the create_example_data script from the WORC exampledata ' +\
            'directory!'
        raise WORCValueError(message)
    elif len(features) > 6:
        message = 'Too many example features for ICC testing not found!' +\
            'Run the create_example_data script from the WORC exampledata ' +\
            'directory!'
        raise WORCValueError(message)

    # Pretend that features are from three observers
    features_multi = list()
    for i in range(0, 3):
        shuffle(features)
        features_multi.append(features[:])

    features_out = [i.replace('examplefeatures_', 'examplefeatures_ICC_') for i in features]
    features_out = [features_out, features_out, features_out]

    # CSV to save ICC values to
    csv_out = os.path.join(example_data_dir, 'ICCValues.csv')

    # Run the ICC threshold function: only for training
    convert_features_ICC_threshold(features_in=features_multi,
                                   csv_out=csv_out,
                                   features_out=features_out)

    # Remove the feature files
    for i in glob.glob(os.path.join(example_data_dir, '*examplefeatures_ICC_*.hdf5')):
        os.remove(i)


if __name__ == "__main__":
    test_iccthreshold()
