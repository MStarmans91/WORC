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

import pandas as pd
import numpy as np
import os

currentdir = os.path.dirname(os.path.realpath(__file__))


def create_random_features(n_objects=7, n_features=10):
    """
    Create n_objects sets of random features and save in files. Format based
    on PREDICT python package.
    """
    # Create some input values for all objects
    feature_labels = [f'rf_randomlabel_{i}' for i in range(n_features)]
    image_type = 'None'
    parameters = {'Random': 'True'}
    panda_labels = ['image_type', 'parameters', 'feature_values',
                    'feature_labels']

    for i in range(n_objects):
        # Create output name and random feature values and labels
        if i < float(n_objects) / 2.0:
            feature_values = [np.random.normal(loc=5.0, scale=2.0) for i in range(n_features)]
        else:
            feature_values = [np.random.normal(loc=10.0, scale=2.0) for i in range(n_features)]

        output = os.path.join(currentdir, f'examplefeatures_Patient-{str(i).zfill(3)}.hdf5')

        # Convert to pandas Series and save as hdf5
        panda_data = pd.Series([image_type, parameters, feature_values,
                                feature_labels],
                               index=panda_labels,
                               name='Image features'
                               )

        print(f'Saving image features for object {i}.')
        panda_data.to_hdf(output, 'image_features')


if __name__ == "__main__":
    create_random_features()
