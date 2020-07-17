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
import fastr
from WORC.featureprocessing.ComBat import ComBat, Synthetictest

# TODO: Matlab and Python currently do not give the same results!


def test_combat():
    """Test ComBat feature harmonization."""
    # Check if example data directory exists
    example_data_dir = th.find_exampledatadir()

    # Check if example data required exists
    features = glob.glob(os.path.join(example_data_dir, 'examplefeatures_Patient*.hdf5'))
    if len(features) < 7:
        message = 'Too few example features for ComBat testing not found! ' +\
            'Run the create_example_data script from the WORC exampledata ' +\
            'directory!'
        raise WORCValueError(message)
    elif len(features) > 7:
        message = 'Too many example features for ComBat testing not found! ' +\
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
    # ComBat(features_train_in=features,
    #        labels_train=objectlabels,
    #        config=config,
    #        features_train_out=features_train_out)

    # # Run the Combat function: now for train + testing
    ComBat(features_train_in=features[0:4],
           labels_train=objectlabels,
           config=config,
           features_train_out=features_train_out[0:4],
           features_test_in=features[4:],
           labels_test=objectlabels,
           features_test_out=features_train_out[4:])

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
    # for i in glob.glob(os.path.join(example_data_dir, '*features_ComBat*.hdf5')):
    #     os.remove(i)


def test_combat_fastr():
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

    # Create the fastr network
    experiment = fastr.create_network('test_ComBat')

    source_features = experiment.create_source('HDF5', id='features_in', node_group='features')
    source_labels = experiment.create_source('PatientInfoFile', id='labels', node_group='pctrain')
    source_config = experiment.create_source('ParameterFile', id='config', node_group='conf')

    sink_features = experiment.create_sink('HDF5', id='features_out')

    node_combat = experiment.create_node('combat/ComBat:1.0',
                                         tool_version='1.0',
                                         id='ComBat',)

    link_combat_1 = experiment.create_link(source_config.output, node_combat.inputs['config'])
    link_combat_2 = experiment.create_link(source_labels.output, node_combat.inputs['patientclass_train'])
    link_combat_1.collapse = 'conf'
    link_combat_2.collapse = 'pctrain'

    # Mimic using two feature toolboxes
    links_Combat1_train = node_combat.inputs['features_train']['MR_0'] << source_features.output
    links_Combat1_train.collapse = 'features'

    links_Combat2_train = node_combat.inputs['features_train']['MR_1'] << source_features.output
    links_Combat2_train.collapse = 'features'

    links_Combat_out_train = sink_features.input << node_combat.outputs['features_train_out']
    links_Combat_out_train.collapse = 'ComBat'

    # Provide source and sink data
    source_data = dict()
    source_data['features_in'] = features
    source_data['labels'] = objectlabels
    source_data['config'] = config

    sink_data = dict()
    sink_data['features_out'] = "vfs://output/test_ComBat/ComBat/features_ComBat_{{sample_id}}_{{cardinality}}{{ext}}"

    # Execute
    experiment.execute(source_data, sink_data, execution_plugin='LinearExecution')

    # Remove the feature files
    for i in glob.glob(os.path.join(example_data_dir, '*features_ComBat*.hdf5')):
        os.remove(i)


if __name__ == "__main__":
    test_combat()
