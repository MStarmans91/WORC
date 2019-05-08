#!/usr/bin/env python

# Copyright 2017-2018 Biomedical Imaging Group Rotterdam, Departments of
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

IS_TEST = True


def create_network():
    # Import the faster environment and set it up
    import fastr
    # Create a new network
    network = fastr.Network(id_='Segmentix_test')

    # Create a source node in the network
    source_segmentation = network.create_source('ITKImageFile', id_='segmentation_in')
    source_mask = network.create_source('ITKImageFile', id_='mask')
    source_parameters = network.create_source('ParameterFile', id_='parameters')

    # Create a new node in the network using toollist
    node_segmentix = network.create_node('Segmentix', id_="segmentix")

    # Create a link between the source output and an input of the addint node
    node_segmentix.inputs['segmentation_in'] = source_segmentation.output
    node_segmentix.inputs['mask'] = source_mask.output
    node_segmentix.inputs['parameters'] = source_parameters.output

    # Create a sink to save the data
    sink_segmentation = network.create_sink('ITKImageFile', id_='segmentation_out')

    # Link the addint node to the sink
    sink_segmentation.input = node_segmentix.outputs['segmentation_out']

    return network


def source_data(network):
    return {'segmentation_in': 'vfs://worc_example_data/CLM/seg_liver.nii.gz',
            'mask':  'vfs://worc_example_data/CLM/seg_tumor.nii.gz',
            'parameters': 'vfs://worc_example_data/CLM/parameters.ini'}


def sink_data(network):
    return {'segmentation_out': 'vfs://tmp/results/{}/segmentix_{{sample_id}}_{{cardinality}}{{ext}}'.format(network.id)}


def main():
    network = create_network()

    # Execute
    # network.draw_network()
    network.execute(source_data(network), sink_data(network), execution_plugin="ProcessPoolExecution")


if __name__ == '__main__':
    main()
