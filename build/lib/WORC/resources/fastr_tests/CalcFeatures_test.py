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
    network = fastr.Network(id_='CalcFeatures_test')

    # Create a source node in the network
    source_segmentation = network.create_source('ITKImageFile', id_='segmentation')
    source_image = network.create_source('ITKImageFile', id_='image')
    source_metadata = network.create_source('DicomImageFile', id_='metadata')
    source_parameters = network.create_source('ParameterFile', id_='parameters')

    # Create a new node in the network using toollist
    node_calfeatures = network.create_node('CalcFeatures', id_="calcfeatures")

    # Create a link between the source output and an input of the addint node
    node_calfeatures.inputs['segmentation'] = source_segmentation.output
    node_calfeatures.inputs['image'] = source_image.output
    node_calfeatures.inputs['metadata'] = source_metadata.output
    node_calfeatures.inputs['parameters'] = source_parameters.output

    # Create a sink to save the data
    sink_features = network.create_sink('HDF5', id_='features')

    # Link the addint node to the sink
    sink_features.input = node_calfeatures.outputs['features']

    return network


def source_data(network):
    return {'segmentation': 'vfs://worc_example_data/CLM/seg_tumor_c.nii.gz',
            'image':  'vfs://worc_example_data/CLM/image.nii.gz',
            'metadata':  'vfs://worc_example_data/CLM/00000.dcm',
            'parameters': 'vfs://worc_example_data/CLM/parameters.ini'}


def sink_data(network):
    return {'features': 'vfs://tmp/results/{}/calcfeatures_{{sample_id}}_{{cardinality}}{{ext}}'.format(network.id)}


def main():
    network = create_network()

    # Execute
    # network.draw_network()
    network.execute(source_data(network), sink_data(network), execution_plugin="ProcessPoolExecution")


if __name__ == '__main__':
    main()
