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

    network = fastr.Network(id_="elastix_test")

    source1 = network.create_source('ITKImageFile', id_='fixed_img')
    source2 = network.create_source('ITKImageFile', id_='moving_img')
    param1 = network.create_source('ElastixParameterFile', id_='param_file')

    elastix_node = network.create_node('elastix_dev', id_='elastix')
    elastix_node.inputs['fixed_image'] = source1.output
    elastix_node.inputs['moving_image'] = source2.output
    link_param = network.create_link(param1.output, elastix_node.inputs['parameters'])
    link_param.converge = 0

    outtrans = network.create_sink('ElastixTransformFile', id_='sink_trans')
    outtrans.inputs['input'] = elastix_node.outputs['transform']

    transformix_node = network.create_node('transformix_dev', id_='transformix')
    transformix_node.inputs['image'] = source2.output
    transformix_node.inputs['transform'] = elastix_node.outputs['transform'][-1]

    outimage = network.create_sink('ITKImageFile', id_='sink_image')
    outimage.inputs['input'] = transformix_node.outputs['image']

    network.draw_network(img_format='svg')
    network.dumpf('{}.json'.format(network.id), indent=2)

    return network


def source_data(network):
    return {'fixed_img': {'s1': 'vfs://worc_example_data/elastix/img0/slice047.mhd'},
            'moving_img': {'s1': 'vfs://worc_example_data/elastix/img1/slice091.mhd'},
            'param_file': ['vfs://worc_example_data/elastix/parAslice.txt', 'vfs://worc_example_data/elastix/parBslice.txt']}


def sink_data(network):
    return {'sink_trans': 'vfs://tmp/results/{}/elastix_output_trans_{{sample_id}}_{{cardinality}}{{ext}}'.format(network.id),
            'sink_image': 'vfs://tmp/results/{}/elastix_output_image_{{sample_id}}_{{cardinality}}{{ext}}'.format(network.id)}


def main():
    network = create_network()

    # Execute
    # network.draw_network()
    network.execute(source_data(network), sink_data(network), execution_plugin="ProcessPoolExecution")


if __name__ == '__main__':
    main()
