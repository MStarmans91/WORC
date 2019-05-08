#!/usr/bin/env python

# Copyright 2016-2019 Biomedical Imaging Group Rotterdam, Departments of
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

import WORC.addexceptions as WORCexceptions
import fastr
import os


class Slicer(object):
    def __init__(self, images=None, segmentations=None,
                 network=None,
                 fastr_plugin='ProcessPoolExecution',
                 name='Example'):
        '''
        Build a network that evaluates the performance of an estimator.

        Parameters
        ----------

        network: fastr network, default None
                If you input a network, the evaluate network is added
                to the existing network.

        '''
        if network is not None:
            self.network = network
            self.mode = 'WORC'
        else:
            self.mode = 'StandAlone'
            self.fastr_plugin = fastr_plugin
            self.name = 'WORC_Slicer_' + name
            self.network = fastr.Network(id_=self.name)
            self.fastr_tmpdir = os.path.join(fastr.config.mounts['tmp'], self.name)

        if images is None and self.mode == 'StandAlone':
            message = 'Either images and segmentations as input or a WORC' +\
             'network is required for the Evaluate network.'
            raise WORCexceptions.IOError(message)

        self.image = images
        self.segmentations = segmentations

        self.create_network()

    def create_network(self):
        '''
        Add evaluate components to network.
        '''

        # Create all nodes
        self.network.node_slicer =\
            self.network.create_node('Slicer', memory='20G', id_='Slicer')

        # Create sinks
        self.network.sink_PNG =\
            self.network.create_sink('PNGFile', id_='PNG')
        self.network.sink_PNGZoomed =\
            self.network.create_sink('PNGFile', id_='PNGZoomed')

        # Create links to sinks
        self.network.sink_PNG.input = self.network.node_slicer.outputs['out']
        self.network.sink_PNGZoomed.input = self.network.node_slicer.outputs['outzoom']

        # Create sources if not supplied by a WORC network
        if self.mode == 'StandAlone':
            self.network.source_images = self.network.create_source('ITKImage', id_='Images')
            self.network.source_segmentations = self.network.create_source('ITKImage', id_='Segmentations')

        # Create links to sources that are not supplied by a WORC network
        # Not needed in this network

        # Create links to the sources that could be in a WORC network
        if self.mode == 'StandAlone':
            # Sources from the Evaluate network are used
            self.network.node_slicer.inputs['image'] = self.network.source_images.output
            self.network.node_slicer.inputs['segmentation'] = self.network.source_segmentations.output
        else:
            # Sources from the WORC network are used
            print('WIP')

    def set(self, images=None, segmentations=None, sink_data={}):
        '''
        Set the sources and sinks based on the provided attributes.
        '''
        if self.mode == 'StandAlone':
            self.source_data = dict()
            self.sink_data = dict()

            self.source_data['Images'] = images
            self.source_data['Segmentations'] = segmentations

            if 'PNG' not in sink_data.keys():
                self.sink_data['PNG'] = ("vfs://output/{}/Slice_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
            if 'PNGZoomed' not in sink_data.keys():
                self.sink_data['PNGZoomed'] = ("vfs://output/{}/Slice_{{sample_id}}_{{cardinality}}_zoomed{{ext}}").format(self.name)

        else:
            print('[WORC Warning] Slicer set attribute not needed when WORC network is provided!')

    def execute(self):
        """ Execute the network through the fastr.network.execute command. """
        # Draw and execute nwtwork
        self.network.draw_network(self.network.id, draw_dimension=True)
        self.network.execute(self.source_data, self.sink_data, execution_plugin=self.fastr_plugin, tmpdir=self.fastr_tmpdir)
