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

# NOTE: Very important to give images and segmentations as dict with patient names!


class Evaluate(object):
    def __init__(self, label_type, ensemble=50, scores='percentages',
                 network=None, features=None,
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
            self.name = 'WORC_Evaluate_' + name
            self.network = fastr.Network(id_=self.name)
            self.fastr_tmpdir = os.path.join(fastr.config.mounts['tmp'], self.name)

        if features is None and self.mode == 'StandAlone':
            raise WORCexceptions.IOError('Either features as input or a WORC network is required for the Evaluate network.')

        self.features = features

        self.label_type = label_type
        self.ensemble = ensemble

        self.create_network()

    def create_network(self):
        '''
        Add evaluate components to network.
        '''

        # Create all nodes
        self.network.node_ROC =\
            self.network.create_node('PlotROC', memory='20G', id_='plot_ROC')
        self.network.node_SVM =\
            self.network.create_node('PlotSVM', memory='20G', id_='plot_SVM')
        self.network.node_Barchart =\
            self.network.create_node('PlotBarchart', memory='4G', id_='plot_Barchart')
        self.network.node_STest =\
            self.network.create_node('StatisticalTestFeatures', memory='4G', id_='statistical_test_features')
        self.network.node_Ranked_Percentages =\
            self.network.create_node('PlotRankedScores', memory='20G', id_='plot_ranked_percentages')
        self.network.node_Ranked_Posteriors =\
            self.network.create_node('PlotRankedScores', memory='20G', id_='plot_ranked_posteriors')

        # Create sinks
        self.network.sink_ROC_PNG =\
            self.network.create_sink('PNGFile', id_='ROC_PNG')
        self.network.sink_ROC_Tex =\
            self.network.create_sink('TexFile', id_='ROC_Tex')
        self.network.sink_ROC_CSV =\
            self.network.create_sink('CSVFile', id_='ROC_CSV')

        self.network.sink_SVM_Json =\
            self.network.create_sink('JsonFile', id_='SVM_Json')

        self.network.sink_Barchart_PNG =\
            self.network.create_sink('PNGFile', id_='Barchart_PNG')
        self.network.sink_Barchart_Tex =\
            self.network.create_sink('TexFile', id_='Barchart_Tex')

        self.network.sink_STest_CSV =\
            self.network.create_sink('CSVFile', id_='StatisticalTestFeatures_CSV')

        self.network.sink_Ranked_Percentages_Zip =\
            self.network.create_sink('ZipFile', id_='RankedPercentages_Zip')
        self.network.sink_Ranked_Percentages_CSV =\
            self.network.create_sink('CSVFile', id_='RankedPercentages_CSV')

        self.network.sink_Ranked_Posteriors_Zip =\
            self.network.create_sink('ZipFile', id_='RankedPosteriors_Zip')
        self.network.sink_Ranked_Posteriors_CSV =\
            self.network.create_sink('CSVFile', id_='RankedPosteriors_CSV')

        # Create links to sinks
        self.network.sink_ROC_PNG.input = self.network.node_ROC.outputs['output_png']
        self.network.sink_ROC_Tex.input = self.network.node_ROC.outputs['output_tex']
        self.network.sink_ROC_CSV.input = self.network.node_ROC.outputs['output_csv']

        self.network.sink_SVM_Json.input = self.network.node_SVM.outputs['output_json']

        self.network.sink_Barchart_PNG.input = self.network.node_Barchart.outputs['output_png']
        self.network.sink_Barchart_Tex.input = self.network.node_Barchart.outputs['output_tex']

        self.network.sink_STest_CSV.input = self.network.node_STest.outputs['performance']

        self.network.sink_Ranked_Percentages_Zip.input = self.network.node_Ranked_Percentages.outputs['output_zip']
        self.network.sink_Ranked_Percentages_CSV.input = self.network.node_Ranked_Percentages.outputs['output_csv']

        self.network.sink_Ranked_Posteriors_Zip.input = self.network.node_Ranked_Posteriors.outputs['output_zip']
        self.network.sink_Ranked_Posteriors_CSV.input = self.network.node_Ranked_Posteriors.outputs['output_csv']

        # Create two constant nodes
        self.network.node_Ranked_Percentages.inputs['scores'] = ['percentages']
        self.network.node_Ranked_Posteriors.inputs['scores'] = ['posteriors']

        # Create sources that are not in WORC and set them
        self.network.source_LabelType = self.network.create_source('String', id_='LabelType')
        self.network.source_Ensemble = self.network.create_source('String', id_='Ensemble')
        self.network.source_LabelType.input = [self.label_type]
        self.network.source_Ensemble.input = [self.ensemble]

        # Create sources if not supplied by a WORC network
        if self.mode == 'StandAlone':
            self.network.source_Estimator = self.network.create_source('HDF5', id_='Estimator')
            self.network.source_PatientInfo = self.network.create_source('PatientInfoFile', id_='PatientInfo')
            self.network.source_Images = self.network.create_source('ITKImageFile', id_='Images', nodegroup='patients')
            self.network.source_Segmentations = self.network.create_source('ITKImageFile', id_='Segmentations', nodegroup='patients')
            self.network.source_Config = self.network.create_source('ParameterFile', id_='Config')

            self.labels = list()
            self.network.source_Features = list()
            for idx in range(0, len(self.features)):
                label = 'Features_' + str(idx)
                self.labels.append(label)
                self.network.source_Features.append(self.network.create_source('HDF5', id_=label, nodegroup='features'))

        # Create links to sources that are not supplied by a WORC network
        self.network.node_ROC.inputs['ensemble'] = self.network.source_Ensemble.output
        self.network.node_ROC.inputs['label_type'] = self.network.source_LabelType.output

        self.network.node_SVM.inputs['ensemble'] = self.network.source_Ensemble.output
        self.network.node_SVM.inputs['label_type'] = self.network.source_LabelType.output

        self.network.node_Barchart.inputs['estimators'] = self.network.source_Ensemble.output
        self.network.node_Barchart.inputs['label_type'] = self.network.source_LabelType.output

        self.network.node_Ranked_Percentages.inputs['ensemble'] = self.network.source_Ensemble.output
        self.network.node_Ranked_Percentages.inputs['label_type'] = self.network.source_LabelType.output

        self.network.node_Ranked_Posteriors.inputs['ensemble'] = self.network.source_Ensemble.output
        self.network.node_Ranked_Posteriors.inputs['label_type'] = self.network.source_LabelType.output

        # Create links to the sources that could be in a WORC network
        if self.mode == 'StandAlone':
            # Sources from the Evaluate network are used
            self.network.node_ROC.inputs['prediction'] = self.network.source_Estimator.output
            self.network.node_ROC.inputs['pinfo'] = self.network.source_PatientInfo.output

            self.network.node_SVM.inputs['prediction'] = self.network.source_Estimator.output
            self.network.node_SVM.inputs['pinfo'] = self.network.source_PatientInfo.output

            self.network.node_Barchart.inputs['prediction'] = self.network.source_Estimator.output

            self.network.links_STest_Features = list()
            for idx, label in enumerate(self.labels):
                self.network.links_STest_Features.append(self.network.node_STest.inputs['features'][str(label)] << self.network.source_Features[idx].output)
                self.network.links_STest_Features[idx].collapse = 'features'
            self.network.node_STest.inputs['patientclass'] = self.network.source_PatientInfo.output
            self.network.node_STest.inputs['config'] = self.network.source_Config.output

            self.network.node_Ranked_Percentages.inputs['estimator'] = self.network.source_Estimator.output
            self.network.node_Ranked_Percentages.inputs['pinfo'] = self.network.source_PatientInfo.output
            self.network.link_images_perc = self.network.create_link(self.network.source_Images.output, self.network.node_Ranked_Percentages.inputs['images'])
            self.network.link_images_perc.collapse = 'patients'
            self.network.link_segmentations_perc = self.network.create_link(self.network.source_Segmentations.output, self.network.node_Ranked_Percentages.inputs['segmentations'])
            self.network.link_segmentations_perc.collapse = 'patients'

            self.network.node_Ranked_Posteriors.inputs['estimator'] = self.network.source_Estimator.output
            self.network.node_Ranked_Posteriors.inputs['pinfo'] = self.network.source_PatientInfo.output
            self.network.link_images_post = self.network.create_link(self.network.source_Images.output, self.network.node_Ranked_Posteriors.inputs['images'])
            self.network.link_images_post.collapse = 'patients'
            self.network.link_segmentations_post = self.network.create_link(self.network.source_Segmentations.output, self.network.node_Ranked_Posteriors.inputs['segmentations'])
            self.network.link_segmentations_post.collapse = 'patients'
        else:
            # Sources from the WORC network are used
            print('WIP')

    def set(self, estimator=None, pinfo=None, images=None,
            segmentations=None, config=None, features=None,
            sink_data={}):
        '''
        Set the sources and sinks based on the provided attributes.
        '''
        if self.mode == 'StandAlone':
            self.source_data = dict()
            self.sink_data = dict()

            self.source_data['Estimator'] = estimator
            self.source_data['PatientInfo'] = pinfo
            self.source_data['Images'] = images
            self.source_data['Segmentations'] = segmentations
            self.source_data['Config'] = config
            self.source_data['LabelType'] = self.label_type
            self.source_data['Ensemble'] = self.ensemble

            for feature, label in zip(features, self.labels):
                self.source_data[label] = feature

            if 'ROC_PNG' not in sink_data.keys():
                self.sink_data['ROC_PNG'] = ("vfs://output/{}/ROC_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
            if 'ROC_Tex' not in sink_data.keys():
                self.sink_data['ROC_Tex'] = ("vfs://output/{}/ROC_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
            if 'ROC_CSV' not in sink_data.keys():
                self.sink_data['ROC_CSV'] = ("vfs://output/{}/ROC_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)

            if 'SVM_Json' not in sink_data.keys():
                self.sink_data['SVM_Json'] = ("vfs://output/{}/performance_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)

            if 'Barchart_PNG' not in sink_data.keys():
                self.sink_data['Barchart_PNG'] = ("vfs://output/{}/Barchart_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
            if 'Barchart_Tex' not in sink_data.keys():
                self.sink_data['Barchart_Tex'] = ("vfs://output/{}/Barchart_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)

            if 'StatisticalTestFeatures_CSV' not in sink_data.keys():
                self.sink_data['StatisticalTestFeatures_CSV'] = ("vfs://output/{}/StatisticalTestFeatures_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)

            if 'RankedPercentages_Zip' not in sink_data.keys():
                self.sink_data['RankedPercentages_Zip'] = ("vfs://output/{}/RankedPercentages_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
            if 'RankedPercentages_CSV' not in sink_data.keys():
                self.sink_data['RankedPercentages_CSV'] = ("vfs://output/{}/RankedPercentages_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)

            if 'RankedPosteriors_Zip' not in sink_data.keys():
                self.sink_data['RankedPosteriors_Zip'] = ("vfs://output/{}/RankedPosteriors_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
            if 'RankedPosteriors_CSV' not in sink_data.keys():
                self.sink_data['RankedPosteriors_CSV'] = ("vfs://output/{}/RankedPosteriors_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)

        else:
            print('[WORC Warning] Evaluate set attribute not needed when WORC network is provided!')

    def execute(self):
        """ Execute the network through the fastr.network.execute command. """
        # Draw and execute nwtwork
        self.network.draw_network(self.network.id, draw_dimension=True)
        self.network.execute(self.source_data, self.sink_data, execution_plugin=self.fastr_plugin, tmpdir=self.fastr_tmpdir)
