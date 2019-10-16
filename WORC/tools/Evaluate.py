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
from fastr.api import ResourceLimit
import os
import graphviz

# NOTE: Very important to give images and segmentations as dict with patient names!


class Evaluate(object):
    def __init__(self, label_type, ensemble=50, scores='percentages',
                 parent=None, features=None,
                 fastr_plugin='LinearExecution',
                 name='Example'):
        '''
        Build a network that evaluates the performance of an estimator.

        Parameters
        ----------

        network: fastr network, default None
                If you input a network, the evaluate network is added
                to the existing network.

        '''
        if parent is not None:
            self.parent = parent
            self.network = parent.network
            self.mode = 'WORC'
            self.name = parent.network.id
            self.ensemble = parent.configs[0]['Ensemble']['Use']
        else:
            self.mode = 'StandAlone'
            self.fastr_plugin = fastr_plugin
            self.name = 'WORC_Evaluate_' + name
            self.network = fastr.create_network(id=self.name)
            self.fastr_tmpdir = os.path.join(fastr.config.mounts['tmp'], self.name)
            self.ensemble = ensemble

        if features is None and self.mode == 'StandAlone':
            raise WORCexceptions.WORCIOError('Either features as input or a WORC network is required for the Evaluate network.')

        self.features = features

        self.label_type = label_type

        self.create_network()

    def create_network(self):
        '''
        Add evaluate components to network.
        '''

        # Create all nodes
        self.node_ROC =\
            self.network.create_node('worc/PlotROC:1.0', tool_version='1.0', id='plot_ROC', resources=ResourceLimit(memory='20G'))
        self.node_SVM =\
            self.network.create_node('worc/PlotSVM:1.0', tool_version='1.0', id='plot_SVM', resources=ResourceLimit(memory='20G'))
        self.node_Barchart =\
            self.network.create_node('worc/PlotBarchart:1.0', tool_version='1.0', id='plot_Barchart', resources=ResourceLimit(memory='4G'))
        self.node_STest =\
            self.network.create_node('worc/StatisticalTestFeatures:1.0', tool_version='1.0', id='statistical_test_features', resources=ResourceLimit(memory='4G'))
        self.node_Ranked_Percentages =\
            self.network.create_node('worc/PlotRankedScores:1.0', tool_version='1.0', id='plot_ranked_percentages', resources=ResourceLimit(memory='20G'))
        self.node_Ranked_Posteriors =\
            self.network.create_node('worc/PlotRankedScores:1.0', tool_version='1.0', id='plot_ranked_posteriors', resources=ResourceLimit(memory='20G'))

        # Create sinks
        self.sink_ROC_PNG =\
            self.network.create_sink('PNGFile', id='ROC_PNG')
        self.sink_ROC_Tex =\
            self.network.create_sink('TexFile', id='ROC_Tex')
        self.sink_ROC_CSV =\
            self.network.create_sink('CSVFile', id='ROC_CSV')

        self.sink_SVM_Json =\
            self.network.create_sink('JsonFile', id='SVM_Json')

        self.sink_Barchart_PNG =\
            self.network.create_sink('PNGFile', id='Barchart_PNG')
        self.sink_Barchart_Tex =\
            self.network.create_sink('TexFile', id='Barchart_Tex')

        self.sink_STest_CSV =\
            self.network.create_sink('CSVFile', id='StatisticalTestFeatures_CSV')

        self.sink_Ranked_Percentages_Zip =\
            self.network.create_sink('ZipFile', id='RankedPercentages_Zip')
        self.sink_Ranked_Percentages_CSV =\
            self.network.create_sink('CSVFile', id='RankedPercentages_CSV')

        self.sink_Ranked_Posteriors_Zip =\
            self.network.create_sink('ZipFile', id='RankedPosteriors_Zip')
        self.sink_Ranked_Posteriors_CSV =\
            self.network.create_sink('CSVFile', id='RankedPosteriors_CSV')

        # Create links to sinks
        self.sink_ROC_PNG.input = self.node_ROC.outputs['output_png']
        self.sink_ROC_Tex.input = self.node_ROC.outputs['output_tex']
        self.sink_ROC_CSV.input = self.node_ROC.outputs['output_csv']

        self.sink_SVM_Json.input = self.node_SVM.outputs['output_json']

        self.sink_Barchart_PNG.input = self.node_Barchart.outputs['output_png']
        self.sink_Barchart_Tex.input = self.node_Barchart.outputs['output_tex']

        self.sink_STest_CSV.input = self.node_STest.outputs['performance']

        self.sink_Ranked_Percentages_Zip.input = self.node_Ranked_Percentages.outputs['output_zip']
        self.sink_Ranked_Percentages_CSV.input = self.node_Ranked_Percentages.outputs['output_csv']

        self.sink_Ranked_Posteriors_Zip.input = self.node_Ranked_Posteriors.outputs['output_zip']
        self.sink_Ranked_Posteriors_CSV.input = self.node_Ranked_Posteriors.outputs['output_csv']

        # Create two constant nodes
        self.node_Ranked_Percentages.inputs['scores'] = ['percentages']
        self.node_Ranked_Posteriors.inputs['scores'] = ['posteriors']

        # Create sources that are not in WORC and set them
        # self.source_LabelType = self.network.create_source('String', id='LabelType')
        # self.source_Ensemble = self.network.create_source('String', id='Ensemble')
        # self.source_LabelType.input = [self.label_type]
        # self.source_Ensemble.input = [self.ensemble]

        self.source_LabelType = self.network.create_constant('String', [self.label_type], id='LabelType')
        self.source_Ensemble = self.network.create_constant('String', [self.ensemble], id='Ensemble')

        # Create sources if not supplied by a WORC network
        if self.mode == 'StandAlone':
            self.source_Estimator = self.network.create_source('HDF5', id='Estimator')
            self.source_PatientInfo = self.network.create_source('PatientInfoFile', id='PatientInfo')
            self.source_Images = self.network.create_source('ITKImageFile', id='Images', node_group='patients')
            self.source_Segmentations = self.network.create_source('ITKImageFile', id='Segmentations', node_group='patients')
            self.source_Config = self.network.create_source('ParameterFile', id='Config')

            self.labels = list()
            self.source_Features = list()
            for idx in range(0, len(self.features)):
                label = 'Features_' + str(idx)
                self.labels.append(label)
                self.source_Features.append(self.network.create_source('HDF5', id=label, node_group='features'))

        # Create links to sources that are not supplied by a WORC network
        self.node_ROC.inputs['ensemble'] = self.source_Ensemble.output
        self.node_ROC.inputs['label_type'] = self.source_LabelType.output

        self.node_SVM.inputs['ensemble'] = self.source_Ensemble.output
        self.node_SVM.inputs['label_type'] = self.source_LabelType.output

        self.node_Barchart.inputs['estimators'] = self.source_Ensemble.output
        self.node_Barchart.inputs['label_type'] = self.source_LabelType.output

        self.node_Ranked_Percentages.inputs['ensemble'] = self.source_Ensemble.output
        self.node_Ranked_Percentages.inputs['label_type'] = self.source_LabelType.output

        self.node_Ranked_Posteriors.inputs['ensemble'] = self.source_Ensemble.output
        self.node_Ranked_Posteriors.inputs['label_type'] = self.source_LabelType.output

        # Create links to the sources that could be in a WORC network
        if self.mode == 'StandAlone':
            self.create_links_Standalone()
        else:
            self.create_links_Addon()

    def create_links_Standalone(self):
            # Sources from the Evaluate network are used
            self.node_ROC.inputs['prediction'] = self.source_Estimator.output
            self.node_ROC.inputs['pinfo'] = self.source_PatientInfo.output

            self.node_SVM.inputs['prediction'] = self.source_Estimator.output
            self.node_SVM.inputs['pinfo'] = self.source_PatientInfo.output

            self.node_Barchart.inputs['prediction'] = self.source_Estimator.output

            self.links_STest_Features = list()
            for idx, label in enumerate(self.labels):
                self.links_STest_Features.append(self.node_STest.inputs['features'][str(label)] << self.source_Features[idx].output)
                self.links_STest_Features[idx].collapse = 'features'
            self.node_STest.inputs['patientclass'] = self.source_PatientInfo.output
            self.node_STest.inputs['config'] = self.source_Config.output

            self.node_Ranked_Percentages.inputs['estimator'] = self.source_Estimator.output
            self.node_Ranked_Percentages.inputs['pinfo'] = self.source_PatientInfo.output
            self.link_images_perc = self.network.create_link(self.source_Images.output, self.node_Ranked_Percentages.inputs['images'])
            self.link_images_perc.collapse = 'patients'
            self.link_segmentations_perc = self.network.create_link(self.source_Segmentations.output, self.node_Ranked_Percentages.inputs['segmentations'])
            self.link_segmentations_perc.collapse = 'patients'

            self.node_Ranked_Posteriors.inputs['estimator'] = self.source_Estimator.output
            self.node_Ranked_Posteriors.inputs['pinfo'] = self.source_PatientInfo.output
            self.link_images_post = self.network.create_link(self.source_Images.output, self.node_Ranked_Posteriors.inputs['images'])
            self.link_images_post.collapse = 'patients'
            self.link_segmentations_post = self.network.create_link(self.source_Segmentations.output, self.node_Ranked_Posteriors.inputs['segmentations'])
            self.link_segmentations_post.collapse = 'patients'

    def create_links_Addon(self):
            # Sources from the WORC network are used
            prediction = self.parent.classify.outputs['classification']
            if self.parent.labels_test:
                pinfo = self.parent.source_patientclass_test.output
            else:
                pinfo = self.parent.source_patientclass_train.output

            config = self.parent.source_class_config.output

            if self.parent.sources_images_train:
                # NOTE: Use images of first modality to depict tumor
                label = self.parent.modlabels[0]
                images = self.parent.sources_images_train[label].output
                segmentations = self.parent.sources_segmentations_train[label].output

            self.node_ROC.inputs['prediction'] = prediction
            self.node_ROC.inputs['pinfo'] = pinfo

            self.node_SVM.inputs['prediction'] = prediction
            self.node_SVM.inputs['pinfo'] = pinfo

            self.node_Barchart.inputs['prediction'] = prediction

            self.links_STest_Features = dict()
            for idx, label in enumerate(self.parent.modlabels):
                # NOTE: Currently statistical testing is only done within the training set
                if self.parent.sources_images_train:
                    # Features are computed within the network
                    self.links_STest_Features[label] = self.node_STest.inputs['features'][str(label)] << self.parent.calcfeatures_train[label].outputs['features']
                else:
                    # Feature are precomputed and given as sources
                    self.links_STest_Features[label] = self.node_STest.inputs['features'][str(label)] << self.parent.sources_features_train[label].output
                self.links_STest_Features[label].collapse = 'train'

            self.node_STest.inputs['patientclass'] = pinfo
            self.node_STest.inputs['config'] = config

            self.node_Ranked_Percentages.inputs['estimator'] = prediction
            self.node_Ranked_Percentages.inputs['pinfo'] = pinfo
            self.node_Ranked_Posteriors.inputs['estimator'] = prediction
            self.node_Ranked_Posteriors.inputs['pinfo'] = pinfo

            if self.parent.sources_images_train:
                self.link_images_perc = self.network.create_link(images, self.node_Ranked_Percentages.inputs['images'])
                self.link_images_perc.collapse = 'train'
                self.link_segmentations_perc = self.network.create_link(segmentations, self.node_Ranked_Percentages.inputs['segmentations'])
                self.link_segmentations_perc.collapse = 'train'

                self.link_images_post = self.network.create_link(images, self.node_Ranked_Posteriors.inputs['images'])
                self.link_images_post.collapse = 'train'
                self.link_segmentations_post = self.network.create_link(segmentations, self.node_Ranked_Posteriors.inputs['segmentations'])
                self.link_segmentations_post.collapse = 'train'

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
        else:
            self.sink_data = self.parent.sink_data

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

    def execute(self):
        """ Execute the network through the fastr.network.execute command. """
        # Draw and execute nwtwork
        try:
            self.network.draw(file_path=self.network.id + '.svg', draw_dimensions=True)
        except graphviz.backend.ExecutableNotFound:
            print('[WORC WARNING] Graphviz executable not found: not drawing network diagram. MAke sure the Graphviz executables are on your systems PATH.')
        self.network.execute(self.source_data, self.sink_data, execution_plugin=self.fastr_plugin, tmpdir=self.fastr_tmpdir)
