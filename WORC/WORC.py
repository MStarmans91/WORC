#!/usr/bin/env python

# Copyright 2011-2017 Biomedical Imaging Group Rotterdam, Departments of
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

import configparser
import fastr
from sklearn.model_selection import ParameterGrid
import os
import json
from random import randint
import addexceptions as WORCexceptions
import IOparser.config_WORC as config_io


class WORC(object):
    """A Workflow for Optimal Radiomics Classification object with the
       following properties:

    Attributes:
        name: name of the network.
        configs: Configuration parameters (list, required)
        labels: labels of patients (required)
        network: the FASTR network to use. Needs to be generated through "build".
        images: to be used for Radiomics computation (list, optional)
        segmentations: annotations for images (list, optional)
        semantics: semantic features per image type (list, optional)
        masks: state which pixels of images are valid (list, optional)
        features: input Radiomics features for classification (list, optional)
        metadata: DICOM headers belonging to images (list, optional)
        Elastix_Para: parameter files for Elastix (list, optional)
        fastr_plugin: plugin to use for FASTR execution
        fastr_tempdir: temporary directory to use for FASTR execution
        additions: additional inputs for your network (dict, optional)
        source_data: data to use as sources for FASTR (dict)
        sink_data: data to use as sinks for FASTR (dict)
    """

    def __init__(self, name=''):
        """Initialize WORC object. Set the initial variables all to None,
           except for some defaults.

        Arguments:
            name: name of the nework (string, optional)

        """
        self.name = name
        self.images = list()
        self.configs = list()
        self.fastrconfigs = list()
        self.segmentations = list()
        self.semantics = list()
        self.labels = list()
        self.masks = list()
        self.features = list()
        self.metadata = list()
        self.Elastix_Para = list()
        self.fastr_plugin = 'ProcessPoolExecution'
        if name == '':
            name = [randint(0, 9) for p in range(0,5)]
            self.fastr_tmpdir = os.path.join(fastr.config.mounts['tmp'], 'fastr' + str(name))
        else:
            self.fastr_tmpdir = os.path.join(fastr.config.mounts['tmp'], name)
        self.additions = dict()

    def generate_parameters(self, parameter_space, name):
        """Generate parameters for a gridsearch among all used feature groups.

        Arguments:
            parameter_space: dictionary containing all parameters and their
                             possible values among which can be searched.

        Returns:
            sources: list of file URI's in FASTR's vfs format pointing to the
                     parameter files
        """

        # Create JSON files for all parameter settings
        sources = dict()
        tempdir = os.path.join(fastr.config.mounts['output'], name[0], 'GS')
        if not os.path.exists(tempdir):
            os.makedirs(tempdir)

        # Convert configparser sections to dict
        parameter_space_dict = dict()
        for key in parameter_space.keys():
            parameter_space_dict[key] =\
                [str(item).strip() for item in
                 parameter_space[key].split(',')]

        for num, parameters in enumerate(ParameterGrid(parameter_space_dict)):
            parameters["Number"] = str(num)

            # Convert parameter set to json
            fname = ('settings_{}_{}.json').format(name[1], str(num))
            output = os.path.join(tempdir, fname)
            with open(output, 'w') as fp:
                json.dump(parameters, fp, indent=4)

            source = ("vfs://output/{}/{}/{}").format(name[0], 'GS', fname)
            sources[str(num)] = source

        return sources

    def defaultconfig(self):
        """Generate a configparser object holding all default configuration values.

        Returns:
            config: configparser configuration file

        """
        # TODO: cluster parallel execution parameters
        config = configparser.ConfigParser()
        config.optionxform = str

        # General configuration of WORC
        config['General'] = dict()
        config['General']['cross_validation'] = 'True'
        config['General']['construction_type'] = 'Single'
        config['General']['write_results'] = 'True'
        config['General']['segment'] = 'True'
        config['General']['patientclass'] = 'XNAT'
        config['General']['gridsearch_SVM'] = 'FASTR'
        config['General']['fastr'] = 'True'
        config['General']['Segmentix'] = 'False'
        config['General']['PCE'] = 'False'  # We do not yet provide this module
        config['General']['FeatureCalculator'] = 'CalcFeatures'
        config['General']['ElastixNode'] = "Elastix"
        config['General']['TransformixNode'] = "Transformix"

        # Segmentix
        config['Segmentix'] = dict()
        config['Segmentix']['mask'] = 'subtract'
        config['Segmentix']['segtype'] = 'None'
        config['Segmentix']['segradius'] = '5'

        # Gridsearch options
        config['SVMFeatures'] = dict()
        config['SVMFeatures']['shape_features'] = 'True'
        config['SVMFeatures']['histogram_features'] = 'True, False'
        config['SVMFeatures']['orientation_features'] = 'True'
        config['SVMFeatures']['texture_features'] = 'True, False, Gabor, LBP, GLCM'
        config['SVMFeatures']['patient_features'] = 'True, False'
        config['SVMFeatures']['semantic_features'] = 'True, False'

        # PREDICT - Feature calculation
        config['ImageFeatures'] = dict()
        config['ImageFeatures']['segtype'] = 'Ring'
        config['ImageFeatures']['segradius'] = '5'

        config['ImageFeatures']['orientation'] = 'True'
        config['ImageFeatures']['texture'] = 'all'

        # Defines what should be done with the images
        config['ImageFeatures']['image_type'] = 'CT'

        # Define frequencies for gabor filter in pixels
        config['ImageFeatures']['gabor_frequencies'] = '0.05, 0.2, 0.5'

        # Gabor angles in degrees
        config['ImageFeatures']['gabor_angles'] = '0, 45, 90, 135'

        # PREDICT - Classification
        config['Classification'] = dict()
        config['Classification']['fastr'] = 'True'
        config['Classification']['classifier'] = 'SVM'

        config['CrossValidation'] = dict()
        config['CrossValidation']['N_iterations'] = '50'
        config['CrossValidation']['test_size'] = '0.2'

        config['Genetics'] = dict()
        config['Genetics']['mutation_type'] = '[GP]'
        config['Genetics']['url'] = 'http://bigr-rad-xnat.erasmusmc.nl'
        config['Genetics']['projectID'] = 'LGG-Radiogenom'

        config['HyperOptimization'] = dict()
        config['HyperOptimization']['scoring_method'] = 'f1_weighted'
        config['HyperOptimization']['test_size'] = '0.15'
        config['HyperOptimization']['N_iterations'] = '10'
        config['HyperOptimization']['score_threshold'] = '0.02'

        return config

    def build(self):
        """Build the network based on the given attributes."""

        # We either need images or features for Radiomics
        if self.images or self.features:
            # We currently require labels for supervised learning
            if self.labels:
                if not self.configs:
                    print("No configuration given, assuming default")
                    if self.images:
                        self.configs = [self.defaultconfig()] * len(self.images)
                    else:
                        self.configs = [self.defaultconfig()] * len(self.features)
                self.network = fastr.Network('WORC' + self.name)

                # BUG: We currently use the first configuration as general config
                image_types = list()
                for c in range(len(self.configs)):
                    if type(self.configs[c]) == str():
                        # Probably, c is a configuration file
                        self.configs[c] = config_io.load_config(c)

                    image_types.append(self.configs[c]['ImageFeatures']['image_type'])

                # Classification tool and label source
                self.network.source_patientclass = self.network.create_source('PatientInfoFile', id_='patientclass', nodegroup='conf')

                self.network.classify = self.network.create_node('SVMGS', memory='6G', id_='classify')

                # Outputs
                self.network.sink_classification = self.network.create_sink('HDF5', id_='classification')
                self.network.sink_performance = self.network.create_sink('JsonFile', id_='performance')

                # Links
                self.network.source_class_config = self.network.create_source('ParameterFile', id_='config_classification', nodegroup='conf')
                self.network.source_class_parameters = self.network.create_source('JsonFile', id_='parameters_classification')
                self.network.classify.inputs['parameters'] = self.network.source_class_parameters.output
                self.network.link_class_1 = self.network.create_link(self.network.source_class_config.output, self.network.classify.inputs['config'])
                self.network.link_class_2 = self.network.create_link(self.network.source_patientclass.output, self.network.classify.inputs['patientclass'])
                self.network.link_class_1.collapse = 'conf'
                self.network.link_class_2.collapse = 'conf'

                self.network.sink_classification.input = self.network.classify.outputs['classification']
                self.network.sink_performance.input = self.network.classify.outputs['performance']

                if not self.features:
                    # Create nodes to compute features
                    self.network.calcfeatures = dict()
                    self.network.sources_images = dict()
                    self.network.sources_metadata = dict()
                    self.network.sources_parameters = dict()
                    self.network.sinks_features = dict()
                    self.network.converters_im = dict()
                    self.network.converters_seg = dict()
                    self.network.links_C1 = dict()

                    # Check which nodes are necessary
                    if not self.segmentations:
                        message = "No automatic segmentations implemented."
                        raise WORCexceptions.WORCNotImplementedError(message)

                    elif len(self.segmentations) == len(image_types):
                        # Segmentations provided
                        self.network.sources_segmentation = dict()
                        segmode = 'Provided'

                    elif len(self.segmentations) == 1:
                        # Assume segmentations need to be registered to other modalities
                        self.network.sources_segmentation = dict()
                        segmode = 'Register'

                        self.network.source_Elastix_Parameters = dict()
                        self.network.elastix_nodes = dict()
                        self.network.transformix_nodes = dict()
                        self.network.sinks_transformations = dict()
                        self.network.sinks_segmentations = dict()
                        pass

                    # BUG: We assume that first type defines if we use segmentix
                    if self.configs[0]['General']['Segmentix']:
                        # Use the segmentix toolbox for segmentation processing
                        self.network.sinks_segmentation = dict()
                        self.network.sources_masks = dict()
                        self.network.converters_masks = dict()
                        self.network.nodes_segmentix = dict()

                    if self.semantics:
                        self.network.sources_semantics = dict()

                    # Create label for each modality/image
                    self.modlabels = list()
                    for num, mod in enumerate(image_types):
                        num = 0
                        label = mod + '_' + str(num)
                        while label in self.network.calcfeatures.keys():
                            # if label already exists, add number to label
                            num += 1
                            label = mod + str(num)
                        self.modlabels.append(label)

                        # Create required sources and sinks
                        self.network.sources_metadata[label] = self.network.create_source('DicomImageFile', id_='metadata_' + label, nodegroup='input')
                        self.network.sources_parameters[label] = self.network.create_source('ParameterFile', id_='parameters_' + label)
                        self.network.sources_images[label] = self.network.create_source('ITKImageFile', id_='images_' + label, nodegroup='input')
                        self.network.sinks_features[label] = self.network.create_sink('HDF5', id_='features_' + label)

                        if self.masks:
                            # Create mask source and convert
                            self.network.sources_masks[label] = self.network.create_source('ITKImageFile', id_='mask_' + label, nodegroup='input')
                            self.network.converters_masks[label] = self.network.create_node('PxCastConvert', memory='4G', id_='convert_mask_' + label, nodegroup='input')
                            self.network.converters_masks[label].inputs['image'] = self.network.sources_masks[label].output

                        # First convert the images
                        if any(modality in mod for modality in ['MR', 'CT', 'MG', 'PET']):
                            # Use ITKTools PXCastConvet for converting image formats
                            self.network.converters_im[label] = self.network.create_node('PxCastConvert', memory='4G', id_='convert_im_' + label)

                        elif 'DTI' in mod:
                            # TODO: This reader is currently missing
                            self.network.converters_im[label] = self.network.create_node('DTIreader', memory='4G', id_='convert_im_' + label)

                        else:
                            raise WORCexceptions.WORCTypeError(('No valid image type for modality {}: {} provided.').format(str(num), mod))

                        calcfeat_node = str(self.configs[num]['General']['FeatureCalculator'])
                        self.network.calcfeatures[label] = self.network.create_node(calcfeat_node, memory='14G', id_='calcfeatures_' + label)

                        # Create required links
                        self.network.converters_im[label].inputs['image'] = self.network.sources_images[label].output
                        self.network.calcfeatures[label].inputs['parameters'] = self.network.sources_parameters[label].output
                        self.network.calcfeatures[label].inputs['metadata'] = self.network.sources_metadata[label].output
                        self.network.calcfeatures[label].inputs['image'] = self.network.converters_im[label].outputs['image']

                        if self.semantics:
                            self.network.sources_semantics[label] = self.network.create_source('CSVFile', id_='semantics_' + label)
                            self.network.calcfeatures[label].inputs['semantics'] = self.network.sources_semantics[label].output

                        if segmode == 'Provided':
                            # Use the provided segmantions for each modality
                            self.network.sources_segmentation[label] = self.network.create_source('ITKImageFile', id_='segmentation_' + label, nodegroup='input')
                            self.network.converters_seg[label] = self.network.create_node('PxCastConvert', memory='4G', id_='convert_seg_' + label)
                            self.network.converters_seg[label].inputs['image'] = self.network.sources_segmentation[label].output

                        elif segmode == 'Register':
                            # Align segmentation of first modality to others using registration with Elastix

                            # Create sources and converter for segmentation
                            self.network.sources_segmentation[label] = self.network.create_source('ITKImageFile', id_='segmentation_' + label, nodegroup='input')
                            self.network.converters_seg[label] = self.network.create_node('PxCastConvert', memory='4G', id_='convert_seg_' + label)
                            self.network.converters_seg[label].inputs['image'] = self.network.sources_segmentation[label].output

                            # Assume provided segmentation is on first modality
                            if num > 0:
                                # Use elastix and transformix for registration
                                # NOTE: Assume elastix node type is on first configuration
                                elastix_node = str(self.configs[0]['General']['ElastixNode'])
                                transformix_node = str(self.configs[0]['General']['TransformixNode'])
                                self.network.elastix_nodes[label] = self.network.create_node(elastix_node, id_='elastix_' + label)
                                self.network.transformix_nodes[label] = self.network.create_node(transformix_node, id_='transformix_' + label)

                                # create sources_segmentation
                                # M1 = moving, others = fixed
                                self.network.elastix_nodes[label].inputs['fixed_image'] = self.network.converters_im[label].outputs['image']
                                self.network.elastix_nodes[label].inputs['moving_image'] = self.network.converters_im[self.modlabels[0]].outputs['image']
                                self.network.transformix_nodes[label] .inputs['image'] = self.network.converters_seg[self.modlabels[0]].outputs['image']

                                # Apply registration to input modalities
                                self.network.source_Elastix_Parameters[label] = self.network.create_source('ElastixParameterFile', id_='Elastix_Para_' + label)
                                self.network.elastix_nodes[label].inputs['parameters'] = self.network.source_Elastix_Parameters[label].output
                                if self.masks:
                                    self.network.elastix_nodes[label].inputs['fixed_mask'] = self.network.converters_masks[label].outputs['image']
                                    self.network.elastix_nodes[label].inputs['moving_mask'] = self.network.converters_masks[self.modlabels[0]].outputs['image']

                                # Link data and transformation to transformix and source
                                self.network.transformix_nodes[label].inputs['transform'] = self.network.elastix_nodes[label].outputs['transform']
                                self.network.calcfeatures[label].inputs['segmentation'] = self.network.transformix_nodes[label] .outputs['image']

                                # Save output
                                self.network.sinks_transformations[label] = self.network.create_sink('ElastixTransformFile', id_='transformation_' + label)
                                self.network.sinks_segmentations[label] = self.network.create_sink('ITKImageFile', id_='segmentation_out_' + label)
                                self.network.sinks_transformations.input = self.network.elastix_nodes[label].outputs['transform']
                                self.network.sinks_segmentations[label].input = self.network.transformix_nodes[label].outputs['image']

                        if self.configs[num]['General']['Segmentix']:
                            # Use segmentix node to convert input segmentation into correct contour
                            self.network.sinks_segmentation[label] = self.network.create_sink('ITKImageFile', id_='segmentation_out_' + label)
                            self.network.nodes_segmentix[label] = self.network.create_node('Segmentix', memory='6G', id_='segmentix_' + label)
                            self.network.nodes_segmentix[label].inputs['segmentation_in'] = self.network.converters_seg[label].outputs['image']
                            self.network.nodes_segmentix[label].inputs['parameters'] = self.network.sources_parameters[label].output
                            self.network.calcfeatures[label].inputs['segmentation'] = self.network.nodes_segmentix[label].outputs['segmentation_out']
                            self.network.sinks_segmentation[label].input = self.network.nodes_segmentix[label].outputs['segmentation_out']

                            if self.masks:
                                # Use masks
                                self.network.nodes_segmentix[label].inputs['mask'] = self.network.converters_masks[label].outputs['image']

                        else:
                            self.network.calcfeatures[label].inputs['segmentation'] = self.network.converters_seg[label].outputs['image']

                        self.network.links_C1[label] = self.network.create_link(self.network.calcfeatures[label].outputs['features'], self.network.classify.inputs['features_m' + str(num + 1)])
                        self.network.links_C1[label].collapse = 'input'

                        # Save output
                        self.network.sinks_features[label].input = self.network.calcfeatures[label].outputs['features']

                else:
                    # Features already provided
                    self.network.sources_features = dict()
                    self.network.links_C1 = dict()

                    # Create label for each modality/image
                    self.modlabels = list()
                    for num, mod in enumerate(image_types):
                        num = 0
                        label = mod + str(num)
                        while label in self.network.sources_features.keys():
                            # if label exists, add number to label
                            num += 1
                            label = mod + str(num)
                        self.modlabels.append(label)

                    self.network.sources_features[label] = self.network.create_source('HDF5', id_='features_' + label, nodegroup='input')
                    self.network.links_C1[label] = self.network.create_link(self.network.sources_features[label].output, self.network.classify.inputs['features_m' + str(num+1)])
                    self.network.links_C1[label].collapse = 'input'

                if self.configs[num]['General']['PCE']:
                    # NOTE: PCE Features is currently not probided.
                    # TODO: PCE currently assumes Gaussian feature normalization
                    self.network.PCEnode = self.network.create_node('PCE', memory='64G', id_='PCE')
                    self.network.PCEnode.inputs['svm'] = self.network.classify.outputs['classification']

                    self.network.sink_PCE = self.network.create_sink('MatlabFile', id_='PCE_mat')
                    self.network.sink_SI = self.network.create_sink('MatlabFile', id_='SI_mat')
                    self.network.sink_TS = self.network.create_sink('CSVFile', id_='TS_csv')

                    self.network.sink_PCE.input = self.network.PCEnode.outputs['PCE']
                    self.network.sink_SI.input = self.network.PCEnode.outputs['SI']
                    self.network.sink_TS.input = self.network.PCEnode.outputs['TS']

            else:
                raise WORCexceptions.WORCIOError("Please provide labels.")
        else:
            raise WORCexceptions.WORCIOError("Please provide either images or features.")

    def set(self):
        """ Set the FASTR source and sink data based on the given attributes."""
        self.fastrconfigs = list()
        self.source_data = dict()
        self.sink_data = dict()

        # If the configuration files are confiparse objects, write to file
        for num, c in enumerate(self.configs):
            if type(c) == configparser.ConfigParser:
                cfile = os.path.join(fastr.config.mounts['tmp'], ("config_{}_{}").format(self.name, num))
                with open(cfile, 'w') as configfile:
                    c.write(configfile)
                self.fastrconfigs.append(os.path.join("vfs://tmp/", ("config_{}_{}").format(self.name, num)))

        # Generate gridsearch parameter files if required
        # TODO: We now use the first configuration for the classifier, but his needs to be separated from the rest per modality
        SVM_parameters = self.generate_parameters(self.configs[0]['SVMFeatures'], [self.name, 'SVM'])
        self.source_data['config_classification'] = self.fastrconfigs[0]

        # Set source and sink data
        self.source_data['patientclass'] = self.labels
        self.source_data['parameters_classification'] = SVM_parameters

        self.sink_data['classification'] = ("vfs://output/{}/svm_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
        self.sink_data['performance'] = ("vfs://output/{}/performance_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
        if self.configs[0]['General']['PCE']:
            self.sink_data['PCE_mat'] = ("vfs://output/{}/PCE_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
            self.sink_data['SI_mat'] = ("vfs://output/{}/SI_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
            self.sink_data['TS_csv'] = ("vfs://output/{}/TS_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)

        # BUG: this is a bug in the FASTR package. Workaround for nifti XNAT links using expansion of FASTR XNAT plugin.
        for num, im in enumerate(self.images):
            if im is not None and 'xnat' in im:
                if 'NIFTI' in im:
                    fastr.ioplugins['xnat']
                    # metalink = im_m1.replace('NIFTI','DICOM')
                    im = fastr.ioplugins['xnat'].expand_url(im)
                    self.images[num] = {v[(v.find('subjects') + 9):v.find('/experiments')]: v + 'image.nii.gz?insecure=true' for k, v in im}

        for num, seg in enumerate(self.segmentations):
            if seg is not None and 'xnat' in seg:
                fastr.ioplugins['xnat']
                seg = fastr.ioplugins['xnat'].expand_url(seg)
                self.segmentations[num] = {v[(v.find('subjects') + 9):v.find('/experiments')]: v + 'mask.nii.gz?insecure=true' for k, v in seg}

        # BUG: if we have only a file on one modality, e.g. semantics, what to do with others? You currently need to provide empty files.
        for num, label in enumerate(self.modlabels):
            self.source_data['parameters_' + label] = self.fastrconfigs[num]

            if self.images and len(self.images) >= num:
                self.source_data['images_' + label] = self.images[num]

            if self.masks and len(self.masks) >= num:
                self.source_data['mask_' + label] = self.masks[num]

            if self.metadata and len(self.metadata) >= num:
                self.source_data['metadata_' + label] = self.metadata[num]

            if self.segmentations and len(self.segmentations) >= num:
                self.source_data['segmentation_' + label] = self.segmentations[num]

            if self.semantics and len(self.semantics) >= num:
                self.source_data['semantics_' + label] = self.semantics[num]

            if self.features and len(self.features) >= num:
                self.source_data['features_' + label] = self.features[num]

            if self.Elastix_Para and len(self.Elastix_Para) >= num:
                self.source_data['Elastix_Para_' + label] = self.Elastix_Para[num]

            self.sink_data['segmentation_out_' + label] = ("vfs://output/{}/seg_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
            self.sink_data['features_' + label] = ("vfs://output/{}/features_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
            self.sink_data['transformation_' + label] = ("vfs://output/{}/transformation_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)

    def execute(self):
        """ Exectute the network through the fastr.network.execute command. """
        # Draw and execute nwtwork
        self.network.draw_network(self.network.id, draw_dimension=True)
        self.network.execute(self.source_data, self.sink_data, execution_plugin=self.fastr_plugin, tmpdir=self.fastr_tmpdir, cluster_queue='week')
