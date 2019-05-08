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

import configparser
import fastr
from fastr.api import ResourceLimit
import os
from random import randint
import WORC.addexceptions as WORCexceptions
import WORC.IOparser.config_WORC as config_io
from WORC.tools.Elastix import Elastix
from WORC.tools.Evaluate import Evaluate


class WORC(object):
    """
    A Workflow for Optimal Radiomics Classification (WORC) object that
    serves as a pipeline spawner and manager for optimizating radiomics
    studies. Depending on the attributes set, the object will spawn an
    appropriate pipeline and manage it.

    Note that many attributes are lists and can therefore contain multiple
    instances. For example, when providing two sequences per patient,
    the "images" list contains two items. The type of items in the lists
    is described below.

    All objects that serve as source for your network, i.e. refer to
    actual files to be used, should be formatted as fastr sources suited for
    one of the fastr plugings, see also
    http://fastr.readthedocs.io/en/stable/fastr.reference.html#ioplugin-reference
    The objects should be lists of these fastr sources or dictionaries with the
    sample ID's, e.g.

    images_train = [{'Patient001': vfs://input/CT001.nii.gz,
                     'Patient002': vfs://input/CT002.nii.gz},
                    {'Patient001': vfs://input/MR001.nii.gz,
                     'Patient002': vfs://input/MR002.nii.gz}]

    Attributes
    ------------------
        name: String, default 'WORC'
            name of the network.

        configs: list, required
            Configuration parameters, either ConfigParser objects
            created through the defaultconfig function or paths of config .ini
            files. (list, required)

        labels: list, required
            Paths to files containing patient labels (.txt files).

        network: automatically generated
            The FASTR network generated through the "build" function.

        images: list, optional
            Paths refering to the images used for Radiomics computation. Images
            should be of the ITK Image type.

        segmentations: list, optional
            Paths refering to the segmentations used for Radiomics computation.
            Segmentations should be of the ITK Image type.

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

        CopyMetadata: Boolean, default True
            when using elastix, copy metadata from image to segmentation or not


    """

    def __init__(self, name='WORC'):
        """Initialize WORC object. Set the initial variables all to None,
           except for some defaults.

        Arguments:
            name: name of the nework (string, optional)

        """
        self.name = name

        # Initialize several objects
        self.configs = list()
        self.fastrconfigs = list()

        self.images_train = list()
        self.segmentations_train = list()
        self.semantics_train = list()
        self.labels_train = list()
        self.masks_train = list()
        self.features_train = list()
        self.metadata_train = list()

        self.images_test = list()
        self.segmentations_test = list()
        self.semantics_test = list()
        self.labels_test = list()
        self.masks_test = list()
        self.features_test = list()
        self.metadata_test = list()

        self.Elastix_Para = list()

        # Set some defaults, name
        self.fastr_plugin = 'ProcessPoolExecution'
        if name == '':
            name = [randint(0, 9) for p in range(0, 5)]
        self.fastr_tmpdir = os.path.join(fastr.config.mounts['tmp'], 'WORC_' + str(name))

        self.additions = dict()
        self.CopyMetadata = True
        self.segmode = []

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
        config['General']['Segmentix'] = 'False'
        config['General']['FeatureCalculator'] = 'predict/CalcFeatures:1.0'
        config['General']['Preprocessing'] = 'worc/PreProcess:1.0'
        config['General']['RegistrationNode'] = "'elastix4.8/Elastix:4.8'"
        config['General']['TransformationNode'] = "'elastix4.8/Transformix:4.8'"
        config['General']['Joblib_ncores'] = '4'
        config['General']['Joblib_backend'] = 'multiprocessing'
        config['General']['tempsave'] = 'False'

        # Segmentix
        config['Segmentix'] = dict()
        config['Segmentix']['mask'] = 'subtract'
        config['Segmentix']['segtype'] = 'None'
        config['Segmentix']['segradius'] = '5'
        config['Segmentix']['N_blobs'] = '1'
        config['Segmentix']['fillholes'] = 'False'

        # Preprocessing
        config['Normalize'] = dict()
        config['Normalize']['ROI'] = 'Full'
        config['Normalize']['Method'] = 'z_score'

        # PREDICT - Feature calculation
        # Determine which features are calculated
        config['ImageFeatures'] = dict()
        config['ImageFeatures']['shape'] = 'True'
        config['ImageFeatures']['histogram'] = 'True'
        config['ImageFeatures']['orientation'] = 'True'
        config['ImageFeatures']['texture_Gabor'] = 'False'
        config['ImageFeatures']['texture_LBP'] = 'True'
        config['ImageFeatures']['texture_GLCM'] = 'True'
        config['ImageFeatures']['texture_GLCMMS'] = 'True'
        config['ImageFeatures']['texture_GLRLM'] = 'True'
        config['ImageFeatures']['texture_GLSZM'] = 'True'
        config['ImageFeatures']['texture_NGTDM'] = 'True'
        config['ImageFeatures']['coliage'] = 'False'
        config['ImageFeatures']['vessel'] = 'False'
        config['ImageFeatures']['log'] = 'False'
        config['ImageFeatures']['phase'] = 'False'

        # Parameter settings for PREDICT feature calculation
        # Defines what should be done with the images
        config['ImageFeatures']['image_type'] = 'CT'

        # Define frequencies for gabor filter in pixels
        config['ImageFeatures']['gabor_frequencies'] = '0.05, 0.2, 0.5'

        # Gabor, GLCM angles in degrees and radians, respectively
        config['ImageFeatures']['gabor_angles'] = '0, 45, 90, 135'
        config['ImageFeatures']['GLCM_angles'] = '0, 0.79, 1.57, 2.36'

        # GLCM discretization levels, distances in pixels
        config['ImageFeatures']['GLCM_levels'] = '16'
        config['ImageFeatures']['GLCM_distances'] = '1, 3'

        # LBP radius, number of points in pixels
        config['ImageFeatures']['LBP_radius'] = '3, 8, 15'
        config['ImageFeatures']['LBP_npoints'] = '12, 24, 36'

        # Phase features minimal wavelength and number of scales
        config['ImageFeatures']['phase_minwavelength'] = '3'
        config['ImageFeatures']['phase_nscale'] = '5'

        # Log features sigma of Gaussian in pixels
        config['ImageFeatures']['log_sigma'] = '1, 5, 10'

        # Vessel features scale range, steps for the range
        config['ImageFeatures']['vessel_scale_range'] = '1, 10'
        config['ImageFeatures']['vessel_scale_step'] = '2'

        # Vessel features radius for erosion to determine boudnary
        config['ImageFeatures']['vessel_radius'] = '5'

        # Feature selection
        config['Featsel'] = dict()
        config['Featsel']['Variance'] = 'True, False'
        config['Featsel']['GroupwiseSearch'] = 'True'
        config['Featsel']['SelectFromModel'] = 'False'
        config['Featsel']['UsePCA'] = 'False'
        config['Featsel']['PCAType'] = '95variance'
        config['Featsel']['StatisticalTestUse'] = 'False'
        config['Featsel']['StatisticalTestMetric'] = 'ttest, Welch, Wilcoxon, MannWhitneyU'
        config['Featsel']['StatisticalTestThreshold'] = '0.02, 0.2'
        config['Featsel']['ReliefUse'] = 'False'
        config['Featsel']['ReliefNN'] = '2, 4'
        config['Featsel']['ReliefSampleSize'] = '1, 1'
        config['Featsel']['ReliefDistanceP'] = '1, 3'
        config['Featsel']['ReliefNumFeatures'] = '25, 200'

        # Groupwise Featureselection options
        config['SelectFeatGroup'] = dict()
        config['SelectFeatGroup']['shape_features'] = 'True, False'
        config['SelectFeatGroup']['histogram_features'] = 'True, False'
        config['SelectFeatGroup']['orientation_features'] = 'True, False'
        config['SelectFeatGroup']['texture_Gabor_features'] = 'True, False'
        config['SelectFeatGroup']['texture_GLCM_features'] = 'True, False'
        config['SelectFeatGroup']['texture_GLCMMS_features'] = 'True, False'
        config['SelectFeatGroup']['texture_GLRLM_features'] = 'True, False'
        config['SelectFeatGroup']['texture_GLSZM_features'] = 'True, False'
        config['SelectFeatGroup']['texture_NGTDM_features'] = 'True, False'
        config['SelectFeatGroup']['texture_LBP_features'] = 'True, False'
        config['SelectFeatGroup']['patient_features'] = 'False'
        config['SelectFeatGroup']['semantic_features'] = 'False'
        config['SelectFeatGroup']['coliage_features'] = 'False'
        config['SelectFeatGroup']['log_features'] = 'False'
        config['SelectFeatGroup']['vessel_features'] = 'False'
        config['SelectFeatGroup']['phase_features'] = 'False'

        # Feature imputation
        config['Imputation'] = dict()
        config['Imputation']['use'] = 'False'
        config['Imputation']['strategy'] = 'mean, median, most_frequent, constant, knn'
        config['Imputation']['n_neighbors'] = '5, 5'

        # Classification
        config['Classification'] = dict()
        config['Classification']['fastr'] = 'True'
        config['Classification']['fastr_plugin'] = self.fastr_plugin
        config['Classification']['classifiers'] = 'SVM'
        config['Classification']['max_iter'] = '100000'
        config['Classification']['SVMKernel'] = 'poly'
        config['Classification']['SVMC'] = '0, 6'
        config['Classification']['SVMdegree'] = '1, 6'
        config['Classification']['SVMcoef0'] = '0, 1'
        config['Classification']['SVMgamma'] = '-5, 5'
        config['Classification']['RFn_estimators'] = '10, 90'
        config['Classification']['RFmin_samples_split'] = '2, 3'
        config['Classification']['RFmax_depth'] = '5, 5'
        config['Classification']['LRpenalty'] = 'l2, l1'
        config['Classification']['LRC'] = '0.01, 1.0'
        config['Classification']['LDA_solver'] = 'svd, lsqr, eigen'
        config['Classification']['LDA_shrinkage'] = '-5, 5'
        config['Classification']['QDA_reg_param'] = '-5, 5'
        config['Classification']['ElasticNet_alpha'] = '-5, 5'
        config['Classification']['ElasticNet_l1_ratio'] = '0, 1'
        config['Classification']['SGD_alpha'] = '-5, 5'
        config['Classification']['SGD_l1_ratio'] = '0, 1'
        config['Classification']['SGD_loss'] = 'hinge, squared_hinge, modified_huber'
        config['Classification']['SGD_penalty'] = 'none, l2, l1'
        config['Classification']['CNB_alpha'] = '0, 1'

        # CrossValidation
        config['CrossValidation'] = dict()
        config['CrossValidation']['N_iterations'] = '100'
        config['CrossValidation']['test_size'] = '0.2'

        # Options for the object/patient labels that are used
        config['Labels'] = dict()
        config['Labels']['label_names'] = 'Label1, Label2'
        config['Labels']['modus'] = 'singlelabel'
        config['Labels']['url'] = 'WIP'
        config['Labels']['projectID'] = 'WIP'

        # Hyperparameter optimization options
        config['HyperOptimization'] = dict()
        config['HyperOptimization']['scoring_method'] = 'f1_weighted'
        config['HyperOptimization']['test_size'] = '0.15'
        config['HyperOptimization']['N_iterations'] = '10000'
        config['HyperOptimization']['n_jobspercore'] = '2000'  # only relevant when using fastr in classification

        # Feature scaling options
        config['FeatureScaling'] = dict()
        config['FeatureScaling']['scale_features'] = 'True'
        config['FeatureScaling']['scaling_method'] = 'z_score'

        # Sample processing options
        config['SampleProcessing'] = dict()
        config['SampleProcessing']['SMOTE'] = 'True'
        config['SampleProcessing']['SMOTE_ratio'] = '1, 0'
        config['SampleProcessing']['SMOTE_neighbors'] = '5, 15'
        config['SampleProcessing']['Oversampling'] = 'False'

        # Ensemble options
        config['Ensemble'] = dict()
        config['Ensemble']['Use'] = 'False'  # Still WIP

        return config

    def add_tools(self):
        self.Tools = Tools()

    def build(self, wtype='training'):
        """Build the  network based on the given attributes.

        Parameters
        ----------
        wtype: string, default 'training'
                Specify the WORC execution type.
                - testing: use if you have a trained classifier and want to
                           train it on some new images.
                - training: use if you want to train a classifier from a dataset.

        """

        self.wtype = wtype
        if wtype == 'training':
            self.build_training()
        elif wtype == 'testing':
            self.build_testing()

    def build_training(self):
        """Build the training network based on the given attributes."""
        # We either need images or features for Radiomics
        if self.images_train or self.features_train:
            # We currently require labels for supervised learning
            if self.labels_train:
                if not self.configs:
                    print("No configuration given, assuming default")
                    if self.images_train:
                        self.configs = [self.defaultconfig()] * len(self.images_train)
                    else:
                        self.configs = [self.defaultconfig()] * len(self.features_train)
                self.network = fastr.create_network('WORC_' + self.name)

                # BUG: We currently use the first configuration as general config
                image_types = list()
                for c in range(len(self.configs)):
                    if type(self.configs[c]) == str:
                        # Probably, c is a configuration file
                        self.configs[c] = config_io.load_config(self.configs[c])
                    image_types.append(self.configs[c]['ImageFeatures']['image_type'])

                # Create config source
                self.source_class_config = self.network.create_source('ParameterFile', id='config_classification_source', node_group='conf')

                # Classification tool and label source
                self.source_patientclass_train = self.network.create_source('PatientInfoFile', id='patientclass_train', node_group='pctrain')
                if self.labels_test:
                    self.source_patientclass_test = self.network.create_source('PatientInfoFile', id='patientclass_test', node_group='pctest')

                self.classify = self.network.create_node('worc/TrainClassifier:1.0', tool_version='1.0', id='classify', resources=ResourceLimit(memory='12G'))

                # Outputs
                self.sink_classification = self.network.create_sink('HDF5', id='classification')
                self.sink_performance = self.network.create_sink('JsonFile', id='performance')
                self.sink_class_config = self.network.create_sink('ParameterFile', id='config_classification_sink', node_group='conf')

                # Links
                self.sink_class_config.input = self.source_class_config.output
                self.link_class_1 = self.network.create_link(self.source_class_config.output, self.classify.inputs['config'])
                self.link_class_2 = self.network.create_link(self.source_patientclass_train.output, self.classify.inputs['patientclass_train'])
                self.link_class_1.collapse = 'conf'
                self.link_class_2.collapse = 'pctrain'

                if self.images_test or self.features_test:
                    # FIXME: the naming here is ugly
                    self.link_class_3 = self.network.create_link(self.source_patientclass_test.output, self.classify.inputs['patientclass_test'])
                    self.link_class_3.collapse = 'pctest'

                self.sink_classification.input = self.classify.outputs['classification']
                self.sink_performance.input = self.classify.outputs['performance']

                if not self.features_train:
                    # Create nodes to compute features
                    self.sources_parameters = dict()

                    self.calcfeatures_train = dict()
                    self.preprocessing_train = dict()
                    self.sources_images_train = dict()
                    self.sinks_features_train = dict()
                    self.converters_im_train = dict()
                    self.converters_seg_train = dict()
                    self.links_C1_train = dict()

                    if self.images_test or self.features_test:
                        # A test set is supplied, for which nodes also need to be created
                        self.preprocessing_test = dict()
                        self.calcfeatures_test = dict()
                        self.sources_images_test = dict()
                        self.sinks_features_test = dict()
                        self.converters_im_test = dict()
                        self.converters_seg_test = dict()
                        self.links_C1_test = dict()

                    # Check which nodes are necessary
                    if not self.segmentations_train:
                        message = "No automatic segmentation method is yet implemented."
                        raise WORCexceptions.WORCNotImplementedError(message)

                    elif len(self.segmentations_train) == len(image_types):
                        # Segmentations provided
                        self.sources_segmentations_train = dict()
                        self.sources_segmentations_test = dict()
                        self.segmode = 'Provided'

                    elif len(self.segmentations_train) == 1:
                        # Assume segmentations need to be registered to other modalities
                        self.sources_segmentation = dict()
                        self.segmode = 'Register'

                        self.source_Elastix_Parameters = dict()
                        self.elastix_nodes_train = dict()
                        self.transformix_seg_nodes_train = dict()
                        self.sources_segmentations_train = dict()
                        self.sinks_transformations_train = dict()
                        self.sinks_segmentations_elastix_train = dict()
                        self.sinks_images_elastix_train = dict()
                        self.converters_seg_train = dict()
                        self.edittransformfile_nodes_train = dict()
                        self.transformix_im_nodes_train = dict()

                        self.elastix_nodes_test = dict()
                        self.transformix_seg_nodes_test = dict()
                        self.sources_segmentations_test = dict()
                        self.sinks_transformations_test = dict()
                        self.sinks_segmentations_elastix_test = dict()
                        self.sinks_images_elastix_test = dict()
                        self.converters_seg_test = dict()
                        self.edittransformfile_nodes_test = dict()
                        self.transformix_im_nodes_test = dict()
                        pass

                    # BUG: We assume that first type defines if we use segmentix
                    if self.configs[0]['General']['Segmentix'] == 'True':
                        # Use the segmentix toolbox for segmentation processing
                        self.sinks_segmentations_segmentix_train = dict()
                        self.sources_masks_train = dict()
                        self.converters_masks_train = dict()
                        self.nodes_segmentix_train = dict()

                        if self.images_test or self.features_test:
                            # Also use segmentix on the tes set
                            self.sinks_segmentations_segmentix_test = dict()
                            self.sources_masks_test = dict()
                            self.converters_masks_test = dict()
                            self.nodes_segmentix_test = dict()

                    if self.semantics_train:
                        # Semantic features are supplied
                        self.sources_semantics_train = dict()

                    if self.metadata_train:
                        # Metadata to extract patient features from is supplied
                        self.sources_metadata_train = dict()

                    if self.semantics_test:
                        # Semantic features are supplied
                        self.sources_semantics_test = dict()

                    if self.metadata_test:
                        # Metadata to extract patient features from is supplied
                        self.sources_metadata_test = dict()

                    # Create a part of the pipeline for each modality
                    self.modlabels = list()
                    for nmod, mod in enumerate(image_types):
                        # Create label for each modality/image
                        num = 0
                        label = mod + '_' + str(num)
                        while label in self.calcfeatures_train.keys():
                            # if label already exists, add number to label
                            num += 1
                            label = mod + '_' + str(num)
                        self.modlabels.append(label)

                        # Create required sources and sinks
                        self.sources_parameters[label] = self.network.create_source('ParameterFile', id='parameters_' + label)
                        self.sources_images_train[label] = self.network.create_source('ITKImageFile', id='images_train_' + label, node_group='train')
                        self.sinks_features_train[label] = self.network.create_sink('HDF5', id='features_train_' + label)
                        if self.images_test or self.features_test:
                            self.sources_images_test[label] = self.network.create_source('ITKImageFile', id='images_test_' + label, node_group='test')
                            self.sinks_features_test[label] = self.network.create_sink('HDF5', id='features_test_' + label)

                        if self.metadata_train and len(self.metadata_train) >= nmod + 1:
                            self.sources_metadata_train[label] = self.network.create_source('DicomImageFile', id='metadata_train_' + label, node_group='train')

                        if self.metadata_test and len(self.metadata_test) >= nmod + 1:
                            self.sources_metadata_test[label] = self.network.create_source('DicomImageFile', id='metadata_test_' + label, node_group='test')

                        if self.masks_train and len(self.masks_train) >= nmod + 1:
                            # Create mask source and convert
                            self.sources_masks_train[label] = self.network.create_source('ITKImageFile', id='mask_train_' + label, node_group='train')
                            self.converters_masks_train[label] = self.network.create_node('worc/WORCCastConvert:0.3.2', tool_version='0.1', id='convert_mask_train_' + label, node_group='train', resources=ResourceLimit(memory='4G'))
                            self.converters_masks_train[label].inputs['image'] = self.sources_masks_train[label].output

                        if self.masks_test and len(self.masks_test) >= nmod + 1:
                            # Create mask source and convert
                            self.sources_masks_test[label] = self.network.create_source('ITKImageFile', id='mask_test_' + label, node_group='test')
                            self.converters_masks_test[label] = self.network.create_node('worc/WORCCastConvert:0.3.2', tool_version='0.1', id='convert_mask_test_' + label, node_group='test', resources=ResourceLimit(memory='4G'))
                            self.converters_masks_test[label].inputs['image'] = self.sources_masks_test[label].output

                        # First convert the images
                        if any(modality in mod for modality in ['MR', 'CT', 'MG', 'PET']):
                            # Use ITKTools PXCastConvet for converting image formats
                            self.converters_im_train[label] = self.network.create_node('worc/WORCCastConvert:0.3.2', tool_version='0.1', id='convert_im_train_' + label, resources=ResourceLimit(memory='4G'))
                            if self.images_test or self.features_test:
                                self.converters_im_test[label] = self.network.create_node('worc/WORCCastConvert:0.3.2', tool_version='0.1', id='convert_im_test_' + label, resources=ResourceLimit(memory='4G'))

                        else:
                            raise WORCexceptions.WORCTypeError(('No valid image type for modality {}: {} provided.').format(str(nmod), mod))

                        # Create required links
                        self.converters_im_train[label].inputs['image'] = self.sources_images_train[label].output
                        if self.images_test or self.features_test:
                            self.converters_im_test[label].inputs['image'] = self.sources_images_test[label].output

                        # -----------------------------------------------------
                        # Preprocessing
                        # Create nodes
                        preprocess_node = str(self.configs[nmod]['General']['Preprocessing'])
                        self.preprocessing_train[label] = self.network.create_node(preprocess_node, tool_version='1.0', id='preprocessing_train_' + label, resources=ResourceLimit(memory='4G'))
                        if self.images_test or self.features_test:
                            self.preprocessing_test[label] = self.network.create_node(preprocess_node, tool_version='1.0', id='preprocessing_test_' + label, resources=ResourceLimit(memory='4G'))

                        # Create required links
                        self.preprocessing_train[label].inputs['parameters'] = self.sources_parameters[label].output
                        self.preprocessing_train[label].inputs['image'] = self.converters_im_train[label].outputs['image']

                        if self.images_test or self.features_test:
                            self.preprocessing_test[label].inputs['parameters'] = self.sources_parameters[label].output
                            self.preprocessing_test[label].inputs['image'] = self.converters_im_test[label].outputs['image']

                        if self.metadata_train and len(self.metadata_train) >= nmod + 1:
                            self.preprocessing_train[label].inputs['metadata'] = self.sources_metadata_train[label].output

                        if self.metadata_test and len(self.metadata_test) >= nmod + 1:
                            self.preprocessing_test[label].inputs['metadata'] = self.sources_metadata_test[label].output

                        # -----------------------------------------------------
                        # Create a feature calculator node
                        calcfeat_node = str(self.configs[nmod]['General']['FeatureCalculator'])
                        self.calcfeatures_train[label] = self.network.create_node(calcfeat_node, tool_version='1.0', id='calcfeatures_train_' + label, resources=ResourceLimit(memory='14G'))
                        if self.images_test or self.features_test:
                            self.calcfeatures_test[label] = self.network.create_node(calcfeat_node, tool_version='1.0', id='calcfeatures_test_' + label, resources=ResourceLimit(memory='14G'))

                        # Create required links
                        self.calcfeatures_train[label].inputs['parameters'] = self.sources_parameters[label].output
                        self.calcfeatures_train[label].inputs['image'] = self.preprocessing_train[label].outputs['image']

                        if self.images_test or self.features_test:
                            self.calcfeatures_test[label].inputs['parameters'] = self.sources_parameters[label].output
                            self.calcfeatures_test[label].inputs['image'] = self.preprocessing_test[label].outputs['image']

                        if self.metadata_train and len(self.metadata_train) >= nmod + 1:
                            self.calcfeatures_train[label].inputs['metadata'] = self.sources_metadata_train[label].output

                        if self.metadata_train and len(self.metadata_test) >= nmod + 1:
                            self.calcfeatures_train[label].inputs['metadata'] = self.sources_metadata_train[label].output

                        if self.semantics_train and len(self.semantics_train) >= nmod + 1:
                            self.sources_semantics_train[label] = self.network.create_source('CSVFile', id='semantics_train_' + label)
                            self.calcfeatures_train[label].inputs['semantics'] = self.sources_semantics_train[label].output

                        if self.semantics_test and len(self.semantics_test) >= nmod + 1:
                            self.sources_semantics_test[label] = self.network.create_source('CSVFile', id='semantics_test_' + label)
                            self.calcfeatures_test[label].inputs['semantics'] = self.sources_semantics_test[label].output

                        if self.segmode == 'Provided':
                            # Segmentation -----------------------------------------------------
                            # Use the provided segmantions for each modality
                            self.sources_segmentations_train[label] = self.network.create_source('ITKImageFile', id='segmentations_train_' + label, node_group='train')
                            self.converters_seg_train[label] = self.network.create_node('worc/WORCCastConvert:0.3.2', tool_version='0.1', id='convert_seg_train_' + label, resources=ResourceLimit(memory='4G'))
                            self.converters_seg_train[label].inputs['image'] = self.sources_segmentations_train[label].output

                            if self.images_test or self.features_test:
                                self.sources_segmentations_test[label] = self.network.create_source('ITKImageFile', id='segmentations_test_' + label, node_group='test')
                                self.converters_seg_test[label] = self.network.create_node('worc/WORCCastConvert:0.3.2', tool_version='0.1', id='convert_seg_test_' + label, resources=ResourceLimit(memory='4G'))
                                self.converters_seg_test[label].inputs['image'] = self.sources_segmentations_test[label].output

                        elif self.segmode == 'Register':
                            # Registration nodes -----------------------------------------------------
                            # Align segmentation of first modality to others using registration with Elastix

                            # Create sources and converter for only for the given segmentation, which should be on the first modality
                            if nmod == 0:
                                self.sources_segmentations_train[label] = self.network.create_source('ITKImageFile', id='segmentations_train_' + label, node_group='input')
                                self.converters_seg_train[label] = self.network.create_node('worc/WORCCastConvert:0.3.2', tool_version='0.1', id='convert_seg_train_' + label, resources=ResourceLimit(memory='4G'))
                                self.converters_seg_train[label].inputs['image'] = self.sources_segmentations_train[label].output

                                if self.images_test or self.features_test:
                                    self.sources_segmentations_test[label] = self.network.create_source('ITKImageFile', id='segmentations_test_' + label, node_group='input')
                                    self.converters_seg_test[label] = self.network.create_node('worc/WORCCastConvert:0.3.2', tool_version='0.1', id='convert_seg_test_' + label, resources=ResourceLimit(memory='4G'))
                                    self.converters_seg_test[label].inputs['image'] = self.sources_segmentations_test[label].output

                            # Assume provided segmentation is on first modality
                            if nmod > 0:
                                # Use elastix and transformix for registration
                                # NOTE: Assume elastix node type is on first configuration
                                elastix_node = str(self.configs[0]['General']['RegistrationNode'])
                                transformix_node = str(self.configs[0]['General']['TransformationNode'])
                                self.elastix_nodes_train[label] = self.network.create_node(elastix_node, tool_version='0.2', id='elastix_train_' + label, resources=ResourceLimit(memory='4G'))
                                self.transformix_seg_nodes_train[label] = self.network.create_node(transformix_node, tool_version= '0.2' , id='transformix_seg_train_' + label)
                                self.transformix_im_nodes_train[label] = self.network.create_node(transformix_node, tool_version= '0.2' , id='transformix_im_train_' + label)

                                if self.images_test or self.features_test:
                                    self.elastix_nodes_test[label] = self.network.create_node(elastix_node, tool_version='0.2', id='elastix_test_' + label, resources=ResourceLimit(memory='4G'))
                                    self.transformix_seg_nodes_test[label] = self.network.create_node(transformix_node, tool_version= '0.2' , id='transformix_seg_test_' + label)
                                    self.transformix_im_nodes_test[label] = self.network.create_node(transformix_node, tool_version= '0.2' , id='transformix_im_test_' + label)

                                # Create sources_segmentation
                                # M1 = moving, others = fixed
                                self.elastix_nodes_train[label].inputs['fixed_image'] = self.converters_im_train[label].outputs['image']
                                self.elastix_nodes_train[label].inputs['moving_image'] = self.converters_im_train[self.modlabels[0]].outputs['image']

                                # Add node that copies metadata from the image to the segmentation if required
                                if self.CopyMetadata:
                                    # Copy metadata from the image which was registered to the segmentation, if it is not created yet
                                    if not hasattr(self, "copymetadata_nodes_train"):
                                        # NOTE: Do this for first modality, as we assume segmentation is on that one
                                        self.copymetadata_nodes_train = dict()
                                        self.copymetadata_nodes_train[self.modlabels[0]] = self.network.create_node('itktools/0.3.2/CopyMetadata:1.0', tool_version= '1.0' , id='CopyMetadata_train_' + self.modlabels[0])
                                        self.copymetadata_nodes_train[self.modlabels[0]].inputs["source"] = self.converters_im_train[self.modlabels[0]].outputs['image']
                                        self.copymetadata_nodes_train[self.modlabels[0]].inputs["destination"] = self.converters_seg_train[self.modlabels[0]].outputs['image']
                                    self.transformix_seg_nodes_train[label].inputs['image'] = self.copymetadata_nodes_train[self.modlabels[0]].outputs['output']
                                else:
                                    self.transformix_seg_nodes_train[label].inputs['image'] = self.converters_seg_train[self.modlabels[0]].outputs['image']

                                if self.images_test or self.features_test:
                                    self.elastix_nodes_test[label].inputs['fixed_image'] = self.converters_im_test[label].outputs['image']
                                    self.elastix_nodes_test[label].inputs['moving_image'] = self.converters_im_test[self.modlabels[0]].outputs['image']

                                    if self.CopyMetadata:
                                        # Copy metadata from the image which was registered to the segmentation
                                        if not hasattr(self, "copymetadata_nodes_test"):
                                            # NOTE: Do this for first modality, as we assume segmentation is on that one
                                            self.copymetadata_nodes_test = dict()
                                            self.copymetadata_nodes_test[self.modlabels[0]] = self.network.create_node('itktools/0.3.2/CopyMetadata:1.0', tool_version= '1.0' , id='CopyMetadata_test_' + self.modlabels[0])
                                            self.copymetadata_nodes_test[self.modlabels[0]].inputs["source"] = self.converters_im_test[self.modlabels[0]].outputs['image']
                                            self.copymetadata_nodes_test[self.modlabels[0]].inputs["destination"] = self.converters_seg_test[self.modlabels[0]].outputs['image']
                                        self.transformix_seg_nodes_test[label].inputs['image'] = self.copymetadata_nodes_test[self.modlabels[0]].outputs['output']
                                    else:
                                        self.transformix_seg_nodes_test[label].inputs['image'] = self.converters_seg_test[self.modlabels[0]].outputs['image']

                                # Apply registration to input modalities
                                self.source_Elastix_Parameters[label] = self.network.create_source('ElastixParameterFile', id='Elastix_Para_' + label, node_group='elpara')
                                self.link_elparam_train = self.network.create_link(self.source_Elastix_Parameters[label].output,
                                                                           self.elastix_nodes_train[label].inputs['parameters'])
                                self.link_elparam_train.collapse = 'elpara'

                                if self.images_test or self.features_test:
                                    self.link_elparam_test = self.network.create_link(self.source_Elastix_Parameters[label].output,
                                                                               self.elastix_nodes_test[label].inputs['parameters'])
                                    self.link_elparam_test.collapse = 'elpara'

                                if self.masks_train:
                                    self.elastix_nodes_train[label].inputs['fixed_mask'] = self.converters_masks_train[label].outputs['image']
                                    self.elastix_nodes_train[label].inputs['moving_mask'] = self.converters_masks_train[self.modlabels[0]].outputs['image']

                                if self.images_test or self.features_test:
                                    if self.masks_test:
                                        self.elastix_nodes_test[label].inputs['fixed_mask'] = self.converters_masks_test[label].outputs['image']
                                        self.elastix_nodes_test[label].inputs['moving_mask'] = self.converters_masks_test[self.modlabels[0]].outputs['image']

                                # Change the FinalBSpline Interpolation order to 0 as required for binarie images: see https://github.com/SuperElastix/elastix/wiki/FAQ
                                self.edittransformfile_nodes_train[label] = self.network.create_node('elastixtools/EditElastixTransformFile:0.1', tool_version= '0.1' , id='EditElastixTransformFile' + label)
                                self.edittransformfile_nodes_train[label].inputs['set'] = ["FinalBSplineInterpolationOrder=0"]
                                self.edittransformfile_nodes_train[label].inputs['transform'] = self.elastix_nodes_train[label].outputs['transform'][-1]

                                if self.images_test or self.features_test:
                                    self.edittransformfile_nodes_test[label] = self.network.create_node('elastixtools/EditElastixTransformFile:0.1', tool_version= '0.1' , id='EditElastixTransformFile' + label)
                                    self.edittransformfile_nodes_test[label].inputs['set'] = ["FinalBSplineInterpolationOrder=0"]
                                    self.edittransformfile_nodes_test[label].inputs['transform'] = self.elastix_nodes_test[label].outputs['transform'][-1]

                                # Link data and transformation to transformix and source
                                self.transformix_seg_nodes_train[label].inputs['transform'] = self.edittransformfile_nodes_train[label].outputs['transform']
                                self.calcfeatures_train[label].inputs['segmentation'] = self.transformix_seg_nodes_train[label].outputs['image']

                                self.transformix_im_nodes_train[label].inputs['transform'] = self.elastix_nodes_train[label].outputs['transform'][-1]
                                self.transformix_im_nodes_train[label].inputs['image'] = self.converters_im_train[self.modlabels[0]].outputs['image']

                                if self.images_test or self.features_test:
                                    self.transformix_seg_nodes_test[label].inputs['transform'] = self.edittransformfile_nodes_test[label].outputs['transform']
                                    self.calcfeatures_test[label].inputs['segmentation'] = self.transformix_seg_nodes_test[label] .outputs['image']

                                    self.transformix_im_nodes_test[label].inputs['transform'] = self.elastix_nodes_test[label].outputs['transform'][-1]
                                    self.transformix_im_nodes_test[label].inputs['image'] = self.converters_im_test[self.modlabels[0]].outputs['image']

                                # Save output
                                self.sinks_transformations_train[label] = self.network.create_sink('ElastixTransformFile', id='transformations_train_' + label)
                                self.sinks_segmentations_elastix_train[label] = self.network.create_sink('ITKImageFile', id='segmentations_out_elastix_train_' + label)
                                self.sinks_images_elastix_train[label] = self.network.create_sink('ITKImageFile', id='images_out_elastix_train_' + label)
                                self.sinks_transformations_train[label].input = self.elastix_nodes_train[label].outputs['transform']
                                self.sinks_segmentations_elastix_train[label].input = self.transformix_seg_nodes_train[label].outputs['image']
                                self.sinks_images_elastix_train[label].input = self.transformix_im_nodes_train[label].outputs['image']

                                if self.images_test or self.features_test:
                                    self.sinks_transformations_test[label] = self.network.create_sink('ElastixTransformFile', id='transformations_test_' + label)
                                    self.sinks_segmentations_elastix_test[label] = self.network.create_sink('ITKImageFile', id='segmentations_out_elastix_test_' + label)
                                    self.sinks_images_elastix_test[label] = self.network.create_sink('ITKImageFile', id='images_out_elastix_test_' + label)
                                    self.sinks_transformations_elastix_test[label].input = self.elastix_nodes_test[label].outputs['transform']
                                    self.sinks_segmentations_elastix_test[label].input = self.transformix_seg_nodes_test[label].outputs['image']
                                    self.sinks_images_elastix_test[label].input = self.transformix_im_nodes_test[label].outputs['image']

                        if self.configs[nmod]['General']['Segmentix'] == 'True':
                            # Segmentix nodes -----------------------------------------------------
                            # Use segmentix node to convert input segmentation into correct contour
                            if label not in self.sinks_segmentations_segmentix_train:
                                self.sinks_segmentations_segmentix_train[label] = self.network.create_sink('ITKImageFile', id='segmentations_out_segmentix_train_' + label)

                            self.nodes_segmentix_train[label] = self.network.create_node('segmentix/Segmentix:1.0', tool_version='1.0', id='segmentix_train_' + label, resources=ResourceLimit(memory='6G'))
                            if hasattr(self, 'transformix_seg_nodes_train'):
                                if label in self.transformix_seg_nodes_train.keys():
                                    # Use output of registration in segmentix
                                    self.nodes_segmentix_train[label].inputs['segmentation_in'] = self.transformix_seg_nodes_train[label].outputs['image']
                                else:
                                    # Use original segmentation
                                    self.nodes_segmentix_train[label].inputs['segmentation_in'] = self.converters_seg_train[label].outputs['image']
                            else:
                                # Use original segmentation
                                self.nodes_segmentix_train[label].inputs['segmentation_in'] = self.converters_seg_train[label].outputs['image']

                            self.nodes_segmentix_train[label].inputs['parameters'] = self.sources_parameters[label].output
                            self.calcfeatures_train[label].inputs['segmentation'] = self.nodes_segmentix_train[label].outputs['segmentation_out']
                            self.sinks_segmentations_segmentix_train[label].input = self.nodes_segmentix_train[label].outputs['segmentation_out']

                            if self.images_test or self.features_test:
                                self.sinks_segmentations_segmentix_test[label] = self.network.create_sink('ITKImageFile', id='segmentations_out_segmentix_test_' + label)
                                self.nodes_segmentix_test[label] = self.network.create_node('segmentix/Segmentix:1.0', tool_version='1.0', id='segmentix_test_' + label, resources=ResourceLimit(memory='6G'))
                                if hasattr(self, 'transformix_seg_nodes_test'):
                                    if label in self.transformix_seg_nodes_test.keys():
                                        # Use output of registration in segmentix
                                        self.nodes_segmentix_test[label].inputs['segmentation_in'] = self.transformix_seg_nodes_test[label].outputs['image']
                                    else:
                                        # Use original segmentation
                                        self.nodes_segmentix_test[label].inputs['segmentation_in'] = self.converters_seg_test[label].outputs['image']
                                else:
                                    # Use original segmentation
                                    self.nodes_segmentix_test[label].inputs['segmentation_in'] = self.converters_seg_test[label].outputs['image']

                                self.nodes_segmentix_test[label].inputs['parameters'] = self.sources_parameters[label].output
                                self.calcfeatures_test[label].inputs['segmentation'] = self.nodes_segmentix_test[label].outputs['segmentation_out']
                                self.sinks_segmentations_segmentix_test[label].input = self.nodes_segmentix_test[label].outputs['segmentation_out']

                            if self.masks_train and len(self.masks_train) >= nmod + 1:
                                # Use masks
                                self.nodes_segmentix_train[label].inputs['mask'] = self.converters_masks_train[label].outputs['image']

                            if self.masks_test and len(self.masks_test) >= nmod + 1:
                                # Use masks
                                self.nodes_segmentix_test[label].inputs['mask'] = self.converters_masks_test[label].outputs['image']

                        else:
                            if self.segmode == 'Provided':
                                self.calcfeatures_train[label].inputs['segmentation'] = self.converters_seg_train[label].outputs['image']
                            elif self.segmode == 'Register':
                                if nmod > 0:
                                    self.calcfeatures_train[label].inputs['segmentation'] = self.transformix_seg_nodes_train[label].outputs['image']
                                else:
                                    self.calcfeatures_train[label].inputs['segmentation'] = self.converters_seg_train[label].outputs['image']

                            if self.images_test or self.features_test:
                                if self.segmode == 'Provided':
                                    self.calcfeatures_train[label].inputs['segmentation'] = self.converters_seg_train[label].outputs['image']
                                elif self.segmode == 'Register':
                                    if nmod > 0:
                                        self.calcfeatures_test[label].inputs['segmentation'] = self.transformix_seg_nodes_test[label] .outputs['image']
                                    else:
                                        self.calcfeatures_train[label].inputs['segmentation'] = self.converters_seg_train[label].outputs['image']

                        # Classification nodes -----------------------------------------------------
                        # Add the features from this modality to the classifier node input
                        self.links_C1_train[label] = self.classify.inputs['features_train'][str(label)] << self.calcfeatures_train[label].outputs['features']
                        self.links_C1_train[label].collapse = 'train'

                        if self.images_test or self.features_test:
                            # Add the features from this modality to the classifier node input
                            self.links_C1_test[label] = self.classify.inputs['features_test'][str(label)] << self.calcfeatures_test[label].outputs['features']
                            self.links_C1_test[label].collapse = 'test'

                        # Save output
                        self.sinks_features_train[label].input = self.calcfeatures_train[label].outputs['features']
                        if self.images_test or self.features_test:
                            self.sinks_features_test[label].input = self.calcfeatures_test[label].outputs['features']

                else:
                    # Features already provided: hence we can skip numerous nodes
                    self.sources_features_train = dict()
                    self.links_C1_train = dict()

                    if self.features_test:
                        self.sources_features_test = dict()
                        self.links_C1_test = dict()

                    # Create label for each modality/image
                    self.modlabels = list()
                    for num, mod in enumerate(image_types):
                        num = 0
                        label = mod + str(num)
                        while label in self.sources_features_train.keys():
                            # if label exists, add number to label
                            num += 1
                            label = mod + str(num)
                        self.modlabels.append(label)

                        # Create a node for the feature computation
                        self.sources_features_train[label] = self.network.create_source('HDF5', id='features_train_' + label, node_group='train')

                        # Add the features from this modality to the classifier node input
                        self.links_C1_train[label] = self.classify.inputs['features_train'][str(label)] << self.sources_features_train[label].output
                        self.links_C1_train[label].collapse = 'train'

                        if self.features_test:
                            self.sources_features_test[label] = self.network.create_source('HDF5', id='features_test_' + label, node_group='test')
                            self.links_C1_test[label] = self.classify.inputs['features_test'][str(label)] << self.sources_features_test[label].output
                            self.links_C1_test[label].collapse = 'test'

            else:
                raise WORCexceptions.WORCIOError("Please provide labels.")
        else:
            raise WORCexceptions.WORCIOError("Please provide either images or features.")

    def build_testing(self):
        ''' todo '''

    def set(self):
        """ Set the FASTR source and sink data based on the given attributes."""
        self.fastrconfigs = list()
        self.source_data = dict()
        self.sink_data = dict()

        # If the configuration files are confiparse objects, write to file
        for num, c in enumerate(self.configs):
            if type(c) != configparser.ConfigParser:
                # A filepath (not a fastr source) is provided. Hence we read
                # the config file and convert it to a configparser object
                config = configparser.ConfigParser()
                config.read(c)
                c = config
            cfile = os.path.join(fastr.config.mounts['tmp'], 'WORC_' + self.name, ("config_{}_{}.ini").format(self.name, num))
            if not os.path.exists(os.path.dirname(cfile)):
                os.makedirs(os.path.dirname(cfile))
            with open(cfile, 'w') as configfile:
                c.write(configfile)
            self.fastrconfigs.append(("vfs://tmp/{}/config_{}_{}.ini").format('WORC_' + self.name, self.name, num))

        # Generate gridsearch parameter files if required
        # TODO: We now use the first configuration for the classifier, but his needs to be separated from the rest per modality
        self.source_data['config_classification_source'] = self.fastrconfigs[0]

        # Set source and sink data
        self.source_data['patientclass_train'] = self.labels_train
        self.source_data['patientclass_test'] = self.labels_test

        self.sink_data['classification'] = ("vfs://output/{}/svm_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
        self.sink_data['performance'] = ("vfs://output/{}/performance_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
        self.sink_data['config_classification_sink'] = ("vfs://output/{}/config_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)

        # Set the source data from the WORC objects you created
        for num, label in enumerate(self.modlabels):
            self.source_data['parameters_' + label] = self.fastrconfigs[num]

            # Add train data sources
            if self.images_train and len(self.images_train) - 1 >= num:
                self.source_data['images_train_' + label] = self.images_train[num]

            if self.masks_train and len(self.masks_train) - 1 >= num:
                self.source_data['mask_train_' + label] = self.masks_train[num]

            if self.metadata_train and len(self.metadata_train) - 1 >= num:
                self.source_data['metadata_train_' + label] = self.metadata_train[num]

            if self.segmentations_train and len(self.segmentations_train) - 1 >= num:
                self.source_data['segmentations_train_' + label] = self.segmentations_train[num]

            if self.semantics_train and len(self.semantics_train) - 1 >= num:
                self.source_data['semantics_train_' + label] = self.semantics_train[num]

            if self.features_train and len(self.features_train) - 1 >= num:
                self.source_data['features_train_' + label] = self.features_train[num]

            if self.Elastix_Para:
                # First modality does not need to be registered
                if num > 0:
                    if len(self.Elastix_Para) > 1:
                        # Each modality has its own registration parameters
                        self.source_data['Elastix_Para_' + label] = self.Elastix_Para[num]
                    else:
                        # Use one fileset for all modalities
                        self.source_data['Elastix_Para_' + label] = self.Elastix_Para[0]

            # Add test data sources
            if self.images_test and len(self.images_test) - 1 >= num:
                self.source_data['images_test_' + label] = self.images_test[num]

            if self.masks_test and len(self.masks_test) - 1 >= num:
                self.source_data['mask_test_' + label] = self.masks_test[num]

            if self.metadata_test and len(self.metadata_test) - 1 >= num:
                self.source_data['metadata_test_' + label] = self.metadata_test[num]

            if self.segmentations_test and len(self.segmentations_test) - 1 >= num:
                self.source_data['segmentations_test_' + label] = self.segmentations_test[num]

            if self.semantics_test and len(self.semantics_test) - 1 >= num:
                self.source_data['semantics_test_' + label] = self.semantics_test[num]

            if self.features_test and len(self.features_test) - 1  >= num:
                self.source_data['features_test_' + label] = self.features_test[num]

            self.sink_data['segmentations_out_segmentix_train_' + label] = ("vfs://output/{}/Segmentations/seg_{}_segmentix_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name, label)
            self.sink_data['segmentations_out_elastix_train_' + label] = ("vfs://output/{}/Elastix/seg_{}_elastix_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name, label)
            self.sink_data['images_out_elastix_train_' + label] = ("vfs://output/{}/Elastix/im_{}_elastix_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name, label)
            self.sink_data['features_train_' + label] = ("vfs://output/{}/Features/features_{}_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name, label)

            if self.labels_test:
                self.sink_data['segmentations_out_segmentix_test_' + label] = ("vfs://output/Segmentations/{}/seg_{}_segmentix_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name, label)
                self.sink_data['segmentations_out_elastix_test_' + label] = ("vfs://output/{}/Elastix/seg_{}_elastix_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name, label)
                self.sink_data['images_out_elastix_test_' + label] = ("vfs://output/{}/Images/im_{}_elastix_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name, label)
                self.sink_data['features_test_' + label] = ("vfs://output/{}/Features/features_{}_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name, label)

            # Add elastix sinks if used
            if self.segmode:
                # Segmode is only non-empty if segmentations are provided
                if self.segmode == 'Register':
                    self.sink_data['transformations_train_' + label] = ("vfs://output/{}/Elastix/transformation_{}_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name, label)
                    if self.images_test or self.features_test:
                        self.sink_data['transformations_test_' + label] = ("vfs://output/{}/Elastix/transformation_{}_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name, label)

    def execute(self):
        """ Execute the network through the fastr.network.execute command. """
        # Draw and execute nwtwork
        self.network.draw(file_path=self.network.id + '.svg', draw_dimensions=True)
        self.network.execute(self.source_data, self.sink_data, execution_plugin=self.fastr_plugin, tmpdir=self.fastr_tmpdir)
        # self.network.execute(self.source_data, self.sink_data)


class Tools(object):
    '''
    This object can be used to create other pipelines besides the default
    Radiomics executions. Currently only includes a registratio pipeline.
    '''
    def __init__(self):
        self.Elastix = Elastix()
        self.Evaluate = Evaluate()
