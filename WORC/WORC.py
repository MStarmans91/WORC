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

import configparser
import fastr
import os
from random import randint
import addexceptions as WORCexceptions
import IOparser.config_WORC as config_io
from tools.Elastix import Elastix
from tools.Evaluate import Evaluate


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
            self.fastr_tmpdir = os.path.join(fastr.config.mounts['tmp'], 'fastr' + str(name))
        else:
            self.fastr_tmpdir = os.path.join(fastr.config.mounts['tmp'], name)
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
        config['General']['PCE'] = 'False'  # We do not yet provide this module
        config['General']['FeatureCalculator'] = 'CalcFeatures'
        config['General']['Preprocessing'] = 'PreProcess'
        config['General']['RegistrationNode'] = "Elastix"
        config['General']['TransformationNode'] = "Transformix"

        # PREDICT General Settings: only applies when using PREDICT
        config['PREDICTGeneral'] = dict()
        config['PREDICTGeneral']['Joblib_ncores'] = '4'
        config['PREDICTGeneral']['Joblib_backend'] = 'multiprocessing'
        config['PREDICTGeneral']['tempsave'] = 'False'

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
        config['ImageFeatures'] = dict()
        config['ImageFeatures']['orientation'] = 'True'
        config['ImageFeatures']['texture'] = 'all'
        config['ImageFeatures']['coliage'] = 'False'
        config['ImageFeatures']['vessel'] = 'False'
        config['ImageFeatures']['log'] = 'False'
        config['ImageFeatures']['phase'] = 'False'

        ## Parameter settings for PREDICT feature calculation
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

        # PREDICT - Feature selection
        config['Featsel'] = dict()
        config['Featsel']['Variance'] = 'True'
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

        # PREDICT - Gridsearch options
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

        # PREDICT - Feature imputation
        config['Imputation'] = dict()
        config['Imputation']['use'] = 'False'
        config['Imputation']['strategy'] = 'mean'
        config['Imputation']['n_neighbors'] = '5'

        # PREDICT - Classification
        config['Classification'] = dict()
        config['Classification']['fastr'] = 'False'
        config['Classification']['fastr_plugin'] = self.fastr_plugin
        config['Classification']['classifier'] = 'SVM'
        config['Classification']['Kernel'] = 'polynomial'

        config['CrossValidation'] = dict()
        config['CrossValidation']['N_iterations'] = '100'
        config['CrossValidation']['test_size'] = '0.2'

        # PREDICT - Options for the labels that are used (not only genetics)
        config['Genetics'] = dict()
        config['Genetics']['label_names'] = 'Label1, Label2'
        config['Genetics']['modus'] = 'singlelabel'
        config['Genetics']['url'] = 'WIP'
        config['Genetics']['projectID'] = 'WIP'

        # PREDICT - Hyperparameter optimization options
        config['HyperOptimization'] = dict()
        config['HyperOptimization']['scoring_method'] = 'f1_weighted'
        config['HyperOptimization']['test_size'] = '0.15'
        config['HyperOptimization']['N_iterations'] = '10000'
        config['HyperOptimization']['n_jobspercore'] = '2000'  # only relevant when using fastr in classification

        # PREDICT - Feature scaling options
        config['FeatureScaling'] = dict()
        config['FeatureScaling']['scale_features'] = 'True'
        config['FeatureScaling']['scaling_method'] = 'z_score'

        # PREDICT - Sample processing options
        config['SampleProcessing'] = dict()
        config['SampleProcessing']['SMOTE'] = 'True'
        config['SampleProcessing']['SMOTE_ratio'] = '1.0'
        config['SampleProcessing']['SMOTE_neighbors'] = '10'
        config['SampleProcessing']['Oversampling'] = 'False'

        # PREDICT - Ensemble options
        config['Ensemble'] = dict()
        config['Ensemble']['Use'] = 'False'  # Still WIP

        # BUG: the FASTR XNAT plugin can only retreive folders. We therefore need to add the filenames of the resources manually
        # This should be fixed from fastr > 2.0.0: need to update.
        config['FASTR_bugs'] = dict()
        config['FASTR_bugs']['images'] = 'image.nii.gz'
        config['FASTR_bugs']['segmentations'] = 'mask.nii.gz'

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

        if wtype == 'training':
            self.build_training()
            self.wtype = wtype
        elif wtype == 'testing':
            self.build_testing()
            self.wtype = wtype

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
                self.network = fastr.Network('WORC_' + self.name)

                # BUG: We currently use the first configuration as general config
                image_types = list()
                for c in range(len(self.configs)):
                    if type(self.configs[c]) == str:
                        # Probably, c is a configuration file
                        self.configs[c] = config_io.load_config(self.configs[c])
                    image_types.append(self.configs[c]['ImageFeatures']['image_type'])

                # Classification tool and label source
                self.network.source_patientclass_train = self.network.create_source('PatientInfoFile', id_='patientclass_train', nodegroup='pctrain')
                if self.labels_test:
                    self.network.source_patientclass_test = self.network.create_source('PatientInfoFile', id_='patientclass_test', nodegroup='pctest')

                self.network.classify = self.network.create_node('TrainClassifier', memory='12G', id_='classify')

                # Outputs
                self.network.sink_classification = self.network.create_sink('HDF5', id_='classification')
                self.network.sink_performance = self.network.create_sink('JsonFile', id_='performance')

                # Links
                self.network.source_class_config = self.network.create_source('ParameterFile', id_='config_classification', nodegroup='conf')
                # self.network.source_class_parameters = self.network.create_source('JsonFile', id_='parameters_classification')
                # self.network.classify.inputs['parameters'] = self.network.source_class_parameters.output
                self.network.link_class_1 = self.network.create_link(self.network.source_class_config.output, self.network.classify.inputs['config'][0])
                self.network.link_class_2 = self.network.create_link(self.network.source_patientclass_train.output, self.network.classify.inputs['patientclass_train'])
                self.network.link_class_1.collapse = 'conf'
                self.network.link_class_2.collapse = 'pctrain'

                if self.images_test or self.features_test:
                    # FIXME: the naming here is ugly
                    self.network.link_class_3 = self.network.create_link(self.network.source_patientclass_test.output, self.network.classify.inputs['patientclass_test'])
                    self.network.link_class_3.collapse = 'pctest'

                self.network.sink_classification.input = self.network.classify.outputs['classification']
                self.network.sink_performance.input = self.network.classify.outputs['performance']

                if not self.features_train:
                    # Create nodes to compute features
                    self.network.sources_parameters = dict()

                    self.network.calcfeatures_train = dict()
                    self.network.preprocessing_train = dict()
                    self.network.sources_images_train = dict()
                    self.network.sinks_features_train = dict()
                    self.network.converters_im_train = dict()
                    self.network.converters_seg_train = dict()
                    self.network.links_C1_train = dict()

                    if self.images_test or self.features_test:
                        # A test set is supplied, for which nodes also need to be created
                        self.network.preprocessing_test = dict()
                        self.network.calcfeatures_test = dict()
                        self.network.sources_images_test = dict()
                        self.network.sinks_features_test = dict()
                        self.network.converters_im_test = dict()
                        self.network.converters_seg_test = dict()
                        self.network.links_C1_test = dict()

                    # Check which nodes are necessary
                    if not self.segmentations_train:
                        message = "No automatic segmentation method is yet implemented."
                        raise WORCexceptions.WORCNotImplementedError(message)

                    elif len(self.segmentations_train) == len(image_types):
                        # Segmentations provided
                        self.network.sources_segmentations_train = dict()
                        self.network.sources_segmentations_test = dict()
                        self.segmode = 'Provided'

                    elif len(self.segmentations_train) == 1:
                        # Assume segmentations need to be registered to other modalities
                        self.network.sources_segmentation = dict()
                        self.segmode = 'Register'

                        self.network.source_Elastix_Parameters = dict()
                        self.network.elastix_nodes_train = dict()
                        self.network.transformix_seg_nodes_train = dict()
                        self.network.sources_segmentations_train = dict()
                        self.network.sinks_transformations_train = dict()
                        self.network.sinks_segmentations_elastix_train = dict()
                        self.network.sinks_images_elastix_train = dict()
                        self.network.converters_seg_train = dict()
                        self.network.edittransformfile_nodes_train = dict()
                        self.network.transformix_im_nodes_train = dict()

                        self.network.elastix_nodes_test = dict()
                        self.network.transformix_seg_nodes_test = dict()
                        self.network.sources_segmentations_test = dict()
                        self.network.sinks_transformations_test = dict()
                        self.network.sinks_segmentations_elastix_test = dict()
                        self.network.sinks_images_elastix_test = dict()
                        self.network.converters_seg_test = dict()
                        self.network.edittransformfile_nodes_test = dict()
                        self.network.transformix_im_nodes_test = dict()
                        pass

                    # BUG: We assume that first type defines if we use segmentix
                    if self.configs[0]['General']['Segmentix'] == 'True':
                        # Use the segmentix toolbox for segmentation processing
                        self.network.sinks_segmentations_segmentix_train = dict()
                        self.network.sources_masks_train = dict()
                        self.network.converters_masks_train = dict()
                        self.network.nodes_segmentix_train = dict()

                        if self.images_test or self.features_test:
                            # Also use segmentix on the tes set
                            self.network.sinks_segmentations_segmentix_test = dict()
                            self.network.sources_masks_test = dict()
                            self.network.converters_masks_test = dict()
                            self.network.nodes_segmentix_test = dict()

                    if self.semantics_train:
                        # Semantic features are supplied
                        self.network.sources_semantics_train = dict()

                    if self.metadata_train:
                        # Metadata to extract patient features from is supplied
                        self.network.sources_metadata_train = dict()

                    if self.semantics_test:
                        # Semantic features are supplied
                        self.network.sources_semantics_test = dict()

                    if self.metadata_test:
                        # Metadata to extract patient features from is supplied
                        self.network.sources_metadata_test = dict()

                    # Create a part of the pipeline for each modality
                    self.modlabels = list()
                    for nmod, mod in enumerate(image_types):
                        # Create label for each modality/image
                        num = 0
                        label = mod + '_' + str(num)
                        while label in self.network.calcfeatures_train.keys():
                            # if label already exists, add number to label
                            num += 1
                            label = mod + '_' + str(num)
                        self.modlabels.append(label)

                        # Create required sources and sinks
                        self.network.sources_parameters[label] = self.network.create_source('ParameterFile', id_='parameters_' + label)
                        self.network.sources_images_train[label] = self.network.create_source('ITKImageFile', id_='images_train_' + label, nodegroup='train')
                        self.network.sinks_features_train[label] = self.network.create_sink('HDF5', id_='features_train_' + label)
                        if self.images_test or self.features_test:
                            self.network.sources_images_test[label] = self.network.create_source('ITKImageFile', id_='images_test_' + label, nodegroup='test')
                            self.network.sinks_features_test[label] = self.network.create_sink('HDF5', id_='features_test_' + label)

                        if self.metadata_train and len(self.metadata_train) >= nmod + 1:
                            self.network.sources_metadata_train[label] = self.network.create_source('DicomImageFile', id_='metadata_train_' + label, nodegroup='train')

                        if self.metadata_test and len(self.metadata_test) >= nmod + 1:
                            self.network.sources_metadata_test[label] = self.network.create_source('DicomImageFile', id_='metadata_test_' + label, nodegroup='test')

                        if self.masks_train and len(self.masks_train) >= nmod + 1:
                            # Create mask source and convert
                            self.network.sources_masks_train[label] = self.network.create_source('ITKImageFile', id_='mask_train_' + label, nodegroup='train')
                            self.network.converters_masks_train[label] = self.network.create_node('WORCCastConvert', memory='4G', id_='convert_mask_train_' + label, nodegroup='train')
                            self.network.converters_masks_train[label].inputs['image'] = self.network.sources_masks_train[label].output

                        if self.masks_test and len(self.masks_test) >= nmod + 1:
                            # Create mask source and convert
                            self.network.sources_masks_test[label] = self.network.create_source('ITKImageFile', id_='mask_test_' + label, nodegroup='test')
                            self.network.converters_masks_test[label] = self.network.create_node('WORCCastConvert', memory='4G', id_='convert_mask_test_' + label, nodegroup='test')
                            self.network.converters_masks_test[label].inputs['image'] = self.network.sources_masks_test[label].output

                        # First convert the images
                        if any(modality in mod for modality in ['MR', 'CT', 'MG', 'PET']):
                            # Use ITKTools PXCastConvet for converting image formats
                            self.network.converters_im_train[label] = self.network.create_node('WORCCastConvert', memory='4G', id_='convert_im_train_' + label)
                            if self.images_test or self.features_test:
                                self.network.converters_im_test[label] = self.network.create_node('WORCCastConvert', memory='4G', id_='convert_im_test_' + label)

                        elif 'DTI' in mod:
                            # TODO: This reader is currently missing
                            self.network.converters_im_train[label] = self.network.create_node('DTIreader', memory='4G', id_='convert_im_train_' + label)
                            if self.images_test or self.features_test:
                                self.network.converters_im_test[label] = self.network.create_node('DTIreader', memory='4G', id_='convert_im_test_' + label)

                        else:
                            raise WORCexceptions.WORCTypeError(('No valid image type for modality {}: {} provided.').format(str(nmod), mod))

                        # Create required links
                        self.network.converters_im_train[label].inputs['image'] = self.network.sources_images_train[label].output
                        if self.images_test or self.features_test:
                            self.network.converters_im_test[label].inputs['image'] = self.network.sources_images_test[label].output

                        # -----------------------------------------------------
                        # Preprocessing
                        # Create nodes
                        preprocess_node = str(self.configs[nmod]['General']['Preprocessing'])
                        self.network.preprocessing_train[label] = self.network.create_node(preprocess_node, memory='4G', id_='preprocessing_train_' + label)
                        if self.images_test or self.features_test:
                            self.network.preprocessing_test[label] = self.network.create_node(preprocess_node, memory='4G', id_='preprocessing_test_' + label)

                        # Create required links
                        self.network.preprocessing_train[label].inputs['parameters'] = self.network.sources_parameters[label].output
                        self.network.preprocessing_train[label].inputs['image'] = self.network.converters_im_train[label].outputs['image']

                        if self.images_test or self.features_test:
                            self.network.preprocessing_test[label].inputs['parameters'] = self.network.sources_parameters[label].output
                            self.network.preprocessing_test[label].inputs['image'] = self.network.converters_im_test[label].outputs['image']

                        if self.metadata_train and len(self.metadata_train) >= nmod + 1:
                            self.network.preprocessing_train[label].inputs['metadata'] = self.network.sources_metadata_train[label].output

                        if self.metadata_test and len(self.metadata_test) >= nmod + 1:
                            self.network.preprocessing_test[label].inputs['metadata'] = self.network.sources_metadata_test[label].output

                        # -----------------------------------------------------
                        # Create a feature calculator node
                        calcfeat_node = str(self.configs[nmod]['General']['FeatureCalculator'])
                        self.network.calcfeatures_train[label] = self.network.create_node(calcfeat_node, memory='14G', id_='calcfeatures_train_' + label)
                        if self.images_test or self.features_test:
                            self.network.calcfeatures_test[label] = self.network.create_node(calcfeat_node, memory='14G', id_='calcfeatures_test_' + label)

                        # Create required links
                        self.network.calcfeatures_train[label].inputs['parameters'] = self.network.sources_parameters[label].output
                        self.network.calcfeatures_train[label].inputs['image'] = self.network.preprocessing_train[label].outputs['image']

                        if self.images_test or self.features_test:
                            self.network.calcfeatures_test[label].inputs['parameters'] = self.network.sources_parameters[label].output
                            self.network.calcfeatures_test[label].inputs['image'] = self.network.preprocessing_test[label].outputs['image']

                        if self.metadata_train and len(self.metadata_train) >= nmod + 1:
                            self.network.calcfeatures_train[label].inputs['metadata'] = self.network.sources_metadata_train[label].output

                        if self.metadata_train and len(self.metadata_test) >= nmod + 1:
                            self.network.calcfeatures_train[label].inputs['metadata'] = self.network.sources_metadata_train[label].output

                        if self.semantics_train and len(self.semantics_train) >= nmod + 1:
                            self.network.sources_semantics_train[label] = self.network.create_source('CSVFile', id_='semantics_train_' + label)
                            self.network.calcfeatures_train[label].inputs['semantics'] = self.network.sources_semantics_train[label].output

                        if self.semantics_test and len(self.semantics_test) >= nmod + 1:
                            self.network.sources_semantics_test[label] = self.network.create_source('CSVFile', id_='semantics_test_' + label)
                            self.network.calcfeatures_test[label].inputs['semantics'] = self.network.sources_semantics_test[label].output

                        if self.segmode == 'Provided':
                            # Segmentation -----------------------------------------------------
                            # Use the provided segmantions for each modality
                            self.network.sources_segmentations_train[label] = self.network.create_source('ITKImageFile', id_='segmentations_train_' + label, nodegroup='train')
                            self.network.converters_seg_train[label] = self.network.create_node('WORCCastConvert', memory='4G', id_='convert_seg_train_' + label)
                            self.network.converters_seg_train[label].inputs['image'] = self.network.sources_segmentations_train[label].output

                            if self.images_test or self.features_test:
                                self.network.sources_segmentations_test[label] = self.network.create_source('ITKImageFile', id_='segmentations_test_' + label, nodegroup='test')
                                self.network.converters_seg_test[label] = self.network.create_node('WORCCastConvert', memory='4G', id_='convert_seg_test_' + label)
                                self.network.converters_seg_test[label].inputs['image'] = self.network.sources_segmentations_test[label].output

                        elif self.segmode == 'Register':
                            # Registration nodes -----------------------------------------------------
                            # Align segmentation of first modality to others using registration with Elastix

                            # Create sources and converter for only for the given segmentation, which should be on the first modality
                            if nmod == 0:
                                self.network.sources_segmentations_train[label] = self.network.create_source('ITKImageFile', id_='segmentations_train_' + label, nodegroup='input')
                                self.network.converters_seg_train[label] = self.network.create_node('WORCCastConvert', memory='4G', id_='convert_seg_train_' + label)
                                self.network.converters_seg_train[label].inputs['image'] = self.network.sources_segmentations_train[label].output

                                if self.images_test or self.features_test:
                                    self.network.sources_segmentations_test[label] = self.network.create_source('ITKImageFile', id_='segmentations_test_' + label, nodegroup='input')
                                    self.network.converters_seg_test[label] = self.network.create_node('WORCCastConvert', memory='4G', id_='convert_seg_test_' + label)
                                    self.network.converters_seg_test[label].inputs['image'] = self.network.sources_segmentations_test[label].output

                            # Assume provided segmentation is on first modality
                            if nmod > 0:
                                # Use elastix and transformix for registration
                                # NOTE: Assume elastix node type is on first configuration
                                elastix_node = str(self.configs[0]['General']['RegistrationNode'])
                                transformix_node = str(self.configs[0]['General']['TransformationNode'])
                                self.network.elastix_nodes_train[label] = self.network.create_node(elastix_node, id_='elastix_train_' + label)
                                self.network.transformix_seg_nodes_train[label] = self.network.create_node(transformix_node, id_='transformix_seg_train_' + label)
                                self.network.transformix_im_nodes_train[label] = self.network.create_node(transformix_node, id_='transformix_im_train_' + label)

                                if self.images_test or self.features_test:
                                    self.network.elastix_nodes_test[label] = self.network.create_node(elastix_node, id_='elastix_test_' + label)
                                    self.network.transformix_seg_nodes_test[label] = self.network.create_node(transformix_node, id_='transformix_seg_test_' + label)
                                    self.network.transformix_im_nodes_test[label] = self.network.create_node(transformix_node, id_='transformix_im_test_' + label)

                                # Create sources_segmentation
                                # M1 = moving, others = fixed
                                self.network.elastix_nodes_train[label].inputs['fixed_image'] = self.network.converters_im_train[label].outputs['image']
                                self.network.elastix_nodes_train[label].inputs['moving_image'] = self.network.converters_im_train[self.modlabels[0]].outputs['image']

                                # Add node that copies metadata from the image to the segmentation if required
                                if self.CopyMetadata:
                                    # Copy metadata from the image which was registered to the segmentation, if it is not created yet
                                    if not hasattr(self.network, "copymetadata_nodes_train"):
                                        # NOTE: Do this for first modality, as we assume segmentation is on that one
                                        self.network.copymetadata_nodes_train = dict()
                                        self.network.copymetadata_nodes_train[self.modlabels[0]] = self.network.create_node("CopyMetadata", id_='CopyMetadata_train_' + self.modlabels[0])
                                        self.network.copymetadata_nodes_train[self.modlabels[0]].inputs["source"] = self.network.converters_im_train[self.modlabels[0]].outputs['image']
                                        self.network.copymetadata_nodes_train[self.modlabels[0]].inputs["destination"] = self.network.converters_seg_train[self.modlabels[0]].outputs['image']
                                    self.network.transformix_seg_nodes_train[label].inputs['image'] = self.network.copymetadata_nodes_train[self.modlabels[0]].outputs['output']
                                else:
                                    self.network.transformix_seg_nodes_train[label].inputs['image'] = self.network.converters_seg_train[self.modlabels[0]].outputs['image']

                                if self.images_test or self.features_test:
                                    self.network.elastix_nodes_test[label].inputs['fixed_image'] = self.network.converters_im_test[label].outputs['image']
                                    self.network.elastix_nodes_test[label].inputs['moving_image'] = self.network.converters_im_test[self.modlabels[0]].outputs['image']

                                    if self.CopyMetadata:
                                        # Copy metadata from the image which was registered to the segmentation
                                        if not hasattr(self.network, "copymetadata_nodes_test"):
                                            # NOTE: Do this for first modality, as we assume segmentation is on that one
                                            self.network.copymetadata_nodes_test = dict()
                                            self.network.copymetadata_nodes_test[self.modlabels[0]] = self.network.create_node("CopyMetadata", id_='CopyMetadata_test_' + self.modlabels[0])
                                            self.network.copymetadata_nodes_test[self.modlabels[0]].inputs["source"] = self.network.converters_im_test[self.modlabels[0]].outputs['image']
                                            self.network.copymetadata_nodes_test[self.modlabels[0]].inputs["destination"] = self.network.converters_seg_test[self.modlabels[0]].outputs['image']
                                        self.network.transformix_seg_nodes_test[label].inputs['image'] = self.network.copymetadata_nodes_test[self.modlabels[0]].outputs['output']
                                    else:
                                        self.network.transformix_seg_nodes_test[label].inputs['image'] = self.network.converters_seg_test[self.modlabels[0]].outputs['image']

                                # Apply registration to input modalities
                                self.network.source_Elastix_Parameters[label] = self.network.create_source('ElastixParameterFile', id_='Elastix_Para_' + label, nodegroup='elpara')
                                self.link_elparam_train = self.network.create_link(self.network.source_Elastix_Parameters[label].output,
                                                                           self.network.elastix_nodes_train[label].inputs['parameters'])
                                self.link_elparam_train.collapse = 'elpara'

                                if self.images_test or self.features_test:
                                    self.link_elparam_test = self.network.create_link(self.network.source_Elastix_Parameters[label].output,
                                                                               self.network.elastix_nodes_test[label].inputs['parameters'])
                                    self.link_elparam_test.collapse = 'elpara'

                                if self.masks_train:
                                    self.network.elastix_nodes_train[label].inputs['fixed_mask'] = self.network.converters_masks_train[label].outputs['image']
                                    self.network.elastix_nodes_train[label].inputs['moving_mask'] = self.network.converters_masks_train[self.modlabels[0]].outputs['image']

                                if self.images_test or self.features_test:
                                    if self.masks_test:
                                        self.network.elastix_nodes_test[label].inputs['fixed_mask'] = self.network.converters_masks_test[label].outputs['image']
                                        self.network.elastix_nodes_test[label].inputs['moving_mask'] = self.network.converters_masks_test[self.modlabels[0]].outputs['image']

                                # Change the FinalBSpline Interpolation order to 0 as required for binarie images: see https://github.com/SuperElastix/elastix/wiki/FAQ
                                self.network.edittransformfile_nodes_train[label] = self.network.create_node('EditElastixTransformFile', id_='EditElastixTransformFile' + label)
                                self.network.edittransformfile_nodes_train[label].inputs['set'] = ["FinalBSplineInterpolationOrder=0"]
                                self.network.edittransformfile_nodes_train[label].inputs['transform'] = self.network.elastix_nodes_train[label].outputs['transform'][-1]

                                if self.images_test or self.features_test:
                                    self.network.edittransformfile_nodes_test[label] = self.network.create_node('EditElastixTransformFile', id_='EditElastixTransformFile' + label)
                                    self.network.edittransformfile_nodes_test[label].inputs['set'] = ["FinalBSplineInterpolationOrder=0"]
                                    self.network.edittransformfile_nodes_test[label].inputs['transform'] = self.network.elastix_nodes_test[label].outputs['transform'][-1]

                                # Link data and transformation to transformix and source
                                self.network.transformix_seg_nodes_train[label].inputs['transform'] = self.network.edittransformfile_nodes_train[label].outputs['transform']
                                self.network.calcfeatures_train[label].inputs['segmentation'] = self.network.transformix_seg_nodes_train[label].outputs['image']

                                self.network.transformix_im_nodes_train[label].inputs['transform'] = self.network.elastix_nodes_train[label].outputs['transform'][-1]
                                self.network.transformix_im_nodes_train[label].inputs['image'] = self.network.converters_im_train[self.modlabels[0]].outputs['image']

                                if self.images_test or self.features_test:
                                    self.network.transformix_seg_nodes_test[label].inputs['transform'] = self.network.edittransformfile_nodes_test[label].outputs['transform']
                                    self.network.calcfeatures_test[label].inputs['segmentation'] = self.network.transformix_seg_nodes_test[label] .outputs['image']

                                    self.network.transformix_im_nodes_test[label].inputs['transform'] = self.network.elastix_nodes_test[label].outputs['transform'][-1]
                                    self.network.transformix_im_nodes_test[label].inputs['image'] = self.network.converters_im_test[self.modlabels[0]].outputs['image']

                                # Save output
                                self.network.sinks_transformations_train[label] = self.network.create_sink('ElastixTransformFile', id_='transformations_train_' + label)
                                self.network.sinks_segmentations_elastix_train[label] = self.network.create_sink('ITKImageFile', id_='segmentations_out_elastix_train_' + label)
                                self.network.sinks_images_elastix_train[label] = self.network.create_sink('ITKImageFile', id_='images_out_elastix_train_' + label)
                                self.network.sinks_transformations_train[label].input = self.network.elastix_nodes_train[label].outputs['transform']
                                self.network.sinks_segmentations_elastix_train[label].input = self.network.transformix_seg_nodes_train[label].outputs['image']
                                self.network.sinks_images_elastix_train[label].input = self.network.transformix_im_nodes_train[label].outputs['image']

                                if self.images_test or self.features_test:
                                    self.network.sinks_transformations_test[label] = self.network.create_sink('ElastixTransformFile', id_='transformations_test_' + label)
                                    self.network.sinks_segmentations_elastix_test[label] = self.network.create_sink('ITKImageFile', id_='segmentations_out_elastix_test_' + label)
                                    self.network.sinks_images_elastix_test[label] = self.network.create_sink('ITKImageFile', id_='images_out_elastix_test_' + label)
                                    self.network.sinks_transformations_elastix_test[label].input = self.network.elastix_nodes_test[label].outputs['transform']
                                    self.network.sinks_segmentations_elastix_test[label].input = self.network.transformix_seg_nodes_test[label].outputs['image']
                                    self.network.sinks_images_elastix_test[label].input = self.network.transformix_im_nodes_test[label].outputs['image']

                        if self.configs[nmod]['General']['Segmentix'] == 'True':
                            # Segmentix nodes -----------------------------------------------------
                            # Use segmentix node to convert input segmentation into correct contour
                            if label not in self.network.sinks_segmentations_segmentix_train:
                                self.network.sinks_segmentations_segmentix_train[label] = self.network.create_sink('ITKImageFile', id_='segmentations_out_segmentix_train_' + label)

                            self.network.nodes_segmentix_train[label] = self.network.create_node('Segmentix', memory='6G', id_='segmentix_train_' + label)
                            if hasattr(self.network, 'transformix_seg_nodes_train'):
                                if label in self.network.transformix_seg_nodes_train.keys():
                                    # Use output of registration in segmentix
                                    self.network.nodes_segmentix_train[label].inputs['segmentation_in'] = self.network.transformix_seg_nodes_train[label].outputs['image']
                                else:
                                    # Use original segmentation
                                    self.network.nodes_segmentix_train[label].inputs['segmentation_in'] = self.network.converters_seg_train[label].outputs['image']
                            else:
                                # Use original segmentation
                                self.network.nodes_segmentix_train[label].inputs['segmentation_in'] = self.network.converters_seg_train[label].outputs['image']

                            self.network.nodes_segmentix_train[label].inputs['parameters'] = self.network.sources_parameters[label].output
                            self.network.calcfeatures_train[label].inputs['segmentation'] = self.network.nodes_segmentix_train[label].outputs['segmentation_out']
                            self.network.sinks_segmentations_segmentix_train[label].input = self.network.nodes_segmentix_train[label].outputs['segmentation_out']

                            if self.images_test or self.features_test:
                                self.network.sinks_segmentations_segmentix_test[label] = self.network.create_sink('ITKImageFile', id_='segmentations_out_segmentix_test_' + label)
                                self.network.nodes_segmentix_test[label] = self.network.create_node('Segmentix', memory='6G', id_='segmentix_test_' + label)
                                if hasattr(self.network, 'transformix_seg_nodes_test'):
                                    if label in self.network.transformix_seg_nodes_test.keys():
                                        # Use output of registration in segmentix
                                        self.network.nodes_segmentix_test[label].inputs['segmentation_in'] = self.network.transformix_seg_nodes_test[label].outputs['image']
                                    else:
                                        # Use original segmentation
                                        self.network.nodes_segmentix_test[label].inputs['segmentation_in'] = self.network.converters_seg_test[label].outputs['image']
                                else:
                                    # Use original segmentation
                                    self.network.nodes_segmentix_test[label].inputs['segmentation_in'] = self.network.converters_seg_test[label].outputs['image']

                                self.network.nodes_segmentix_test[label].inputs['parameters'] = self.network.sources_parameters[label].output
                                self.network.calcfeatures_test[label].inputs['segmentation'] = self.network.nodes_segmentix_test[label].outputs['segmentation_out']
                                self.network.sinks_segmentations_segmentix_test[label].input = self.network.nodes_segmentix_test[label].outputs['segmentation_out']

                            if self.masks_train:
                                # Use masks
                                self.network.nodes_segmentix_train[label].inputs['mask'] = self.network.converters_masks_train[label].outputs['image']

                            if self.masks_test:
                                # Use masks
                                self.network.nodes_segmentix_test[label].inputs['mask'] = self.network.converters_masks_test[label].outputs['image']

                        else:
                            if self.segmode == 'Provided':
                                self.network.calcfeatures_train[label].inputs['segmentation'] = self.network.converters_seg_train[label].outputs['image']
                            elif self.segmode == 'Register':
                                if nmod > 0:
                                    self.network.calcfeatures_train[label].inputs['segmentation'] = self.network.transformix_seg_nodes_train[label].outputs['image']
                                else:
                                    self.network.calcfeatures_train[label].inputs['segmentation'] = self.network.converters_seg_train[label].outputs['image']

                            if self.images_test or self.features_test:
                                if self.segmode == 'Provided':
                                    self.network.calcfeatures_train[label].inputs['segmentation'] = self.network.converters_seg_train[label].outputs['image']
                                elif self.segmode == 'Register':
                                    if nmod > 0:
                                        self.network.calcfeatures_test[label].inputs['segmentation'] = self.network.transformix_seg_nodes_test[label] .outputs['image']
                                    else:
                                        self.network.calcfeatures_train[label].inputs['segmentation'] = self.network.converters_seg_train[label].outputs['image']

                        # Classification nodes -----------------------------------------------------
                        # Add the features from this modality to the classifier node input
                        self.network.links_C1_train[label] = self.network.classify.inputs['features_train'][str(label)] << self.network.calcfeatures_train[label].outputs['features']
                        self.network.links_C1_train[label].collapse = 'train'

                        if self.images_test or self.features_test:
                            # Add the features from this modality to the classifier node input
                            self.network.links_C1_test[label] = self.network.classify.inputs['features_test'][str(label)] << self.network.calcfeatures_test[label].outputs['features']
                            self.network.links_C1_test[label].collapse = 'test'

                        # Save output
                        self.network.sinks_features_train[label].input = self.network.calcfeatures_train[label].outputs['features']
                        if self.images_test or self.features_test:
                            self.network.sinks_features_test[label].input = self.network.calcfeatures_test[label].outputs['features']

                else:
                    # Features already provided: hence we can skip numerous nodes
                    self.network.sources_features_train = dict()
                    self.network.links_C1_train = dict()

                    if self.features_test:
                        self.network.sources_features_test = dict()
                        self.network.links_C1_test = dict()

                    # Create label for each modality/image
                    self.modlabels = list()
                    for num, mod in enumerate(image_types):
                        num = 0
                        label = mod + str(num)
                        while label in self.network.sources_features_train.keys():
                            # if label exists, add number to label
                            num += 1
                            label = mod + str(num)
                        self.modlabels.append(label)

                        # Create a node for the feature computation
                        self.network.sources_features_train[label] = self.network.create_source('HDF5', id_='features_train_' + label, nodegroup='train')

                        # Add the features from this modality to the classifier node input
                        self.network.links_C1_train[label] = self.network.classify.inputs['features_train'][str(label)] << self.network.sources_features_train[label].output
                        self.network.links_C1_train[label].collapse = 'train'

                        if self.features_test:
                            self.network.sources_features_test[label] = self.network.create_source('HDF5', id_='features_test_' + label, nodegroup='test')
                            self.network.links_C1_test[label] = self.network.classify.inputs['features_test'][str(label)] << self.network.sources_features_test[label].output
                            self.network.links_C1_test[label].collapse = 'test'

                if self.configs[num]['General']['PCE'] == 'True':
                    # NOTE: PCE feature is currently not open-source.
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
            cfile = os.path.join(fastr.config.mounts['tmp'], self.name, ("config_{}_{}.ini").format(self.name, num))
            if not os.path.exists(os.path.dirname(cfile)):
                os.makedirs(os.path.dirname(cfile))
            with open(cfile, 'w') as configfile:
                c.write(configfile)
            self.fastrconfigs.append(os.path.join("vfs://tmp/", self.name, ("config_{}_{}.ini").format(self.name, num)))

        # Generate gridsearch parameter files if required
        # TODO: We now use the first configuration for the classifier, but his needs to be separated from the rest per modality
        self.source_data['config_classification'] = self.fastrconfigs[0]

        # Set source and sink data
        self.source_data['patientclass_train'] = self.labels_train
        self.source_data['patientclass_test'] = self.labels_test

        self.sink_data['classification'] = ("vfs://output/{}/svm_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
        self.sink_data['performance'] = ("vfs://output/{}/performance_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
        if self.configs[0]['General']['PCE']:
            self.sink_data['PCE_mat'] = ("vfs://output/{}/PCE_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
            self.sink_data['SI_mat'] = ("vfs://output/{}/SI_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
            self.sink_data['TS_csv'] = ("vfs://output/{}/TS_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)

        # NOTE: Below bug should be fixed, need to check
        # BUG: this is a bug in the FASTR package. Workaround for nifti XNAT links using expansion of FASTR XNAT plugin.
        for num, im in enumerate(self.images_train):
            if im is not None and 'xnat' in im:
                if 'DICOM' not in im:
                    fastr.ioplugins['xnat']
                    # metalink = im_m1.replace('NIFTI','DICOM')
                    im = fastr.ioplugins['xnat'].expand_url(im)
                    imagename = self.configs[num]['FASTR_bugs']['images']
                    self.images_train[num] = {v[(v.find('subjects') + 9):v.find('/experiments')]: v + imagename + '?insecure=true' for k, v in im}

        for num, seg in enumerate(self.segmentations_train):
            if seg is not None and 'xnat' in seg:
                fastr.ioplugins['xnat']
                seg = fastr.ioplugins['xnat'].expand_url(seg)
                segname = self.configs[num]['FASTR_bugs']['segmentations']
                self.segmentations_train[num] = {v[(v.find('subjects') + 9):v.find('/experiments')]: v + segname + '?insecure=true' for k, v in seg}

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
        self.network.draw_network(self.network.id, draw_dimension=True)
        self.network.execute(self.source_data, self.sink_data, execution_plugin=self.fastr_plugin, tmpdir=self.fastr_tmpdir)


class Tools(object):
    '''
    This object can be used to create other pipelines besides the default
    Radiomics executions. Currently only includes a registratio pipeline.
    '''
    def __init__(self):
        self.Elastix = Elastix()
        self.Evaluate = Evaluate()
