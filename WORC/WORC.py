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

import configparser
import fastr
from fastr.api import ResourceLimit
import os
from random import randint
import graphviz
import WORC.addexceptions as WORCexceptions
import WORC.IOparser.config_WORC as config_io
from WORC.tools.Elastix import Elastix
from WORC.tools.Evaluate import Evaluate
from WORC.tools.Slicer import Slicer
from WORC.detectors.detectors import DebugDetector
from pathlib import Path
import yaml
import WORC.IOparser.file_io as io


class WORC(object):
    """Workflow for Optimal Radiomics Classification.

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

    def __init__(self, name='test'):
        """Initialize WORC object.

        Set the initial variables all to None, except for some defaults.

        Arguments:
            name: name of the nework (string, optional)

        """
        self.name = 'WORC_' + name

        # Initialize several objects
        self.configs = list()
        self.fastrconfigs = list()

        self.images_train = list()
        self.segmentations_train = list()
        self.semantics_train = list()
        self.labels_train = list()
        self.masks_train = list()
        self.masks_normalize_train = list()
        self.features_train = list()
        self.metadata_train = list()

        self.images_test = list()
        self.segmentations_test = list()
        self.semantics_test = list()
        self.labels_test = list()
        self.masks_test = list()
        self.masks_normalize_test = list()
        self.features_test = list()
        self.metadata_test = list()

        self.Elastix_Para = list()
        self.label_names = 'Label1, Label2'

        # Set some defaults, name
        self.fastr_plugin = 'LinearExecution'
        if name == '':
            name = [randint(0, 9) for p in range(0, 5)]
        self.fastr_tmpdir = os.path.join(fastr.config.mounts['tmp'], self.name)

        self.additions = dict()
        self.CopyMetadata = True
        self.segmode = []
        self._add_evaluation = False
        self.TrainTest = False

        # Memory settings for all fastr nodes
        self.fastr_memory_parameters = dict()
        self.fastr_memory_parameters['FeatureCalculator'] = '14G'
        self.fastr_memory_parameters['Classification'] = '12G'
        self.fastr_memory_parameters['WORCCastConvert'] = '4G'
        self.fastr_memory_parameters['Preprocessing'] = '4G'
        self.fastr_memory_parameters['Elastix'] = '4G'
        self.fastr_memory_parameters['Transformix'] = '4G'
        self.fastr_memory_parameters['Segmentix'] = '6G'
        self.fastr_memory_parameters['ComBat'] = '12G'

        if DebugDetector().do_detection():
            print(fastr.config)

    def defaultconfig(self):
        """Generate a configparser object holding all default configuration values.

        Returns:
            config: configparser configuration file

        """
        config = configparser.ConfigParser()
        config.optionxform = str

        # General configuration of WORC
        config['General'] = dict()
        config['General']['cross_validation'] = 'True'
        config['General']['Segmentix'] = 'True'
        config['General']['FeatureCalculators'] = '[predict/CalcFeatures:1.0, pyradiomics/Pyradiomics:1.0]'
        config['General']['Preprocessing'] = 'worc/PreProcess:1.0'
        config['General']['RegistrationNode'] = "'elastix4.8/Elastix:4.8'"
        config['General']['TransformationNode'] = "'elastix4.8/Transformix:4.8'"
        config['General']['Joblib_ncores'] = '1'
        config['General']['Joblib_backend'] = 'threading'
        config['General']['tempsave'] = 'False'
        config['General']['AssumeSameImageAndMaskMetadata'] = 'False'
        config['General']['ComBat'] = 'False'

        # Segmentix
        config['Segmentix'] = dict()
        config['Segmentix']['mask'] = 'subtract'
        config['Segmentix']['segtype'] = 'None'
        config['Segmentix']['segradius'] = '5'
        config['Segmentix']['N_blobs'] = '0'
        config['Segmentix']['fillholes'] = 'True'
        config['Segmentix']['remove_small_objects'] = 'False'
        config['Segmentix']['min_object_size'] = '2'

        # Preprocessing
        config['Normalize'] = dict()
        config['Normalize']['ROI'] = 'Full'
        config['Normalize']['ROIDetermine'] = 'Provided'
        config['Normalize']['ROIdilate'] = 'False'
        config['Normalize']['ROIdilateradius'] = '10'
        config['Normalize']['Method'] = 'z_score'

        # PREDICT - Feature calculation
        # Determine which features are calculated
        config['ImageFeatures'] = dict()
        config['ImageFeatures']['shape'] = 'True'
        config['ImageFeatures']['histogram'] = 'True'
        config['ImageFeatures']['orientation'] = 'True'
        config['ImageFeatures']['texture_Gabor'] = 'True'
        config['ImageFeatures']['texture_LBP'] = 'True'
        config['ImageFeatures']['texture_GLCM'] = 'True'
        config['ImageFeatures']['texture_GLCMMS'] = 'True'
        config['ImageFeatures']['texture_GLRLM'] = 'False'
        config['ImageFeatures']['texture_GLSZM'] = 'False'
        config['ImageFeatures']['texture_NGTDM'] = 'False'
        config['ImageFeatures']['coliage'] = 'False'
        config['ImageFeatures']['vessel'] = 'True'
        config['ImageFeatures']['log'] = 'True'
        config['ImageFeatures']['phase'] = 'True'

        # Parameter settings for PREDICT feature calculation
        # Defines only naming of modalities
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

        # PyRadiomics - Feature calculation
        # Addition to the above, specifically for PyRadiomics
        # Mostly based on specific MR Settings: see https://github.com/Radiomics/pyradiomics/blob/master/examples/exampleSettings/exampleMR_NoResampling.yaml
        config['PyRadiomics'] = dict()
        config['PyRadiomics']['geometryTolerance'] = '0.0001'
        config['PyRadiomics']['normalize'] = 'False'
        config['PyRadiomics']['normalizeScale'] = '100'
        config['PyRadiomics']['interpolator'] = 'sitkBSpline'
        config['PyRadiomics']['preCrop'] = 'True'
        config['PyRadiomics']['binCount'] = config['ImageFeatures']['GLCM_levels'] # BinWidth to sensitive for normalization, thus use binCount
        config['PyRadiomics']['force2D'] = 'False'
        config['PyRadiomics']['force2Ddimension'] = '0'  # axial slices, for coronal slices, use dimension 1 and for sagittal, dimension 2.
        config['PyRadiomics']['voxelArrayShift'] = '300'
        config['PyRadiomics']['Original'] = 'True'
        config['PyRadiomics']['Wavelet'] = 'False'
        config['PyRadiomics']['LoG'] = 'False'

        if config['General']['Segmentix'] == 'True':
            config['PyRadiomics']['label'] = '1'
        else:
            config['PyRadiomics']['label'] = '255'

        # Enabled PyRadiomics features
        config['PyRadiomics']['extract_firstorder'] = 'False'
        config['PyRadiomics']['extract_shape'] = 'True'
        config['PyRadiomics']['texture_GLCM'] = 'False'
        config['PyRadiomics']['texture_GLRLM'] = 'True'
        config['PyRadiomics']['texture_GLSZM'] = 'True'
        config['PyRadiomics']['texture_GLDM'] = 'True'
        config['PyRadiomics']['texture_NGTDM'] = 'True'

        # ComBat Feature Harmonization
        config['ComBat'] = dict()
        config['ComBat']['language'] = 'python'
        config['ComBat']['batch'] = 'Hospital'
        config['ComBat']['par'] = '1'
        config['ComBat']['eb'] = '1'
        config['ComBat']['per_feature'] = '0'
        config['ComBat']['excluded_features'] = 'sf_, of_, semf_, pf_'
        config['ComBat']['matlab'] = 'C:\\Program Files\\MATLAB\\R2015b\\bin\\matlab.exe'

        # Feature preprocessing before all below takes place
        config['FeatPreProcess'] = dict()
        config['FeatPreProcess']['Use'] = 'False'

        # Feature selection
        config['Featsel'] = dict()
        config['Featsel']['Variance'] = '1.0'
        config['Featsel']['GroupwiseSearch'] = 'True'
        config['Featsel']['SelectFromModel'] = '0.0'
        config['Featsel']['UsePCA'] = '0.25'
        config['Featsel']['PCAType'] = '95variance, 10, 50, 100'
        config['Featsel']['StatisticalTestUse'] = '0.25'
        config['Featsel']['StatisticalTestMetric'] = 'MannWhitneyU'
        config['Featsel']['StatisticalTestThreshold'] = '-3, 2.5'
        config['Featsel']['ReliefUse'] = '0.25'
        config['Featsel']['ReliefNN'] = '2, 4'
        config['Featsel']['ReliefSampleSize'] = '1, 1'
        config['Featsel']['ReliefDistanceP'] = '1, 3'
        config['Featsel']['ReliefNumFeatures'] = '25, 100'

        # Groupwise Featureselection options
        config['SelectFeatGroup'] = dict()
        config['SelectFeatGroup']['shape_features'] = 'True, False'
        config['SelectFeatGroup']['histogram_features'] = 'True, False'
        config['SelectFeatGroup']['orientation_features'] = 'True, False'
        config['SelectFeatGroup']['texture_Gabor_features'] = 'False'
        config['SelectFeatGroup']['texture_GLCM_features'] = 'True, False'
        config['SelectFeatGroup']['texture_GLDM_features'] = 'True, False'
        config['SelectFeatGroup']['texture_GLCMMS_features'] = 'True, False'
        config['SelectFeatGroup']['texture_GLRLM_features'] = 'True, False'
        config['SelectFeatGroup']['texture_GLSZM_features'] = 'True, False'
        config['SelectFeatGroup']['texture_GLDZM_features'] = 'True, False'
        config['SelectFeatGroup']['texture_NGTDM_features'] = 'True, False'
        config['SelectFeatGroup']['texture_NGLDM_features'] = 'True, False'
        config['SelectFeatGroup']['texture_LBP_features'] = 'True, False'
        config['SelectFeatGroup']['patient_features'] = 'False'
        config['SelectFeatGroup']['semantic_features'] = 'False'
        config['SelectFeatGroup']['coliage_features'] = 'False'
        config['SelectFeatGroup']['vessel_features'] = 'True, False'
        config['SelectFeatGroup']['phase_features'] = 'True, False'
        config['SelectFeatGroup']['fractal_features'] = 'True, False'
        config['SelectFeatGroup']['location_features'] = 'True, False'
        config['SelectFeatGroup']['rgrd_features'] = 'True, False'

        # Select features per toolbox, or simply all
        config['SelectFeatGroup']['toolbox'] = 'All, PREDICT, PyRadiomics'

        # Select original features, or after transformation of feature space
        config['SelectFeatGroup']['original_features'] = 'True'
        config['SelectFeatGroup']['wavelet_features'] = 'True, False'
        config['SelectFeatGroup']['log_features'] = 'True, False'

        # Feature imputation
        config['Imputation'] = dict()
        config['Imputation']['use'] = 'True'
        config['Imputation']['strategy'] = 'mean, median, most_frequent, constant, knn'
        config['Imputation']['n_neighbors'] = '5, 5'

        # Classification
        config['Classification'] = dict()
        config['Classification']['fastr'] = 'True'
        config['Classification']['fastr_plugin'] = self.fastr_plugin
        config['Classification']['classifiers'] = 'SVM, SVM, SVM, RF, LR, LDA, QDA, GaussianNB'
        config['Classification']['max_iter'] = '100000'
        config['Classification']['SVMKernel'] = 'poly, rbf, linear'
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
        config['CrossValidation']['fixed_seed'] = 'False'

        # Options for the object/patient labels that are used
        config['Labels'] = dict()
        config['Labels']['label_names'] = 'Label1, Label2'
        config['ComBat']['mod'] = config['Labels']['label_names']  # Variation due to label to predict should be maintained
        config['Labels']['modus'] = 'singlelabel'
        config['Labels']['url'] = 'WIP'
        config['Labels']['projectID'] = 'WIP'

        # Hyperparameter optimization options
        config['HyperOptimization'] = dict()
        config['HyperOptimization']['scoring_method'] = 'f1_weighted'
        config['HyperOptimization']['test_size'] = '0.15'
        config['HyperOptimization']['n_splits'] = '5'
        config['HyperOptimization']['N_iterations'] = '10000'
        config['HyperOptimization']['n_jobspercore'] = '2000'  # only relevant when using fastr in classification
        config['HyperOptimization']['maxlen'] = '100'
        config['HyperOptimization']['ranking_score'] = 'test_score'

        # Feature scaling options
        config['FeatureScaling'] = dict()
        config['FeatureScaling']['scale_features'] = 'True'
        config['FeatureScaling']['scaling_method'] = 'z_score'

        # Sample processing options
        config['SampleProcessing'] = dict()
        config['SampleProcessing']['SMOTE'] = 'True, False'
        config['SampleProcessing']['SMOTE_ratio'] = '1, 0'
        config['SampleProcessing']['SMOTE_neighbors'] = '5, 15'
        config['SampleProcessing']['Oversampling'] = 'False'

        # Ensemble options
        config['Ensemble'] = dict()
        config['Ensemble']['Use'] = '50'

        # Evaluation options
        config['Evaluation'] = dict()
        config['Evaluation']['OverfitScaler'] = 'False'

        # Bootstrap options
        config['Bootstrap'] = dict()
        config['Bootstrap']['Use'] = 'False'
        config['Bootstrap']['N_iterations'] = '100'

        return config

    def add_tools(self):
        """Add several tools to the WORC object."""
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
        if self.images_test or self.features_test:
            self.TrainTest = True

        if self.images_train or self.features_train:
            print('Building training network...')
            # We currently require labels for supervised learning
            if self.labels_train:
                if not self.configs:
                    print("No configuration given, assuming default")
                    if self.images_train:
                        self.configs = [self.defaultconfig()] * len(self.images_train)
                    else:
                        self.configs = [self.defaultconfig()] * len(self.features_train)
                self.network = fastr.create_network(self.name)

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

                memory = self.fastr_memory_parameters['Classification']
                self.classify = self.network.create_node('worc/TrainClassifier:1.0', tool_version='1.0', id='classify', resources=ResourceLimit(memory=memory))

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

                if self.TrainTest:
                    # FIXME: the naming here is ugly
                    self.link_class_3 = self.network.create_link(self.source_patientclass_test.output, self.classify.inputs['patientclass_test'])
                    self.link_class_3.collapse = 'pctest'

                self.sink_classification.input = self.classify.outputs['classification']
                self.sink_performance.input = self.classify.outputs['performance']

                if self.masks_normalize_train:
                    self.sources_masks_normalize_train = dict()

                if self.masks_normalize_test:
                    self.sources_masks_normalize_test = dict()

                # -----------------------------------------------------
                # Optionally, add ComBat Harmonization. Currently done
                # on full dataset, not in a cross-validation
                if self.configs[0]['General']['ComBat'] == 'True':
                    self.add_ComBat()

                if not self.features_train:
                    # Create nodes to compute features
                    # General
                    self.sources_parameters = dict()
                    self.source_config_pyradiomics = dict()
                    self.source_toolbox_name = dict()

                    # Training only
                    self.calcfeatures_train = dict()
                    self.featureconverter_train = dict()
                    self.preprocessing_train = dict()
                    self.sources_images_train = dict()
                    self.sinks_features_train = dict()
                    self.converters_im_train = dict()
                    self.converters_seg_train = dict()
                    self.links_C1_train = dict()

                    self.featurecalculators = dict()

                    if self.TrainTest:
                        # A test set is supplied, for which nodes also need to be created
                        self.calcfeatures_test = dict()
                        self.featureconverter_test = dict()
                        self.preprocessing_test = dict()
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
                        print('\t - Adding Elastix node for image registration.')
                        self.add_elastix_sourcesandsinks()
                        pass

                    else:
                        nseg = len(self.segmentations_train)
                        nim = len(image_types)
                        m = f'Length of segmentations for training is ' +\
                            f'{nseg}: should be equal to number of images' +\
                            f' ({nim}) or 1 when using registration.'
                        raise WORCexceptions.WORCValueError(m)

                    # BUG: We assume that first type defines if we use segmentix
                    if self.configs[0]['General']['Segmentix'] == 'True':
                        # Use the segmentix toolbox for segmentation processing
                        print('\t - Adding segmentix node for segmentation preprocessing.')
                        self.sinks_segmentations_segmentix_train = dict()
                        self.sources_masks_train = dict()
                        self.converters_masks_train = dict()
                        self.nodes_segmentix_train = dict()

                        if self.TrainTest:
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
                        self.sources_parameters[label] = self.network.create_source('ParameterFile', id='config_' + label)
                        self.sources_images_train[label] = self.network.create_source('ITKImageFile', id='images_train_' + label, node_group='train')
                        if self.TrainTest:
                            self.sources_images_test[label] = self.network.create_source('ITKImageFile', id='images_test_' + label, node_group='test')

                        if self.metadata_train and len(self.metadata_train) >= nmod + 1:
                            self.sources_metadata_train[label] = self.network.create_source('DicomImageFile', id='metadata_train_' + label, node_group='train')

                        if self.metadata_test and len(self.metadata_test) >= nmod + 1:
                            self.sources_metadata_test[label] = self.network.create_source('DicomImageFile', id='metadata_test_' + label, node_group='test')

                        if self.masks_train and len(self.masks_train) >= nmod + 1:
                            # Create mask source and convert
                            self.sources_masks_train[label] = self.network.create_source('ITKImageFile', id='mask_train_' + label, node_group='train')
                            memory = self.fastr_memory_parameters['WORCCastConvert']
                            self.converters_masks_train[label] = self.network.create_node('worc/WORCCastConvert:0.3.2', tool_version='0.1', id='convert_mask_train_' + label, node_group='train', resources=ResourceLimit(memory=memory))
                            self.converters_masks_train[label].inputs['image'] = self.sources_masks_train[label].output

                        if self.masks_test and len(self.masks_test) >= nmod + 1:
                            # Create mask source and convert
                            self.sources_masks_test[label] = self.network.create_source('ITKImageFile', id='mask_test_' + label, node_group='test')
                            memory = self.fastr_memory_parameters['WORCCastConvert']
                            self.converters_masks_test[label] = self.network.create_node('worc/WORCCastConvert:0.3.2', tool_version='0.1', id='convert_mask_test_' + label, node_group='test', resources=ResourceLimit(memory=memory))
                            self.converters_masks_test[label].inputs['image'] = self.sources_masks_test[label].output

                        # First convert the images
                        if any(modality in mod for modality in ['MR', 'CT', 'MG', 'PET']):
                            # Use WORC PXCastConvet for converting image formats
                            memory = self.fastr_memory_parameters['WORCCastConvert']
                            self.converters_im_train[label] = self.network.create_node('worc/WORCCastConvert:0.3.2', tool_version='0.1', id='convert_im_train_' + label, resources=ResourceLimit(memory=memory))
                            if self.TrainTest:
                                self.converters_im_test[label] = self.network.create_node('worc/WORCCastConvert:0.3.2', tool_version='0.1', id='convert_im_test_' + label, resources=ResourceLimit(memory=memory))

                        else:
                            raise WORCexceptions.WORCTypeError(('No valid image type for modality {}: {} provided.').format(str(nmod), mod))

                        # Create required links
                        self.converters_im_train[label].inputs['image'] = self.sources_images_train[label].output
                        if self.TrainTest:
                            self.converters_im_test[label].inputs['image'] = self.sources_images_test[label].output

                        # -----------------------------------------------------
                        # Preprocessing
                        preprocess_node = str(self.configs[nmod]['General']['Preprocessing'])
                        print('\t - Adding preprocessing node for image preprocessing.')
                        self.add_preprocessing(preprocess_node, label, nmod)

                        # -----------------------------------------------------
                        # Feature calculation
                        feature_calculators =\
                            self.configs[nmod]['General']['FeatureCalculators']
                        feature_calculators = feature_calculators.strip('][').split(', ')
                        self.featurecalculators[label] = [f.split('/')[0] for f in feature_calculators]

                        # Add lists for feature calculation and converter objects
                        self.calcfeatures_train[label] = list()
                        self.featureconverter_train[label] = list()
                        if self.TrainTest:
                            self.calcfeatures_test[label] = list()
                            self.featureconverter_test[label] = list()

                        for f in feature_calculators:
                            print(f'\t - Adding feature calculation node: {f}.')
                            self.add_feature_calculator(f, label, nmod)

                        # -----------------------------------------------------
                        # Create the neccesary nodes for the segmentation
                        if self.segmode == 'Provided':
                            # Segmentation ----------------------------------------------------
                            # Use the provided segmantions for each modality
                            memory = self.fastr_memory_parameters['WORCCastConvert']
                            self.sources_segmentations_train[label] =\
                                self.network.create_source('ITKImageFile',
                                                           id='segmentations_train_' + label,
                                                           node_group='train')

                            self.converters_seg_train[label] =\
                                self.network.create_node('worc/WORCCastConvert:0.3.2',
                                                         tool_version='0.1',
                                                         id='convert_seg_train_' + label,
                                                         resources=ResourceLimit(memory=memory))

                            self.converters_seg_train[label].inputs['image'] =\
                                self.sources_segmentations_train[label].output

                            if self.TrainTest:
                                self.sources_segmentations_test[label] =\
                                    self.network.create_source('ITKImageFile',
                                                               id='segmentations_test_' + label,
                                                               node_group='test')

                                self.converters_seg_test[label] =\
                                    self.network.create_node('worc/WORCCastConvert:0.3.2',
                                                             tool_version='0.1',
                                                             id='convert_seg_test_' + label,
                                                             resources=ResourceLimit(memory=memory))

                                self.converters_seg_test[label].inputs['image'] =\
                                    self.sources_segmentations_test[label].output

                        elif self.segmode == 'Register':
                            # ---------------------------------------------
                            # Registration nodes: Align segmentation of first
                            # modality to others using registration ith Elastix
                            self.add_elastix(label, nmod)

                        # -----------------------------------------------------
                        # Optionally, add segmentix, the in-house segmentation
                        # processor of WORC
                        if self.configs[nmod]['General']['Segmentix'] == 'True':
                            self.add_segmentix(label, nmod)

                        else:
                            # Provide source or elastix segmentations to
                            # feature calculator
                            for i_node in range(len(self.calcfeatures_train[label])):
                                if self.segmode == 'Provided':
                                    self.calcfeatures_train[label][i_node].inputs['segmentation'] =\
                                        self.converters_seg_train[label].outputs['image']
                                elif self.segmode == 'Register':
                                    if nmod > 0:
                                        self.calcfeatures_train[label][i_node].inputs['segmentation'] =\
                                            self.transformix_seg_nodes_train[label].outputs['image']
                                    else:
                                        self.calcfeatures_train[label][i_node].inputs['segmentation'] =\
                                            self.converters_seg_train[label].outputs['image']

                                if self.TrainTest:
                                    if self.segmode == 'Provided':
                                        self.calcfeatures_test[label][i_node].inputs['segmentation'] =\
                                            self.converters_seg_test[label].outputs['image']
                                    elif self.segmode == 'Register':
                                        if nmod > 0:
                                            self.calcfeatures_test[label][i_node].inputs['segmentation'] =\
                                                self.transformix_seg_nodes_test[label].outputs['image']
                                        else:
                                            self.calcfeatures_test[label][i_node].inputs['segmentation'] =\
                                                self.converters_seg_test[label].outputs['image']

                        # -----------------------------------------------------
                        # Optionally, add ComBat Harmonization
                        if self.configs[0]['General']['ComBat'] == 'True':
                            # Link features to ComBat
                            self.links_Combat1_train[label] = list()
                            for i_node, fname in enumerate(self.featurecalculators[label]):
                                self.links_Combat1_train[label].append(self.ComBat.inputs['features_train'][f'{label}_{self.featurecalculators[label][i_node]}'] << self.featureconverter_train[label][i_node].outputs['feat_out'])
                                self.links_Combat1_train[label][i_node].collapse = 'train'

                            if self.TrainTest:
                                self.links_Combat1_test[label] = list()
                                for i_node, fname in enumerate(self.featurecalculators[label]):
                                    self.links_Combat1_test[label].append(self.ComBat.inputs['features_test'][f'{label}_{self.featurecalculators[label][i_node]}'] << self.featureconverter_test[label][i_node].outputs['feat_out'])
                                    self.links_Combat1_test[label][i_node].collapse = 'test'

                        # -----------------------------------------------------
                        # Classification nodes
                        # Add the features from this modality to the classifier node input
                        self.links_C1_train[label] = list()
                        self.sinks_features_train[label] = list()
                        if self.TrainTest:
                            self.links_C1_test[label] = list()
                            self.sinks_features_test[label] = list()

                        for i_node, fname in enumerate(self.featurecalculators[label]):
                            # Create sink for feature outputs
                            self.sinks_features_train[label].append(self.network.create_sink('HDF5', id='features_train_' + label + '_' + fname))

                            # Append features to the classification
                            if not self.configs[0]['General']['ComBat'] == 'True':
                                self.links_C1_train[label].append(self.classify.inputs['features_train'][f'{label}_{self.featurecalculators[label][i_node]}'] << self.featureconverter_train[label][i_node].outputs['feat_out'])
                                self.links_C1_train[label][i_node].collapse = 'train'

                            # Save output
                            self.sinks_features_train[label][i_node].input = self.featureconverter_train[label][i_node].outputs['feat_out']

                            # Similar for testing workflow
                            if self.TrainTest:
                                # Create sink for feature outputs
                                self.sinks_features_test[label].append(self.network.create_sink('HDF5', id='features_test_' + label + '_' + fname))

                                # Append features to the classification
                                if not self.configs[0]['General']['ComBat'] == 'True':
                                    self.links_C1_test[label].append(self.classify.inputs['features_test'][f'{label}_{self.featurecalculators[label][i_node]}'] << self.featureconverter_test[label][i_node].outputs['feat_out'])
                                    self.links_C1_test[label][i_node].collapse = 'test'

                                # Save output
                                self.sinks_features_test[label][i_node].input = self.featureconverter_test[label][i_node].outputs['feat_out']

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
                            # Create a node for the feature computation
                            self.sources_features_test[label] = self.network.create_source('HDF5', id='features_test_' + label, node_group='test')

                            # Add the features from this modality to the classifier node input
                            self.links_C1_test[label] = self.classify.inputs['features_test'][str(label)] << self.sources_features_test[label].output
                            self.links_C1_test[label].collapse = 'test'

            else:
                raise WORCexceptions.WORCIOError("Please provide labels.")
        else:
            raise WORCexceptions.WORCIOError("Please provide either images or features.")

    def add_ComBat(self):
        """Add ComBat harmonization to the network.

        Note: applied on all objects, not in a train-test or cross-val setting.
        """
        memory = self.fastr_memory_parameters['ComBat']
        self.ComBat =\
            self.network.create_node('combat/ComBat:1.0',
                                     tool_version='1.0',
                                     id='ComBat',
                                     resources=ResourceLimit(memory=memory))

        # Create sink for ComBat output
        self.sinks_features_train_ComBat = self.network.create_sink('HDF5', id='features_train_ComBat')

        # Create links for inputs
        self.link_combat_1 = self.network.create_link(self.source_class_config.output, self.ComBat.inputs['config'])
        self.link_combat_2 = self.network.create_link(self.source_patientclass_train.output, self.ComBat.inputs['patientclass_train'])
        self.link_combat_1.collapse = 'conf'
        self.link_combat_2.collapse = 'pctrain'
        self.links_Combat1_train = dict()
        self.links_Combat1_test = dict()

        # Link Combat output to both sink and classify node
        self.links_Combat_out_train = self.network.create_link(self.ComBat.outputs['features_train_out'], self.classify.inputs['features_train'])
        self.links_Combat_out_train.collapse = 'ComBat'
        self.sinks_features_train_ComBat.input = self.ComBat.outputs['features_train_out']

        if self.TrainTest:
            # Create sink for ComBat output
            self.sinks_features_test_ComBat = self.network.create_sink('HDF5', id='features_test_ComBat')

            # Create links for inputs
            self.link_combat_3 = self.network.create_link(self.source_patientclass_test.output, self.ComBat.inputs['patientclass_test'])
            self.link_combat_3.collapse = 'pctest'

            # Link Combat output to both sink and classify node
            self.links_Combat_out_test = self.network.create_link(self.ComBat.outputs['features_test_out'], self.classify.inputs['features_test'])
            self.links_Combat_out_test.collapse = 'ComBat'
            self.sinks_features_test_ComBat.input = self.ComBat.outputs['features_test_out']

    def add_preprocessing(self, preprocess_node, label, nmod):
        """Add nodes required for preprocessing of images."""
        memory = self.fastr_memory_parameters['Preprocessing']
        self.preprocessing_train[label] = self.network.create_node(preprocess_node, tool_version='1.0', id='preprocessing_train_' + label, resources=ResourceLimit(memory=memory))
        if self.TrainTest:
            self.preprocessing_test[label] = self.network.create_node(preprocess_node, tool_version='1.0', id='preprocessing_test_' + label, resources=ResourceLimit(memory=memory))

        # Create required links
        self.preprocessing_train[label].inputs['parameters'] = self.sources_parameters[label].output
        self.preprocessing_train[label].inputs['image'] = self.converters_im_train[label].outputs['image']

        if self.TrainTest:
            self.preprocessing_test[label].inputs['parameters'] = self.sources_parameters[label].output
            self.preprocessing_test[label].inputs['image'] = self.converters_im_test[label].outputs['image']

        if self.metadata_train and len(self.metadata_train) >= nmod + 1:
            self.preprocessing_train[label].inputs['metadata'] = self.sources_metadata_train[label].output

        if self.metadata_test and len(self.metadata_test) >= nmod + 1:
            self.preprocessing_test[label].inputs['metadata'] = self.sources_metadata_test[label].output

        # If there are masks to use in normalization, add them here
        if self.masks_normalize_train:
            self.sources_masks_normalize_train[label] = self.network.create_source('ITKImageFile', id='masks_normalize_train_' + label, node_group='train')
            self.preprocessing_train[label].inputs['mask'] = self.sources_masks_normalize_train[label].output

        if self.masks_normalize_test:
            self.sources_masks_normalize_test[label] = self.network.create_source('ITKImageFile', id='masks_normalize_test_' + label, node_group='test')
            self.preprocessing_test[label].inputs['mask'] = self.sources_masks_normalize_test[label].output

    def add_feature_calculator(self, calcfeat_node, label, nmod):
        """Add a feature calculation node to the network."""
        # Name of fastr node has to exclude some specific symbols, which
        # are used in the node name
        node_ID = '_'.join([calcfeat_node.replace(':', '_').replace('.', '_').replace('/', '_'),
                            label])

        memory = self.fastr_memory_parameters['FeatureCalculator']
        node_train =\
            self.network.create_node(calcfeat_node,
                                     tool_version='1.0',
                                     id='calcfeatures_train_' + node_ID,
                                     resources=ResourceLimit(memory=memory))

        if self.TrainTest:
            node_test =\
                self.network.create_node(calcfeat_node,
                                         tool_version='1.0',
                                         id='calcfeatures_test_' + node_ID,
                                         resources=ResourceLimit(memory=memory))

        # Check if we need to add pyradiomics specific sources
        if 'pyradiomics' in calcfeat_node.lower():
            # Add a config source
            self.source_config_pyradiomics[label] =\
                self.network.create_source('YamlFile',
                                           id='config_pyradiomics_' + label,
                                           node_group='train')

            # Add a format source, which we are going to set to a constant
            # And attach to the tool node
            self.source_format_pyradiomics =\
                self.network.create_constant('String', 'csv',
                                             id='format_pyradiomics_' + label,
                                             node_group='train')
            node_train.inputs['format'] =\
                self.source_format_pyradiomics.output

            if self.TrainTest:
                node_test.inputs['format'] =\
                    self.source_format_pyradiomics.output

        # Create required links
        # We can have a different config for different tools
        if 'pyradiomics' in calcfeat_node.lower():
            node_train.inputs['parameters'] =\
                self.source_config_pyradiomics[label].output
        else:
            node_train.inputs['parameters'] =\
                self.sources_parameters[label].output

        node_train.inputs['image'] =\
            self.preprocessing_train[label].outputs['image']

        if self.TrainTest:
            if 'pyradiomics' in calcfeat_node.lower():
                node_test.inputs['parameters'] =\
                    self.source_config_pyradiomics[label].output
            else:
                node_test.inputs['parameters'] =\
                    self.sources_parameters[label].output

            node_test.inputs['image'] =\
                self.preprocessing_test[label].outputs['image']

        # PREDICT can extract semantic and metadata features
        if 'predict' in calcfeat_node.lower():
            if self.metadata_train and len(self.metadata_train) >= nmod + 1:
                node_train.inputs['metadata'] =\
                    self.sources_metadata_train[label].output

            if self.metadata_test and len(self.metadata_test) >= nmod + 1:
                node_test.inputs['metadata'] =\
                    self.sources_metadata_test[label].output

            # If a semantics file is provided, connect to feature extraction tool
            if self.semantics_train and len(self.semantics_train) >= nmod + 1:
                self.sources_semantics_train[label] =\
                    self.network.create_source('CSVFile',
                                               id='semantics_train_' + label)

                node_train.inputs['semantics'] =\
                    self.sources_semantics_train[label].output

            if self.semantics_test and len(self.semantics_test) >= nmod + 1:
                self.sources_semantics_test[label] =\
                    self.network.create_source('CSVFile',
                                               id='semantics_test_' + label)
                node_test.inputs['semantics'] =\
                    self.sources_semantics_test[label].output

        # Add feature converter to make features WORC compatible
        conv_train =\
            self.network.create_node('worc/FeatureConverter:1.0',
                                     tool_version='1.0',
                                     id='featureconverter_train_' + node_ID,
                                     resources=ResourceLimit(memory='4G'))

        conv_train.inputs['feat_in'] = node_train.outputs['features']

        # Add source to tell converter which toolbox we use
        if 'pyradiomics' in calcfeat_node.lower():
            toolbox = 'PyRadiomics'
        elif 'predict' in calcfeat_node.lower():
            toolbox = 'PREDICT'
        else:
            message = f'Toolbox {calcfeat_node} not recognized!'
            raise WORCexceptions.WORCKeyError(message)

        self.source_toolbox_name[label] =\
            self.network.create_constant('String', toolbox,
                                         id=f'toolbox_name_{toolbox}_{label}')

        conv_train.inputs['toolbox'] = self.source_toolbox_name[label].output
        conv_train.inputs['config'] = self.sources_parameters[label].output

        if self.TrainTest:
            conv_test =\
                self.network.create_node('worc/FeatureConverter:1.0',
                                         tool_version='1.0',
                                         id='featureconverter_test_' + node_ID,
                                         resources=ResourceLimit(memory='4G'))

            conv_test.inputs['feat_in'] = node_test.outputs['features']
            conv_test.inputs['toolbox'] = self.source_toolbox_name[label].output
            conv_test.inputs['config'] = self.sources_parameters[label].output

        # Append to nodes to list
        self.calcfeatures_train[label].append(node_train)
        self.featureconverter_train[label].append(conv_train)
        if self.TrainTest:
            self.calcfeatures_test[label].append(node_test)
            self.featureconverter_test[label].append(conv_test)

    def add_elastix_sourcesandsinks(self):
        """Add sources and sinks required for image registration."""
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

    def add_elastix(self, label, nmod):
        """ Add image registration through elastix to network."""
        # Create sources and converter for only for the given segmentation,
        # which should be on the first modality
        if nmod == 0:
            memory = self.fastr_memory_parameters['WORCCastConvert']
            self.sources_segmentations_train[label] =\
                self.network.create_source('ITKImageFile',
                                           id='segmentations_train_' + label,
                                           node_group='input')

            self.converters_seg_train[label] =\
                self.network.create_node('worc/WORCCastConvert:0.3.2',
                                         tool_version='0.1',
                                         id='convert_seg_train_' + label,
                                         resources=ResourceLimit(memory=memory))

            self.converters_seg_train[label].inputs['image'] =\
                self.sources_segmentations_train[label].output

            if self.TrainTest:
                self.sources_segmentations_test[label] =\
                    self.network.create_source('ITKImageFile',
                                               id='segmentations_test_' + label,
                                               node_group='input')

                self.converters_seg_test[label] =\
                    self.network.create_node('worc/WORCCastConvert:0.3.2',
                                             tool_version='0.1',
                                             id='convert_seg_test_' + label,
                                             resources=ResourceLimit(memory=memory))

                self.converters_seg_test[label].inputs['image'] =\
                    self.sources_segmentations_test[label].output

        # Assume provided segmentation is on first modality
        if nmod > 0:
            # Use elastix and transformix for registration
            # NOTE: Assume elastix node type is on first configuration
            elastix_node =\
                str(self.configs[0]['General']['RegistrationNode'])

            transformix_node =\
                str(self.configs[0]['General']['TransformationNode'])

            memory_elastix = self.fastr_memory_parameters['Elastix']
            self.elastix_nodes_train[label] =\
                self.network.create_node(elastix_node,
                                         tool_version='0.2',
                                         id='elastix_train_' + label,
                                         resources=ResourceLimit(memory=memory_elastix))

            memory_transformix = self.fastr_memory_parameters['Elastix']
            self.transformix_seg_nodes_train[label] =\
                self.network.create_node(transformix_node,
                                         tool_version='0.2',
                                         id='transformix_seg_train_' + label,
                                         resources=ResourceLimit(memory=memory_transformix))

            self.transformix_im_nodes_train[label] =\
                self.network.create_node(transformix_node,
                                         tool_version='0.2',
                                         id='transformix_im_train_' + label,
                                         resources=ResourceLimit(memory=memory_transformix))

            if self.TrainTest:
                self.elastix_nodes_test[label] =\
                    self.network.create_node(elastix_node,
                                             tool_version='0.2',
                                             id='elastix_test_' + label,
                                             resources=ResourceLimit(memory=memory_elastix))

                self.transformix_seg_nodes_test[label] =\
                    self.network.create_node(transformix_node,
                                             tool_version='0.2',
                                             id='transformix_seg_test_' + label,
                                             resources=ResourceLimit(memory=memory_transformix))

                self.transformix_im_nodes_test[label] =\
                    self.network.create_node(transformix_node,
                                             tool_version='0.2',
                                             id='transformix_im_test_' + label,
                                             resources=ResourceLimit(memory=memory_transformix))

            # Create sources_segmentation
            # M1 = moving, others = fixed
            self.elastix_nodes_train[label].inputs['fixed_image'] =\
                self.converters_im_train[label].outputs['image']

            self.elastix_nodes_train[label].inputs['moving_image'] =\
                self.converters_im_train[self.modlabels[0]].outputs['image']

            # Add node that copies metadata from the image to the
            # segmentation if required
            if self.CopyMetadata:
                # Copy metadata from the image which was registered to
                # the segmentation, if it is not created yet
                if not hasattr(self, "copymetadata_nodes_train"):
                    # NOTE: Do this for first modality, as we assume
                    # the segmentation is on that one
                    self.copymetadata_nodes_train = dict()
                    self.copymetadata_nodes_train[self.modlabels[0]] =\
                        self.network.create_node('itktools/0.3.2/CopyMetadata:1.0',
                                                 tool_version='1.0',
                                                 id='CopyMetadata_train_' + self.modlabels[0])

                    self.copymetadata_nodes_train[self.modlabels[0]].inputs["source"] =\
                        self.converters_im_train[self.modlabels[0]].outputs['image']

                    self.copymetadata_nodes_train[self.modlabels[0]].inputs["destination"] =\
                        self.converters_seg_train[self.modlabels[0]].outputs['image']

                self.transformix_seg_nodes_train[label].inputs['image'] =\
                    self.copymetadata_nodes_train[self.modlabels[0]].outputs['output']
            else:
                self.transformix_seg_nodes_train[label].inputs['image'] =\
                    self.converters_seg_train[self.modlabels[0]].outputs['image']

            if self.TrainTest:
                self.elastix_nodes_test[label].inputs['fixed_image'] =\
                    self.converters_im_test[label].outputs['image']
                self.elastix_nodes_test[label].inputs['moving_image'] =\
                    self.converters_im_test[self.modlabels[0]].outputs['image']

                if self.CopyMetadata:
                    # Copy metadata from the image which was registered
                    # to the segmentation
                    if not hasattr(self, "copymetadata_nodes_test"):
                        # NOTE: Do this for first modality, as we assume
                        # the segmentation is on that one
                        self.copymetadata_nodes_test = dict()
                        self.copymetadata_nodes_test[self.modlabels[0]] =\
                            self.network.create_node('itktools/0.3.2/CopyMetadata:1.0',
                                                     tool_version='1.0',
                                                     id='CopyMetadata_test_' + self.modlabels[0])

                        self.copymetadata_nodes_test[self.modlabels[0]].inputs["source"] =\
                            self.converters_im_test[self.modlabels[0]].outputs['image']

                        self.copymetadata_nodes_test[self.modlabels[0]].inputs["destination"] =\
                            self.converters_seg_test[self.modlabels[0]].outputs['image']

                    self.transformix_seg_nodes_test[label].inputs['image'] =\
                        self.copymetadata_nodes_test[self.modlabels[0]].outputs['output']
                else:
                    self.transformix_seg_nodes_test[label].inputs['image'] =\
                        self.converters_seg_test[self.modlabels[0]].outputs['image']

            # Apply registration to input modalities
            self.source_Elastix_Parameters[label] =\
                self.network.create_source('ElastixParameterFile',
                                           id='Elastix_Para_' + label,
                                           node_group='elpara')

            self.link_elparam_train =\
                self.network.create_link(self.source_Elastix_Parameters[label].output,
                                         self.elastix_nodes_train[label].inputs['parameters'])

            self.link_elparam_train.collapse = 'elpara'

            if self.TrainTest:
                self.link_elparam_test =\
                    self.network.create_link(self.source_Elastix_Parameters[label].output,
                                             self.elastix_nodes_test[label].inputs['parameters'])
                self.link_elparam_test.collapse = 'elpara'

            if self.masks_train:
                self.elastix_nodes_train[label].inputs['fixed_mask'] =\
                    self.converters_masks_train[label].outputs['image']

                self.elastix_nodes_train[label].inputs['moving_mask'] =\
                    self.converters_masks_train[self.modlabels[0]].outputs['image']

            if self.TrainTest:
                if self.masks_test:
                    self.elastix_nodes_test[label].inputs['fixed_mask'] =\
                        self.converters_masks_test[label].outputs['image']

                    self.elastix_nodes_test[label].inputs['moving_mask'] =\
                        self.converters_masks_test[self.modlabels[0]].outputs['image']

            # Change the FinalBSpline Interpolation order to 0 as required for binarie images: see https://github.com/SuperElastix/elastix/wiki/FAQ
            self.edittransformfile_nodes_train[label] =\
                self.network.create_node('elastixtools/EditElastixTransformFile:0.1',
                                         tool_version='0.1',
                                         id='EditElastixTransformFile' + label)

            self.edittransformfile_nodes_train[label].inputs['set'] =\
                ["FinalBSplineInterpolationOrder=0"]

            self.edittransformfile_nodes_train[label].inputs['transform'] =\
                self.elastix_nodes_train[label].outputs['transform'][-1]

            if self.TrainTest:
                self.edittransformfile_nodes_test[label] =\
                    self.network.create_node('elastixtools/EditElastixTransformFile:0.1',
                                             tool_version='0.1',
                                             id='EditElastixTransformFile' + label)

                self.edittransformfile_nodes_test[label].inputs['set'] =\
                    ["FinalBSplineInterpolationOrder=0"]

                self.edittransformfile_nodes_test[label].inputs['transform'] =\
                    self.elastix_nodes_test[label].outputs['transform'][-1]

            # Link data and transformation to transformix and source
            self.transformix_seg_nodes_train[label].inputs['transform'] =\
                self.edittransformfile_nodes_train[label].outputs['transform']

            self.transformix_im_nodes_train[label].inputs['transform'] =\
                self.elastix_nodes_train[label].outputs['transform'][-1]

            self.transformix_im_nodes_train[label].inputs['image'] =\
                self.converters_im_train[self.modlabels[0]].outputs['image']

            if self.TrainTest:
                self.transformix_seg_nodes_test[label].inputs['transform'] =\
                    self.edittransformfile_nodes_test[label].outputs['transform']

                self.transformix_im_nodes_test[label].inputs['transform'] =\
                    self.elastix_nodes_test[label].outputs['transform'][-1]

                self.transformix_im_nodes_test[label].inputs['image'] =\
                    self.converters_im_test[self.modlabels[0]].outputs['image']

            for i_node in range(len(self.calcfeatures_train[label])):
                self.calcfeatures_train[label][i_node].inputs['segmentation'] =\
                    self.transformix_seg_nodes_train[label].outputs['image']
                if self.TrainTest:
                    self.calcfeatures_test[label][i_node].inputs['segmentation'] =\
                        self.transformix_seg_nodes_test[label].outputs['image']

            # Save outputfor the training set
            self.sinks_transformations_train[label] =\
                self.network.create_sink('ElastixTransformFile',
                                         id='transformations_train_' + label)

            self.sinks_segmentations_elastix_train[label] =\
                self.network.create_sink('ITKImageFile',
                                         id='segmentations_out_elastix_train_' + label)

            self.sinks_images_elastix_train[label] =\
                self.network.create_sink('ITKImageFile',
                                         id='images_out_elastix_train_' + label)

            self.sinks_transformations_train[label].input =\
                self.elastix_nodes_train[label].outputs['transform']

            self.sinks_segmentations_elastix_train[label].input =\
                self.transformix_seg_nodes_train[label].outputs['image']

            self.sinks_images_elastix_train[label].input =\
                self.transformix_im_nodes_train[label].outputs['image']

            # Save output for the test set
            if self.TrainTest:
                self.sinks_transformations_test[label] =\
                    self.network.create_sink('ElastixTransformFile',
                                             id='transformations_test_' + label)

                self.sinks_segmentations_elastix_test[label] =\
                    self.network.create_sink('ITKImageFile',
                                             id='segmentations_out_elastix_test_' + label)
                self.sinks_images_elastix_test[label] =\
                    self.network.create_sink('ITKImageFile', id='images_out_elastix_test_' + label)
                self.sinks_transformations_elastix_test[label].input =\
                    self.elastix_nodes_test[label].outputs['transform']
                self.sinks_segmentations_elastix_test[label].input =\
                    self.transformix_seg_nodes_test[label].outputs['image']
                self.sinks_images_elastix_test[label].input =\
                    self.transformix_im_nodes_test[label].outputs['image']

    def add_segmentix(self, label, nmod):
        """Add segmentix to the network."""
        # Segmentix nodes -------------------------------------------------
        # Use segmentix node to convert input segmentation into
        # correct contour
        if label not in self.sinks_segmentations_segmentix_train:
            self.sinks_segmentations_segmentix_train[label] =\
                self.network.create_sink('ITKImageFile',
                                         id='segmentations_out_segmentix_train_' + label)

        memory = self.fastr_memory_parameters['Segmentix']
        self.nodes_segmentix_train[label] =\
            self.network.create_node('segmentix/Segmentix:1.0',
                                     tool_version='1.0',
                                     id='segmentix_train_' + label,
                                     resources=ResourceLimit(memory=memory))

        # Input the image
        self.nodes_segmentix_train[label].inputs['image'] =\
            self.converters_im_train[label].outputs['image']

        # Input the segmentation
        if hasattr(self, 'transformix_seg_nodes_train'):
            if label in self.transformix_seg_nodes_train.keys():
                # Use output of registration in segmentix
                self.nodes_segmentix_train[label].inputs['segmentation_in'] =\
                    self.transformix_seg_nodes_train[label].outputs['image']
            else:
                # Use original segmentation
                self.nodes_segmentix_train[label].inputs['segmentation_in'] =\
                    self.converters_seg_train[label].outputs['image']
        else:
            # Use original segmentation
            self.nodes_segmentix_train[label].inputs['segmentation_in'] =\
                self.converters_seg_train[label].outputs['image']

        # Input the parameters
        self.nodes_segmentix_train[label].inputs['parameters'] =\
            self.sources_parameters[label].output
        self.sinks_segmentations_segmentix_train[label].input =\
            self.nodes_segmentix_train[label].outputs['segmentation_out']

        if self.TrainTest:
            self.sinks_segmentations_segmentix_test[label] =\
                self.network.create_sink('ITKImageFile',
                                         id='segmentations_out_segmentix_test_' + label)
            self.nodes_segmentix_test[label] =\
                self.network.create_node('segmentix/Segmentix:1.0',
                                         tool_version='1.0',
                                         id='segmentix_test_' + label, resources=ResourceLimit(memory=memory))

            self.nodes_segmentix_test[label].inputs['image'] =\
                self.converters_im_test[label].outputs['image']

            if hasattr(self, 'transformix_seg_nodes_test'):
                if label in self.transformix_seg_nodes_test.keys():
                    # Use output of registration in segmentix
                    self.nodes_segmentix_test[label].inputs['segmentation_in'] =\
                        self.transformix_seg_nodes_test[label].outputs['image']
                else:
                    # Use original segmentation
                    self.nodes_segmentix_test[label].inputs['segmentation_in'] =\
                        self.converters_seg_test[label].outputs['image']
            else:
                # Use original segmentation
                self.nodes_segmentix_test[label].inputs['segmentation_in'] =\
                    self.converters_seg_test[label].outputs['image']

            self.nodes_segmentix_test[label].inputs['parameters'] =\
                self.sources_parameters[label].output
            self.sinks_segmentations_segmentix_test[label].input =\
                self.nodes_segmentix_test[label].outputs['segmentation_out']

        for i_node in range(len(self.calcfeatures_train[label])):
            self.calcfeatures_train[label][i_node].inputs['segmentation'] =\
                self.nodes_segmentix_train[label].outputs['segmentation_out']

            if self.TrainTest:
                self.calcfeatures_test[label][i_node].inputs['segmentation'] =\
                    self.nodes_segmentix_test[label].outputs['segmentation_out']


        if self.masks_train and len(self.masks_train) >= nmod + 1:
            # Use masks
            self.nodes_segmentix_train[label].inputs['mask'] =\
                self.converters_masks_train[label].outputs['image']

        if self.masks_test and len(self.masks_test) >= nmod + 1:
            # Use masks
            self.nodes_segmentix_test[label].inputs['mask'] =\
                self.converters_masks_test[label].outputs['image']

    def set(self):
        """Set the FASTR source and sink data based on the given attributes."""
        self.fastrconfigs = list()
        self.source_data = dict()
        self.sink_data = dict()

        # Save the configurations as files
        self.save_config()

        # Generate gridsearch parameter files if required
        self.source_data['config_classification_source'] = self.fastrconfigs[0]

        # Set source and sink data
        self.source_data['patientclass_train'] = self.labels_train
        self.source_data['patientclass_test'] = self.labels_test

        self.sink_data['classification'] = ("vfs://output/{}/estimator_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
        self.sink_data['performance'] = ("vfs://output/{}/performance_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
        self.sink_data['config_classification_sink'] = ("vfs://output/{}/config_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
        self.sink_data['features_train_ComBat'] = ("vfs://output/{}/ComBat/features_ComBat_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)
        self.sink_data['features_test_ComBat'] = ("vfs://output/{}/ComBat/features_ComBat_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name)

        # Set the source data from the WORC objects you created
        for num, label in enumerate(self.modlabels):
            self.source_data['config_' + label] = self.fastrconfigs[num]
            if self.pyradiomics_configs:
                self.source_data['config_pyradiomics_' + label] = self.pyradiomics_configs[num]

            # Add train data sources
            if self.images_train and len(self.images_train) - 1 >= num:
                self.source_data['images_train_' + label] = self.images_train[num]

            if self.masks_train and len(self.masks_train) - 1 >= num:
                self.source_data['mask_train_' + label] = self.masks_train[num]

            if self.masks_normalize_train and len(self.masks_normalize_train) - 1 >= num:
                self.source_data['masks_normalize_train_' + label] = self.masks_normalize_train[num]

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

            if self.masks_normalize_test and len(self.masks_normalize_test) - 1 >= num:
                self.source_data['masks_normalize_test_' + label] = self.masks_normalize_test[num]

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
            if hasattr(self, 'featurecalculators'):
                for f in self.featurecalculators[label]:
                    self.sink_data['features_train_' + label + '_' + f] = ("vfs://output/{}/Features/features_{}_{}_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name, f, label)

            if self.labels_test:
                self.sink_data['segmentations_out_segmentix_test_' + label] = ("vfs://output/{}/Segmentations/seg_{}_segmentix_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name, label)
                self.sink_data['segmentations_out_elastix_test_' + label] = ("vfs://output/{}/Elastix/seg_{}_elastix_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name, label)
                self.sink_data['images_out_elastix_test_' + label] = ("vfs://output/{}/Images/im_{}_elastix_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name, label)
                if hasattr(self, 'featurecalculators'):
                    for f in self.featurecalculators[label]:
                        self.sink_data['features_test_' + label + '_' + f] = ("vfs://output/{}/Features/features_{}_{}_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name, f, label)

            # Add elastix sinks if used
            if self.segmode:
                # Segmode is only non-empty if segmentations are provided
                if self.segmode == 'Register':
                    self.sink_data['transformations_train_' + label] = ("vfs://output/{}/Elastix/transformation_{}_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name, label)
                    if self.TrainTest:
                        self.sink_data['transformations_test_' + label] = ("vfs://output/{}/Elastix/transformation_{}_{{sample_id}}_{{cardinality}}{{ext}}").format(self.name, label)

        if self._add_evaluation:
            self.Evaluate.set()

    def execute(self):
        """Execute the network through the fastr.network.execute command."""
        # Draw and execute nwtwork
        try:
            self.network.draw(file_path=self.network.id + '.svg', draw_dimensions=True)
        except graphviz.backend.ExecutableNotFound:
            print('[WORC WARNING] Graphviz executable not found: not drawing network diagram. Make sure the Graphviz executables are on your systems PATH.')

        if DebugDetector().do_detection():
            print("Source Data:")
            for k in self.source_data.keys():
                print(f"\t {k}: {self.source_data[k]}.")
            print("\n Sink Data:")
            for k in self.sink_data.keys():
                print(f"\t {k}: {self.sink_data[k]}.")

            # When debugging, set the tempdir to the default of fastr + name
            self.fastr_tmpdir = os.path.join(fastr.config.mounts['tmp'],
                                             self.name)

        self.network.execute(self.source_data, self.sink_data, execution_plugin=self.fastr_plugin, tmpdir=self.fastr_tmpdir)

    def add_evaluation(self, label_type):
        """Add branch for evaluation of performance to network."""
        self.Evaluate = Evaluate(label_type=label_type, parent=self)
        self._add_evaluation = True

    def save_config(self):
        """Save the config files to physical files and add to network."""
        # If the configuration files are confiparse objects, write to file
        self.pyradiomics_configs = list()

        # Make sure we can dump blank values for PyRadiomics
        yaml.SafeDumper.add_representer(type(None),
                                        lambda dumper, value: dumper.represent_scalar(u'tag:yaml.org,2002:null', ''))

        for num, c in enumerate(self.configs):
            if type(c) != configparser.ConfigParser:
                # A filepath (not a fastr source) is provided. Hence we read
                # the config file and convert it to a configparser object
                config = configparser.ConfigParser()
                config.read(c)
                c = config
            cfile = os.path.join(self.fastr_tmpdir, f"config_{self.name}_{num}.ini")
            if not os.path.exists(os.path.dirname(cfile)):
                os.makedirs(os.path.dirname(cfile))
            with open(cfile, 'w') as configfile:
                c.write(configfile)

            # If PyRadiomics is used, also write a config for PyRadiomics
            if 'pyradiomics' in c['General']['FeatureCalculators']:
                cfile_pyradiomics = os.path.join(self.fastr_tmpdir, f"config_pyradiomics_{self.name}_{num}.yaml")
                config_pyradiomics = io.convert_config_pyradiomics(c)
                with open(cfile_pyradiomics, 'w') as file:
                    yaml.safe_dump(config_pyradiomics, file)
                cfile_pyradiomics = Path(self.fastr_tmpdir) / f"config_pyradiomics_{self.name}_{num}.yaml"
                self.pyradiomics_configs.append(cfile_pyradiomics.as_uri().replace('%20', ' '))

            # BUG: Make path with pathlib to create windows double slashes
            cfile = Path(self.fastr_tmpdir) / f"config_{self.name}_{num}.ini"
            self.fastrconfigs.append(cfile.as_uri().replace('%20', ' '))


class Tools(object):
    """
    Create other pipelines besides the default radiomics executions.

    Currently includes:
    1. Registration pipeline
    2. Evaluation pipeline
    3. Slicer pipeline, to create pngs of middle slice of images.
    """

    def __init__(self):
        """Initialize object with all pipelines."""
        self.Elastix = Elastix()
        self.Evaluate = Evaluate()
        self.Slicer = Slicer()
