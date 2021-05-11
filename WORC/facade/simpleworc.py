#!/usr/bin/env python

# Copyright 2016-2021 Biomedical Imaging Group Rotterdam, Departments of
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

import fastr.exceptions
from pathlib import Path
import inspect
import os
import pandas as pd
from WORC import WORC
from .helpers.processing import convert_radiomix_features
from .helpers.exceptions import PathNotFoundException, NoImagesFoundException, \
    NoSegmentationsFoundException, InvalidCsvFileException, \
    NoFeaturesFoundException
from WORC.addexceptions import WORCKeyError, WORCValueError, WORCAssertionError
from .helpers.configbuilder import ConfigBuilder
from WORC.detectors.detectors import CsvDetector, BigrClusterDetector, \
    CartesiusClusterDetector

from WORC.validators.preflightcheck import ValidatorsFactory
from functools import wraps


def _for_all_methods(decorator):
    """Methods to attach an object to all methods."""
    def decorate(cls):
        for attr in cls.__dict__:  # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate


def _error_bulldozer(func):
    """Checks whether raised errors are known or should never occur."""
    _valid_exceptions = [
        PathNotFoundException, NoImagesFoundException,
        NoSegmentationsFoundException, InvalidCsvFileException,
        TypeError, ValueError, NotImplementedError, WORCKeyError,
        WORCValueError, WORCAssertionError
    ]
    _valid_exceptions += [c[1] for c in inspect.getmembers(fastr.exceptions, inspect.isclass)]

    unexpected_exception_exception = Exception('A blackhole to another dimenstion has opened. This exception should never be thrown. Double check your code or make an issue on the WORC github so that we can fix this issue.')

    @wraps(func)
    def dec(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            if e.__class__ not in _valid_exceptions:
                raise unexpected_exception_exception
            raise e
    return dec


@_for_all_methods(_error_bulldozer)
class SimpleWORC():
    """Facade around the main WORC object for simple interaction.

    Please also see the `WORCTutorial Github <https://github.com/MStarmans91/WORCTutorial/>`_.
    """

    def __init__(self, name='WORC'):
        """Initialize SimpleWORC object.

        Parameters
        -----------
        name: string, default WORC
            String to identify name of experiments. Will be used in the temporary files and outputs.

        """
        # Set some config values
        self._worc = WORC(name)

        self._images_train = []
        self._images_test = []
        self._features_train = []
        self._features_test = []
        self._segmentations_train = []
        self._segmentations_test = []
        self._semantics_file_train = None
        self._semantics_file_test = None
        self._radiomix_feature_file = None

        self._labels_file_train = None
        self._labels_file_test = None
        self._label_names = []

        self._method = None

        self._fixed_splits = None

        self._config_builder = ConfigBuilder()
        self._add_evaluation = False

        # Detect wether we are on a cluster
        if BigrClusterDetector().do_detection():
            self._worc.fastr_plugin = 'DRMAAExecution'
        elif CartesiusClusterDetector().do_detection():
            self._worc.fastr_plugin = 'ProcessPoolExecution'

    def set_fixed_splits(self, fixed_splits_csv):
        if not Path(fixed_splits_csv).is_file():
            raise PathNotFoundException(fixed_splits_csv)

        if self._fixed_splits is not None:
            print('WARN: set_fixed_splits already set. Please check your script to make sure this is ok!')

        self._fixed_splits = fixed_splits_csv

    def features_from_this_directory(self, directory,
                                     feature_file_name='features.hdf5',
                                     glob='*/', is_training=True):
        """Use features from a directory as sources in WORC.

        SimpleWORC uses a directory glob to look for files meeting
        the requirements to include, based on the input parameters.

        Example:
        When using "directory = C:\\Users\\MyName\\FeatureFolder",
        features_from_this_directory will include all features.hdf5 files from all subfolders in the directory.

        Parameters
        ----------
        directory: string
            Identifies the root directory in which to search for feature files.
        feature_file_name: string, default features.hdf5
            Name of the files which will be included. Can include wildcards (*).
        glob: string, default */
            Identify the search string to be used in the glob. Can include wildcards (*).
        is_training: Boolean, default True
            Identify whether these features should be used in the training or test dataset.
        """
        directory = Path(directory).expanduser()
        if not directory.exists():
            raise PathNotFoundException(directory)

        features = list(directory.glob(f'{glob}{feature_file_name}'))

        if len(features) == 0:
            raise NoFeaturesFoundException(f'{directory}{glob}{feature_file_name}')

        features_per_subject = {feature.parent.name: feature.as_uri().replace('%20', ' ') for feature in features}
        if is_training:
            self._features_train.append(features_per_subject)
        else:
            self._features_test.append(features_per_subject)

    def images_from_this_directory(self, directory, image_file_name='image.nii.gz', glob='*/', is_training=True):
        """Use images from a directory as sources in WORC.

        SimpleWORC uses a directory glob to look for files meeting
        the requirements to include, based on the input parameters.

        Example:
        When using "directory = C:\\Users\\MyName\\ImageFolder",
        images_from_this_directory will include all image.nii.gz files from all subfolders in the directory.

        Parameters
        ----------
        directory: string
            Identifies the root directory in which to search for image files.
        image_file_name: string, default image.nii.gz
            Name of the files which will be included. Can include wildcards (*).
        glob: string, default */
            Identify the search string to be used in the glob. Can include wildcards (*).
        is_training: Boolean, default True
            Identify whether these images should be used in the training or test dataset.
        """
        directory = Path(directory).expanduser()
        if not directory.exists():
            raise PathNotFoundException(directory)

        images = list(directory.glob(f'{glob}{image_file_name}'))

        if len(images) == 0:
            raise NoImagesFoundException(f'{directory}{glob}{image_file_name}')

        images_per_subject = {image.parent.name: image.as_uri().replace('%20', ' ') for image in images}
        if is_training:
            self._images_train.append(images_per_subject)
        else:
            self._images_test.append(images_per_subject)

    def segmentations_from_this_directory(self, directory, segmentation_file_name='segmentation.nii.gz', glob='*/',
                                          is_training=True):
        """Use segmentations from a directory as sources in WORC.

         SimpleWORC uses a directory glob to look for files meeting
         the requirements to include, based on the input parameters.

         Example:
         When using "directory = C:\\Users\\MyName\\SegmentationFolder",
         segmentations_from_this_directory will include all segmentation.nii.gz files from all subfolders in the directory.

         Parameters
         ----------
         directory: string
             Identifies the root directory in which to search for segmentation files.
         segmentation_file_name: string, default segmentation.nii.gz
             Name of the files which will be included. Can include wildcards (*).
         glob: string, default */
             Identify the search string to be used in the glob. Can include wildcards (*).
         is_training: Boolean, default True
             Identify whether these segmentations should be used in the training or test dataset.
         """
        directory = Path(directory).expanduser()
        if not directory.exists():
            raise PathNotFoundException(directory)

        segmentations = list(directory.glob(f'{glob}{segmentation_file_name}'))

        if len(segmentations) == 0:
            raise NoSegmentationsFoundException(str(directory))

        segmentations_per_subject = {segmentation.parent.name: segmentation.as_uri().replace('%20', ' ') for segmentation in segmentations}
        if is_training:
            self._segmentations_train.append(segmentations_per_subject)
        else:
            self._segmentations_test.append(segmentations_per_subject)

    def labels_from_this_file(self, file_path, is_training=True):
        """Define which file should be used by WORC to extract the object labels.

        Should be a .csv or .txt file: see
        :ref:`the WORC user manual <usermanual-chapter:>` for more details on
        the formatting of this file.

        Parameters
        ----------
        file_path: basestring
            Location of the file to be used as label file. Can be a .csv or .txt file.
        is_training: Boolean, default True
            Identify whether this label file should be used in the training or test dataset.

        """
        labels_file = Path(file_path).expanduser()

        if not labels_file.is_file():
            raise PathNotFoundException(file_path)

        if not CsvDetector(labels_file.absolute()):
            raise InvalidCsvFileException(labels_file.absolute())

        if is_training:
            self._labels_file_train = labels_file.as_uri().replace('%20', ' ')
        else:
            self._labels_file_test = labels_file.as_uri().replace('%20', ' ')

    def semantics_from_this_file(self, file_path, is_training=True):
        """Define which file should be used by WORC to extract the semantic features.

        The values in these file can be used as semantic, i.e. non-computational,
        features in WORC. Should be a .csv file: see
        :ref:`the WORC user manual <usermanual-chapter:>` for more details on
        the formatting of this file.

        Parameters
        ----------
        file_path: basestring
            Location of the file to be used as semantics file. Can be a .csv or .txt file.
        is_training: Boolean, default True
            Identify whether this semantics  file should be used in the training or test dataset.

        """
        semantics_file = Path(file_path).expanduser()

        if not semantics_file.is_file():
            raise PathNotFoundException(file_path)

        if not CsvDetector(semantics_file.absolute()):
            raise InvalidCsvFileException(semantics_file.absolute())

        # TODO: implement sanity check semantics file e.g. is it a semantics file and are there semantics available
        if is_training:
            self._semantics_file_train = [semantics_file.as_uri().replace('%20', ' ')]
        else:
            self._semantics_file_test = [semantics_file.as_uri().replace('%20', ' ')]

    def predict_labels(self, label_names: list):
        """Determine which label(s) to predict in your experiments.

        The labels(s) you want to predict should be given in strings and
        should be included in the header of your labels_from_this_file csv.
        Note that you therefore first need to use that function to determine
        which label file to use, and afterwards use this function
        to select one of the headers / columns containing the actual label.

        Parameters
        -----------
        label_names: list
            List of strings containing the label name(s) to predict. For each
            label, a separate (classification) model will be created, or,
            if a multilabel experiment is run, a singel multiclass / multilabel
            classification model


        """
        if not self._labels_file_train:
            if not self.labels_file_train:
                raise ValueError('No labels file set! You can do this through labels_from_this_file')

        if not isinstance(label_names, list):
            raise TypeError(f'label_names is of type {type(label_names)} while list is expected')

        for label in label_names:
            if len(label.strip()) == 0:
                raise ValueError('Invalid label, length = 0')

        # TODO: check if labels is in labels file

        # self._worc.label_names = ', '.join(label_names)
        self._label_names = label_names

    def _set_and_validate_estimators(self, estimators, scoring_method, method, coarse):
        """Check whether the given estimators in the config are valid for the experiment.

        Parameters
        -----------
        estimators: list
            String(s) of the estimators to be used. See the
            :ref:`WORC Config chapter <config-chapter>` for allowed options.
        scoring_method: string
            Name of metric to be used for ranking the workflows.
        method: string
            Currently supported: classification or regression
        coarse: boolean
            Determine whether to do a coarse or full experiment.

        """
        # validate
        if method == 'classification':
            valid_estimators = ['SVM', 'RF', 'SGD', 'LR', 'GaussianNB', 'ComplementNB', 'LDA', 'QDA', 'RankedSVM']
        elif method == 'regression':
            valid_estimators = ['SVR', 'RFR', 'ElasticNet', 'Lasso', 'SGDR', 'XGBRegressor', 'AdaBoostRegressor', 'LinR', 'Ridge']
        else:
            valid_estimators = []

        for estimator in estimators:
            if estimator not in valid_estimators:
                raise ValueError(
                    f'Invalid estimator {estimator} for {method}; must be one of {", ".join(valid_estimators)}')

        # TODO: sanity check scoring method per estimator

        # set
        self._config_builder.estimator_scoring_overrides(estimators, scoring_method)

        if coarse:
            self._config_builder.coarse_overrides()

        self._method = method

    def count_num_subjects(self):
        """Count the number of subjects in the experiment."""
        if self._radiomix_feature_file:
            f = pd.read_excel(self._radiomix_feature_file)
            pids = f.values[:, 4]
            tocount = pids
        elif self._images_train:
            tocount = self._images_train[0]
        elif self._features_train:
            tocount = self._features_train[0]
        elif self.images_train:
            tocount = self.images_train[0]
        elif self.features_train:
            tocount = self.features_train[0]
        else:
            message = 'No features or images given, cannot count number ' +\
                ' of subjects. Make sure you input at least one of these ' +\
                'as source.'
            raise WORCValueError(message)

        if type(tocount) == dict():
            num_subjects = len(list(tocount.keys()))
        else:
            num_subjects = len(tocount)

        self._num_subjects = num_subjects

    def _validate(self):
        """Run various validators to validate the experiment."""
        validators = ValidatorsFactory.factor_validators()
        self.count_num_subjects()

        for validator in validators:
            validator.do_validation(self)

    def execute(self):
        """Execute the experiment.

        Before executing the actual experiment, this function will first run several validators
        and check the provided setup to make sure some of the most common
        made error are caught before running the experiment.
        """
        # Do some final sanity checking before we execute the experiment
        self._validate()

        if self._fixed_splits:
            self._worc.fixedsplits = self._fixed_splits

        if self._radiomix_feature_file:
            # Convert radiomix features and use those as inputs
            output_folder = os.path.join(fastr.config.mounts['tmp'],
                                         'Radiomix_features')

            # Check if output folder exists: otherwise create
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            # convert the features
            convert_radiomix_features(self._radiomix_feature_file, output_folder)

            # Set the newly created feature files as the WORC input
            self.features_from_this_directory(output_folder)

        # Give set sources to the WORC object
        self._worc.images_train = self._images_train
        self._worc.features_train = self._features_train
        self._worc.segmentations_train = self._segmentations_train
        self._worc.labels_train = self._labels_file_train
        self._worc.semantics_train = self._semantics_file_train

        # If a specific train-test setup is provided, add test sources
        if self._images_test:
            self._worc.images_test = self._images_test

        if self._features_test:
            self._worc.features_test = self._features_test

        if self._segmentations_test:
            self._worc.segmentations_test = self._segmentations_test

        if self._labels_file_test:
            self._worc.labels_test = self._labels_file_test

        if self._semantics_file_test:
            self._worc.semantics_test = self._semantics_file_test

        # Set the labels to predict
        self._worc.label_names = ', '.join(self._label_names)
        self._config_builder._custom_overrides['Labels'] = dict()
        self._config_builder._custom_overrides['Labels']['label_names'] = self._worc.label_names

        # Find out how many configs we need to make
        if self._worc.images_train:
            nmod = len(self._worc.images_train)
        else:
            nmod = len(self._worc.features_train)

        # Create configuration files
        self._worc.configs = [self._config_builder.build_config(self._worc.defaultconfig())] * nmod

        # Build the fastr network
        self._worc.build()
        if self._add_evaluation:
            self._worc.add_evaluation(label_type=self._label_names[self._selected_label],
                                      modus=self._method)

        # Set the sources and sinks and execute the experiment.
        self._worc.set()
        self._worc.execute()

    def binary_classification(self, estimators=None,
                              scoring_method='f1_weighted',
                              coarse=True):
        """Tell WORC do to a binary classification experiment.

        Parameters
        ----------
        estimators: list
            List of strings with names of valid estimators. See the
            :ref:`WORC Config chapter <config-chapter>` for allowed options.

            If coarse, only an SVM will be used. If not, the default full
            config will be used.
        scoring_method: string, default f1
            Name of the scoring method used to rank the workflows. See the
            :ref:`WORC Config chapter <config-chapter>` for allowed options.
        coarse: boolean, default True
            Determine whether to do a coarse or full experiment.

        """
        if coarse and estimators is None:
            estimators = ['SVM']
        elif estimators is None:
            estimators = ['SVM', 'SVM', 'SVM', 'RF', 'LR', 'LDA', 'QDA', 'GaussianNB']

        self._set_and_validate_estimators(estimators, scoring_method, 'classification', coarse)

    def regression(self, estimators=None, scoring_method='r2', coarse=True):
        """Tell WORC do to a regression experiment.

        Parameters
        ----------
        estimators: list
            List of strings with names of valid estimators. See the
            :ref:`WORC Config chapter <config-chapter>` for allowed options.

            If coarse, only SVR will be used. If not, the default full
            config will be used.
        scoring_method: string, default r2
            Name of the scoring method used to rank the workflows. See the
            :ref:`WORC Config chapter <config-chapter>` for allowed options.
        coarse: boolean, default True
            Determine whether to do a coarse or full experiment.

        """
        if coarse and estimators is None:
            estimators = ['SVR']
        elif estimators is None:
            estimators = ['SVR', 'RFR', 'ElasticNet', 'Lasso', 'AdaBoostRegressor', 'XGBRegressor', 'LinR', 'Ridge']

        # regression-specific override
        overrides = {
            'Featsel': {
                'SelectFromModel': 0.0,
                'StatisticalTestUse': 0.0,
                'ReliefUse': 0.0,
            },
        }
        self.add_config_overrides(overrides)

        self._set_and_validate_estimators(estimators, scoring_method, 'regression', coarse)

    def survival(self, estimators, scoring_method, coarse=True):
        """Tell WORC do to a regression experiment. Not implemented yet."""
        raise NotImplementedError()

    def add_config_overrides(self, config):
        """Add manual overrides for the WORC configuration.

        For a full list of options, see the
        :ref:`WORC Config chapter <config-chapter>` for allowed options.

        Parameters
        ----------
        config: dictionary
            Determine which options to override with which values.
        """
        self._config_builder.custom_config_overrides(config)

    def add_evaluation(self, selected_label=0):
        """Add the evaluation workflow to the standard WORC workflow.

        Adds several evaluation measures, including:

        - Computation of 95% confidence intervals for performance
        - Univariate testing of features
        - ROC curve with confidence bands construction
        - Ranking of patients based on percentage / posterior
        - Decompositions (e.g. PCA, t-SNE)

        See the :ref:`WORC documentation <additonalfunctionality-chapter>` for additional info.

        Parameters
        ----------
        selected_label: integer, default 0
            Determine for which of the labels to be predicted the evaluate workflow
            should be executed.

        """
        self._add_evaluation = True
        self._selected_label = 0
        self._worc.modus = self._method

    def set_tmpdir(self, tmpdir):
        """Set a directory for storing temporary files from the experiment.

        If not specified, the default fastr tmpdir is used, see ``fastr.config.mounts['tmp']`` .
        """
        self._worc.fastr_tmpdir = tmpdir

    def set_multicore_execution(self):
        """"Execute experiment in multicore mode.

        By default, SimpleWORC executes experiments in LinearExecution mode, meaning that
        only a single core will be used and jobs are executed in series. When multicore mode
        is enabled, jobs are parallellized over all available cores, which majorly speeds
        up the computation.

        Note: SimpleWORC has an automatic detector for the BIGR and Cartesius cluster. Hence,
        on those clusters, do not use the multicore execution, as this will overwrite
        the changes applied by the detectors.
        """
        self._worc.fastr_plugin = 'ProcessPoolExecution'
        self.add_config_overrides({'Classification': {'fastr_plugin': 'ProcessPoolExecution'}})

    def features_from_radiomix_xlsx(self, feature_file):
        """Use a feature file which is generated by the OncoRadiomics Radiomix tool."""
        self._radiomix_feature_file = feature_file
