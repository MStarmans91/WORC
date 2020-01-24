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


def _for_all_methods(decorator):
    def decorate(cls):
        for attr in cls.__dict__:  # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate


def _error_buldozer(func):
    _valid_exceptions = [
        PathNotFoundException, NoImagesFoundException,
        NoSegmentationsFoundException, InvalidCsvFileException,
        TypeError, ValueError, NotImplementedError, WORCKeyError,
        WORCValueError, WORCAssertionError
    ]
    _valid_exceptions += [c[1] for c in inspect.getmembers(fastr.exceptions, inspect.isclass)]

    unexpected_exception_exception = Exception('A blackhole to another dimenstion has opened. This exception should never be thrown. Double check your code or make an issue on the WORC github so that we can fix this issue.')

    def dec(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            if e.__class__ not in _valid_exceptions:
                raise unexpected_exception_exception
            raise e
    return dec


@_for_all_methods(_error_buldozer)
class SimpleWORC():
    def __init__(self, name='WORC'):
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

        self._config_builder = ConfigBuilder()
        self._add_evaluation = False

        # Detect wether we are on a cluster
        if BigrClusterDetector().do_detection():
            self._worc.fastr_plugin = 'DRMAAExecution'
        elif CartesiusClusterDetector().do_detection():
            self._worc.fastr_plugin = 'ProcessPoolExecution'

    def features_from_this_directory(self, directory,
                                     feature_file_name='features.hdf5',
                                     glob='*/', is_training=True):
        directory = Path(directory).expanduser()
        if not directory.exists():
            raise PathNotFoundException(directory)

        features = list(directory.glob(f'{glob}{feature_file_name}'))

        if len(features) == 0:
            raise NoFeaturesFoundException(f'{directory}{glob}{image_file_name}')

        features_per_subject = {feature.parent.name: feature.as_uri() for feature in features}
        if is_training:
            self._features_train.append(features_per_subject)
        else:
            self._features_test.append(features_per_subject)

    def images_from_this_directory(self, directory, image_file_name='image.nii.gz', glob='*/', is_training=True):
        directory = Path(directory).expanduser()
        if not directory.exists():
            raise PathNotFoundException(directory)

        images = list(directory.glob(f'{glob}{image_file_name}'))

        if len(images) == 0:
            raise NoImagesFoundException(f'{directory}{glob}{image_file_name}')

        images_per_subject = {image.parent.name: image.as_uri() for image in images}
        if is_training:
            self._images_train.append(images_per_subject)
        else:
            self._images_test.append(images_per_subject)

    def segmentations_from_this_directory(self, directory, segmentation_file_name='segmentation.nii.gz', glob='*/',
                                          is_training=True):
        directory = Path(directory).expanduser()
        if not directory.exists():
            raise PathNotFoundException(directory)

        segmentations = list(directory.glob(f'{glob}{segmentation_file_name}'))

        if len(segmentations) == 0:
            raise NoSegmentationsFoundException(str(directory))

        segmentations_per_subject = {image.parent.name: image.as_uri() for image in segmentations}
        if is_training:
            self._segmentations_train.append(segmentations_per_subject)
        else:
            self._segmentations_test.append(segmentations_per_subject)

    def labels_from_this_file(self, file_path, is_training=True):
        labels_file = Path(file_path).expanduser()

        if not labels_file.is_file():
            raise PathNotFoundException(file_path)

        if not CsvDetector(labels_file.absolute()):
            raise InvalidCsvFileException(labels_file.absolute())

        # TODO: implement sanity check labels file e.g. is it a labels file and are there labels available
        if is_training:
            self._labels_file_train = labels_file.as_uri()
        else:
            self._labels_file_test = labels_file.as_uri()

    def semantics_from_this_file(self, file_path, is_training=True):
        semantics_file = Path(file_path).expanduser()

        if not semantics_file.is_file():
            raise PathNotFoundException(file_path)

        if not CsvDetector(semantics_file.absolute()):
            raise InvalidCsvFileException(semantics_file.absolute())

        # TODO: implement sanity check semantics file e.g. is it a semantics file and are there semantics available
        if is_training:
            self._semantics_file_train = semantics_file.as_uri()
        else:
            self._semantics_file_test = semantics_file.as_uri()

    def predict_labels(self, label_names: list):
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
        # validate
        if method == 'classification':
            valid_estimators = ['SVM', 'RF', 'SGD', 'LR', 'GaussianNB', 'ComplementNB', 'LDA', 'QDA', 'RankedSVM']
        elif method == 'regression':
            valid_estimators = ['SVR', 'RFR', 'ElasticNet', 'Lasso', 'SGDR']
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
        else:
            self._config_builder.full_overrides()

        self._method = method

    def count_num_subjects(self):
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
        validators = ValidatorsFactory.factor_validators()
        self.count_num_subjects()

        for validator in validators:
            validator.do_validation(self)

    def execute(self):
        # this function is kind of like the build()-function in a builder, except it peforms execute on the object being built as well
        self._validate()  # do some final sanity checking before we execute the thing

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

        self._worc.images_train = self._images_train
        self._worc.features_train = self._features_train
        self._worc.segmentations_train = self._segmentations_train
        self._worc.labels_train = self._labels_file_train
        self._worc.semantics_train = self._semantics_file_train

        if self._images_test:
            self._worc.images_test = self._images_test

        if self._features_test:
            self._worc.features_test = self._features_test

        if self._segmentations_test:
            self._worc.segmentations_test = self._segmentations_test

        if self._labels_file_test:
            self._worc.labels_test = self._labels_file_test

        self._worc.label_names = ', '.join(self._label_names)
        self._config_builder._custom_overrides['Labels'] = dict()
        self._config_builder._custom_overrides['Labels']['label_names'] = self._worc.label_names

        self._worc.configs = [self._config_builder.build_config(self._worc.defaultconfig())]
        self._worc.build()
        if self._add_evaluation:
            self._worc.add_evaluation(label_type=self._label_names[self._selected_label])

        self._worc.set()
        self._worc.execute()

    def binary_classification(self, estimators=['SVM'], scoring_method='f1', coarse=True):
        self._set_and_validate_estimators(estimators, scoring_method, 'classification', coarse)

    def regression(self, estimators=['SVR'], scoring_method='r2', coarse=True):
        self._set_and_validate_estimators(estimators, scoring_method, 'regression', coarse)

    def survival(self, estimators, scoring_method, coarse=True):
        raise NotImplementedError()

    def add_config_overrides(self, config):
        self._config_builder.custom_config_overrides(config)

    def add_evaluation(self, selected_label=0):
        self._add_evaluation = True
        self._selected_label = 0

    def set_tmpdir(self, tmpdir):
        self._worc.fastr_tmpdir = tmpdir

    def set_multicore_execution(self):
        self._worc.fastr_plugin = 'ProcessPoolExecution'
        self._config_builder.custom_config_overrides['Classification']['fastr_plugin'] = 'ProcessPoolExecution'

    def features_from_radiomix_xlsx(self, feature_file):
        self._radiomix_feature_file = feature_file
