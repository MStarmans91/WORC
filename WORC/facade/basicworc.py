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

from .simpleworc import SimpleWORC, _for_all_methods, _error_bulldozer
import os
import fastr
from .helpers.processing import convert_radiomix_features


@_for_all_methods(_error_bulldozer)
class BasicWORC(SimpleWORC):
    """Facade around the main WORC object for simple interaction.

    Based upon the SimpleWORC object, but with additional functionality:
    - Sources of WORC (e.g. images_train, segmentations_train, ...)
      can be directly assessed.

    Please also see the `WORCTutorial Github <https://github.com/MStarmans91/WORCTutorial/>`_.
    """

    def __init__(self, name='WORC'):
        super().__init__(name)

        # All hidden objects are now linked to unhidden objects
        self.images_train = []
        self.images_test = []
        self.features_train = []
        self.features_test = []
        self.segmentations_train = []
        self.segmentations_test = []
        self.metadata_train = []
        self.metadata_test = []
        self.semantics_file_train = []
        self.semantics_file_test = []
        self.radiomix_feature_file = None

        self.labels_file_train = None
        self.labels_file_test = None
        self.label_names = []

        self.fixed_splits = None

    def execute(self):
        """Execute the experiment.

        Before executing the actual experiment, this function will first run several validators
        and check the provided setup to make sure some of the most common
        made error are caught before running the experiment.
        """
        # this function is kind of like the build()-function in a builder, except it peforms execute on the object being built as well
        self._validate()  # do some final sanity checking before we execute the thing

        if self._fixed_splits:
            self._worc.fixedsplits = self._fixed_splits
        elif self.fixed_splits:
            self._worc.fixedsplits = self.fixed_splits

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

        # Training sources
        if self.images_train:
            self._worc.images_train = self.images_train
        elif self._images_train:
            self._worc.images_train = self._images_train

        if self.features_train:
            self._worc.features_train = self.features_train
        elif self._features_train:
            self._worc.features_train = self._features_train

        if self.segmentations_train:
            self._worc.segmentations_train = self.segmentations_train
        elif self._segmentations_train:
            self._worc.segmentations_train = self._segmentations_train

        if self.labels_file_train:
            self._worc.labels_train = self.labels_train
        elif self._labels_file_train:
            self._worc.labels_train = self._labels_file_train

        if self.semantics_file_train:
            self._worc.semantics_train = self.semantics_file_train
        elif self._semantics_file_train:
            self._worc.semantics_train = self._semantics_file_train

        # Testing sources
        if self.images_test:
            self._worc.images_test = self.images_test
        elif self._images_test:
            self._worc.images_test = self._images_test

        if self.features_test:
            self._worc.features_test = self.features_test
        elif self._features_test:
            self._worc.features_test = self._features_test

        if self.segmentations_test:
            self._worc.segmentations_test = self.segmentations_test
        elif self._segmentations_test:
            self._worc.segmentations_test = self._segmentations_test

        if self.labels_file_test:
            self._worc.labels_test = self.labels_test
        elif self._labels_file_test:
            self._worc.labels_test = self._labels_file_test

        if self.semantics_file_test:
            self._worc.semantics_test = self.semantics_file_test
        elif self._semantics_file_test:
            self._worc.semantics_test = self._semantics_file_test

        self._worc.label_names = ', '.join(self._label_names)
        self._config_builder._custom_overrides['Labels'] = dict()
        self._config_builder._custom_overrides['Labels']['label_names'] = self._worc.label_names

        # Find out how many configs we need to make
        if self._worc.images_train:
            nmod = len(self._worc.images_train)
        else:
            nmod = len(self.features_train)

        self._worc.configs = [self._config_builder.build_config(self._worc.defaultconfig())] * nmod
        self._worc.build()
        if self._add_evaluation:
            self._worc.add_evaluation(label_type=self._label_names[self._selected_label],
                                      modus=self._method)

        self._worc.set()
        self._worc.execute()
