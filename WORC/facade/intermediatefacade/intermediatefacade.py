from enum import Enum

from WORC import WORC


from pathlib import Path

from WORC.facade.intermediatefacade.configbuilder import ConfigBuilder
from .exceptions import PathNotFoundException, NoImagesFoundException, NoSegmentationsFoundException


class IntermediateFacade():
    def __init__(self, name='WORC'):
        # Set some config values
        self._worc = WORC(name)
        self._config_builder = ConfigBuilder()
        self._config_builder.cluster_config_overrides()

    def images_from_this_directory(self, directory, image_file_name='image.nii.gz', glob='*/', is_training=True):
        directory = Path(directory).expanduser()
        if not directory.exists():
            raise PathNotFoundException(directory)

        images = list(directory.glob(f'{glob}{image_file_name}'))

        if len(images) == 0:
            raise NoImagesFoundException(f'{directory}{glob}{image_file_name}')

        if is_training:
            self._worc.images_train = [{image.parent.name: str(image.absolute()) for image in images}]
        else:
            self._worc.images_test = [{image.parent.name: str(image.absolute()) for image in images}]


    def segmentations_from_this_directory(self, directory, segmentation_file_name='segmentation.nii.gz', glob='*/', is_training=True):
        directory = Path(directory).expanduser()
        if not directory.exists():
            raise PathNotFoundException(directory)

        segmentations = list(directory.glob(f'{glob}{segmentation_file_name}'))

        if len(segmentations) == 0:
            raise NoSegmentationsFoundException(str(directory))

        if is_training:
            self._worc.segmentations_train = [{image.parent.name: str(image.absolute()) for image in segmentations}]
        else:
            self._worc.segmentations_test = [{image.parent.name: str(image.absolute()) for image in segmentations}]


    def labels_from_this_file(self, file_path, is_training=True):
        labels_file = Path(file_path).expanduser()

        if not labels_file.is_file():
            raise PathNotFoundException(file_path)

        # TODO: implement sanity check labels file e.g. is it a labels file and are there labels available
        if is_training:
            self._worc.labels_train = labels_file.absolute()
        else:
            self._worc.labels_test = labels_file.absolute()


    def predict_labels(self, label_names: list):
        if not isinstance(label_names, list):
            raise TypeError(f'label_names is of type {type(label_names)} while list is expected')

        for label in label_names:
            if len(label.strip()) == 0:
                raise ValueError('Invalid label, length = 0')

        # TODO: check if labels is in labels file

        self._worc.label_names = ', '.join(label_names)

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
                raise ValueError(f'Invalid estimator {estimator} for {method}; must be one of {", ".join(valid_estimators)}')

        # TODO: sanity check scoring method per estimator

        # set
        self._config_builder.estimator_scoring_overrides(estimators, scoring_method)

        if coarse:
            self._config_builder.coarse_overrides()
        else:
            self._config_builder.full_overrides()

        config = self._config_builder.build_config(self._worc.defaultconfig())
        self._worc.configs = [config]

    def execute(self):
        self._worc.build()
        self._worc.set()
        self._worc.execute()

    def binary_classification(self, estimators=['SVM'], scoring_method='f1', coarse=True):
        self._set_estimators(estimators, scoring_method, 'classification', coarse)

    def regression(self, estimators=['SVR'], scoring_method='r2', coarse=True):
        self._set_estimators(estimators, scoring_method, 'regression', coarse)

    def survival(self, estimators, scoring_method, coarse=True):
        raise NotImplementedError()

    def add_config_overrides(self, config):
        self._config_builder.custom_config_overrides(config)
