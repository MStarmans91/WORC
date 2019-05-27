from WORC import WORC
#from fastr.core.vfs import VirtualFileSystem

from abc import ABC, abstractmethod

from pathlib import Path

from .exceptions import PathNotFoundException, NoImagesFoundException, NoSegmentationsFoundException


class AbstractWorcBuilder(ABC):
    def __init__(self, name='WORC', cpucores=1, joblib_backend='threading', hyperoptimization_iterations=100000, hyperoptimization_jobspercore=4000, crossvalidation_iterations=100):
        # Set some config values
        self._worc = WORC(name)
        #self._vfs = VirtualFileSystem()
        self._image_sources = None
        self._segmentation_sources = None
        self._config = {
            'PREDICTGeneral': {
                'Joblib_ncores': cpucores,
                'Joblib_backend': joblib_backend
            },
            'Classification': {
                'fastr': True
            },
            'HyperOptimization': {
                'N_iterations': hyperoptimization_iterations,
                'n_jobspercore': hyperoptimization_jobspercore
            },
            'CrossValidation': {
                'N_iterations': crossvalidation_iterations
            }
        }

    def using_images_and_segmentations_from_this_directory(self, directory, image_file_name='image.nii.gz', segmentation_file_name='segmentation.nii', glob='*/'):
        pass

    def using_images_from_this_directory(self, directory, image_file_name='image.nii.gz', glob='*/'):
        directory = Path(directory).expanduser()
        if not directory.exists():
            raise PathNotFoundException(directory)

        images = list(directory.glob(f'{glob}{image_file_name}'))

        if len(images) == 0:
            raise NoImagesFoundException(f'{directory}{glob}{image_file_name}')

        self._image_sources = {image.parent.name: str(image.absolute()) for image in images}

        return self

    def using_segmentations_from_this_directory(self, directory, segmentation_file_name='segmentation.nii.gz', glob='*/'):
        directory = Path(directory).expanduser()
        if not directory.exists():
            raise PathNotFoundException(directory)

        segmentations = list(directory.glob(f'{glob}{segmentation_file_name}'))

        if len(segmentations) == 0:
            raise NoSegmentationsFoundException(str(directory))

        self._segmentation_sources = {image.parent.name: str(image.absolute()) for image in segmentations}

        return self