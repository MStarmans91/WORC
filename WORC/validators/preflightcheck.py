from abc import ABC, abstractmethod

from WORC.processing.label_processing import load_label_csv
import WORC.addexceptions as ae

# Global variables
min_subjects = 10
recommended_subjects = 50


class AbstractValidator(ABC):
    # noinspection PyBroadException
    def do_validation(self, *args, **kwargs):
        # try:
        result = self._validate(*args, **kwargs)
        if result is None:
            result = True
        # except:
        #     result = False

        msg = self._generate_detector_message(result)
        if msg:
            print(msg)
        return result

    def _generate_detector_message(self, validated_value):
        return f"{self.__class__.__name__[0:-8]} validated: {validated_value}."

    @abstractmethod
    def _validate(self, *args, **kwargs):
        pass


class SimpleValidator(AbstractValidator):
    def _validate(self, simpleworc, *args, **kwargs):
        if not simpleworc._labels_file_train:
            if hasattr(simpleworc, 'labels_file_train'):
                if not simpleworc.labels_file_train:
                    raise ae.WORCValueError(f'No labels, use SimpleWorc().labels_from_this_file(**) to add labels.')
            else:
                raise ae.WORCValueError(f'No labels, use SimpleWorc().labels_from_this_file(**) to add labels.')

        if not simpleworc._label_names:
            if not simpleworc.label_names:
                raise ae.WORCValueError(f'No label(s) to predict selected. Use SimpleWorc().predict_labels(**) to select labels.')

        if not simpleworc._method:
            raise ae.WORCValueError(f'No method selected. Call function binary_classification(**) or regression(**) or survival(**) on SimpleWorc().')

        if simpleworc._images_train:
            for num, (ims, segs) in enumerate(zip(simpleworc._images_train, simpleworc._segmentations_train)):
                if ims.keys() != segs.keys():
                    raise ae.WORCValueError(f'Subjects in images_train and segmentations_train are not the same for modality {num}.')

        if hasattr(simpleworc, 'images_train'):
            if simpleworc.images_train:
                for num, (ims, segs) in enumerate(zip(simpleworc.images_train, simpleworc.segmentations_train)):
                    if ims.keys() != segs.keys():
                        raise ae.WORCValueError(f'Subjects in images_train and segmentations_train are not the same for modality {num}.')


class MinSubjectsValidator(AbstractValidator):
    def _validate(self, simpleworc, *args, **kwargs):
        if simpleworc._num_subjects < min_subjects:
            raise ae.WORCValueError(f'Less than {min_subjects} subjects (you have {simpleworc._num_subjects}) will probably make WORC crash due to a split in the test/validation set having only one subject. Use at least {min_subjects} subjects or more.')


class EvaluateValidator(AbstractValidator):
    def _validate(self, simpleworc, *args, **kwargs):
        if simpleworc._add_evaluation:
            if not simpleworc._images_train:
                if hasattr(simpleworc, 'images_train'):
                    if not simpleworc.images_train:
                        raise ae.WORCValueError(f'You have added the evaluation pipeline, but have not provided images, which is currently required. We will work on this option in a future release.')
                else:
                    raise ae.WORCValueError(f'You have added the evaluation pipeline, but have not provided images, which is currently required. We will work on this option in a future release.')


class SamplesWarning(AbstractValidator):
    # Not really a validator, but more a good practice. Hence this won't throw an exception but prints a warning instead.
    def _validate(self, simpleworc, *args, **kwargs):
        if simpleworc._method == 'classification':
            if simpleworc._num_subjects < len(simpleworc._label_names) * recommended_subjects: # at least 100 subjects per label recommended
                print(f'Warning: at least {len(simpleworc._label_names) * recommended_subjects} subjects is recommended when predicting {len(simpleworc._label_names)} labels. Current subject count is: {simpleworc._num_subjects}')
        elif simpleworc._method == 'regression':
            # TODO @martijn not sure how to tackle this, what would be a reasonable amount of subjects for regression?
            pass


class InvalidLabelsValidator(AbstractValidator):
    def _validate(self, simpleworc):
        errstr = None

        try:
            if simpleworc._labels_file_train:
                labels, subjects, _ = load_label_csv(simpleworc._labels_file_train)
            elif simpleworc.labels_file_train:
                labels, subjects, _ = load_label_csv(simpleworc.labels_file_train)
            else:
                raise ae.WORCValueError(f'No labels, use SimpleWorc().labels_from_this_file(**) to add labels.')

        except ae.WORCAssertionError as wae:
            if 'First column should be patient ID' in str(wae):
                # TODO: print wrong column name and file so that it is clear what needs to be replaced in which file
                raise ae.WORCValueError(f'First column in the file given to SimpleWORC().labels_from_this_file(**) needs to be named Patient.')

        # check labels for substrings of eachother
        labels_matches = self._get_all_substrings_for_array(labels)

        if labels_matches:
            # if not empty we have a problem
            errstr = "Found label(s) that are a substring of other label(s). This is currently not allowed in WORC. Rename the following label(s):\n"
            for label, matches in labels_matches.items():
                for match in matches:
                    errstr += f"{label} is a substring of {match}\n"

        # check subject names for substrings of eachother
        subjects_matches = self._get_all_substrings_for_array(subjects)
        if subjects_matches:
            # if not empty we have a problem
            errstr = "Found subject(s) that are a substring of other subject(s). This is currently not allowed in WORC. Rename the following subject(s):\n"
            for subject, matches in subjects_matches.items():
                for match in matches:
                    errstr += f"{subject} is a substring of {match}\n"

        if errstr:
            raise ae.WORCValueError(errstr)

    def _get_all_substrings_for_array(self, arr):
        # generate a dict with substrings of each element in array
        all_matches = {}
        for strcmp in arr:
            matches = [s for s in arr if s != strcmp and strcmp in s]
            if matches:
                all_matches[strcmp] = matches

        return all_matches


class ValidatorsFactory:
    @staticmethod
    def factor_validators():
        return [
            SimpleValidator(),
            MinSubjectsValidator(),
            SamplesWarning(),
            EvaluateValidator(),
            InvalidLabelsValidator()
        ]


__all__ = [ValidatorsFactory]
