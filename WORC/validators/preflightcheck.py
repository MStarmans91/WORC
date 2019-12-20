from abc import ABC, abstractmethod

from WORC.processing.label_processing import load_label_csv
import WORC.addexceptions as ae

class AbstractValidator(ABC):
    # noinspection PyBroadException
    def do_validation(self, *args, **kwargs):
        try:
            result = self._validate(*args, **kwargs)
            if result is None:
                result = True
        except:
            result = False

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
            raise ValueError(f'No labels, use SimpleWorc().labels_from_this_file(**) to add labels.')

        if not simpleworc._label_names:
            raise ValueError(f'No label(s) to predict selected. Use SimpleWorc().predict_labels(**) to select labels.')

        if not simpleworc._method:
            raise ValueError(f'No method selected. Call function binary_classification(**) or regression(**) or survival(**) on SimpleWorc().')

        if len(simpleworc._images_train) == len(simpleworc._segmentations_train):
            for key, subjects_dict in self._images_train.items():
                if subjects_dict.keys() != self._segmentations_train[key].keys():
                    raise ValueError('Subjects in images_train and segmentations_train are not the same')

class MinSubjectsValidator(AbstractValidator):
    def _validate(self, simpleworc, *args, **kwargs):
        minsubjects = 10
        if simpleworc.count_num_subjects() <= minsubjects:
            raise ValueError(f'Less than {minsubjects + 1} subjects will make WORC crash due to a split in the validation set having only one subject. Add at least {minsubjects + 1} subjects or more.')

class SamplesWarning(AbstractValidator):
    # Not really a validator, but more a good practice. Hence this won't throw an exception but prints a warning instead.
    def _validate(self, simpleworc, *args, **kwargs):
        if simpleworc._method == 'classification':
            if simpleworc.count_num_subjects() < len(simpleworc._label_names) * 100 # at least 100 subjects per label recommended
                print(f'Warning: at least {len(simpleworc._label_names) * 100} subjects is recommended when predicting {len(simpleworc._label_names)} labels. Current subject count is: {simpleworc.count_num_subjects()}')
        elif simpleworc._method == 'regression':
            # TODO @martijn not sure how to tackle this, what would be a reasonable amount of subjects for regression?
            pass

class InvalidLabelsValidator(AbstractValidator):
    def _validate(self, simpleworc):
        try:
            labels, subjects, _ = load_label_csv(simpleworc._labels_file_train)
        except ae.WORCAssertionError as wae:
            if 'First column should be patient ID' in str(wae):
                # TODO: print wrong column name and file so that it is clear what needs to be replaced in which file
                raise ValueError(f'First column in the file given to SimpleWORC().labels_from_this_file(**) needs to be named Patient.')

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
        if labels_matches:
            # if not empty we have a problem
            errstr = "Found subject(s) that are a substring of other subject(s). This is currently not allowed in WORC. Rename the following subject(s):\n"
            for subject, matches in subjects_matches.items():
                for match in matches:
                    errstr += f"{label} is a substring of {match}\n"

        if errstr:
            raise ValueError(errstr)


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
            SamplesWarning()
        ]

__all__ = [ValidatorsFactory]

