from abc import ABC, abstractmethod

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
        return f"{self.__class__.__name__} validated: {validated_value}."

    @abstractmethod
    def _validate(self, *args, **kwargs):
        pass