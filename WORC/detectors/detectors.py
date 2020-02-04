import csv
import string
from abc import ABC, abstractmethod
from pathlib import Path
from os import environ
import platform
import pandas as pd

# maintain a detector cache in-memory
_detector_cache = {}

class AbstractDetector(ABC):
    # noinspection PyBroadException
    def do_detection(self, *args, **kwargs):
        if self._is_cached() and self.__class__.__name__ in _detector_cache:
            return _detector_cache[self.__class__.__name__]

        try:
            result = self._is_detected(*args, **kwargs)
        except:
            result = False
        print(self._generate_detector_message(result))

        if self._is_cached():
            _detector_cache[self.__class__.__name__] = result

        return result

    def _generate_detector_message(self, detected_value):
        return f"{self.__class__.__name__[0:-8]} detected: {detected_value}."

    def _is_cached(self):
        return True

    @abstractmethod
    def _is_detected(self, *args, **kwargs):
        pass


class CsvDetector(AbstractDetector):
    def __init__(self, csv_file_path):
        self._csv_file_path = csv_file_path

    def _is_cached(self):
        return False

    def _is_detected(self, *args, **kwargs):
        try:
            with open(self._csv_file_path, newline='') as csvfile:
                start = csvfile.read(4096)

                # isprintable does not allow newlines, printable does not allow umlauts...
                if not all([c in string.printable or c.isprintable() for c in start]):
                    return False
                dialect = csv.Sniffer().sniff(start) # this triggers csv.Error if it can't sniff the csv dialect
                return True
        except csv.Error:
            # Could not get a csv dialect -> probably not a csv.
            return False


class SurvivalFileDetector(AbstractDetector):
    def __init__(self, survival_file_path, time, event):
        self._survival_file_path = survival_file_path
        self._columns = [time, event]

    def _is_cached(self):
        return False

    def _is_detected(self, *args, **kwargs):
        if not CsvDetector(self._survival_file_path).do_detection():
            return False

        df = pd.read_csv(self._survival_file_path)
        self._found_columns = ', '.join(df.columns)

        return len(df.columns) == 3 and df.columns[0] == 'Patient' and df.columns[1] in self._columns and df.columns[2] in self._columns

    def _generate_detector_message(self, detected_value):
        if detected_value:
            return f"Survival file {self._survival_file_path} seems valid"
        else:
            return f"Survival file {self._survival_file_path} is not valid, expected columns Patient, {self._columns[0]}, {self._columns[1]} but instead got {self._found_columns}"


class CartesiusClusterDetector(AbstractDetector):
    def _is_detected(self):
        if LinuxDetector()._is_detected():
            try:
                if 'cartesius' in Path('/etc/hosts').read_text():
                    return True
            except:
                return False
        return False


class DebugDetector(AbstractDetector):
    def _is_cached(self):
        return False

    def _is_detected(self):
        try:
            if environ.get('WORCDEBUG') is not None:
                return True
            else:
                return False
        except:
            return False


class BigrClusterDetector(AbstractDetector):
    def _is_detected(self):
        if LinuxDetector()._is_detected():
            try:
                if 'bigr-cluster' in Path('/etc/hosts').read_text():
                    return True
            except:
                return False
        return False


class HostnameDetector(AbstractDetector):
    def _is_detected(self):
        if platform.node() == self._expected_hostname:
            return True
        return False

    def __init__(self, expected_hostname):
        self._expected_hostname = expected_hostname


class LinuxDetector(AbstractDetector):
    def _is_detected(self):
        if platform.system().lower().strip() == 'linux':
            return True
        return False
