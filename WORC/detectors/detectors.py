import csv
import string
from abc import ABC, abstractmethod
from pathlib import Path
from os import environ
import platform


class AbstractDetector(ABC):
    # noinspection PyBroadException
    def do_detection(self, *args, **kwargs):
        try:
            result = self._is_detected(*args, **kwargs)
        except:
            result = False
        print(self._generate_detector_message(result))
        return result

    def _generate_detector_message(self, detected_Value):
        return f"{self.__class__.__name__[0:-8]} detected: {detected_Value}."

    @abstractmethod
    def _is_detected(self, *args, **kwargs):
        pass


class CsvDetector(AbstractDetector):
    def __init__(self, csv_file_path):
        self._csv_file_path = csv_file_path

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
