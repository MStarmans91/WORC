import csv
import string
from abc import ABC, abstractmethod
from pathlib import Path

import platform


class AbstractDetector(ABC):
    # noinspection PyBroadException
    def do_detection(self, *args, **kwargs):
        print(self.is_detected(*args, **kwargs))
        try:
            return self.is_detected(*args, **kwargs)
        except:
            return False

    @abstractmethod
    def is_detected(self, *args, **kwargs):
        pass


class CsvDetector(AbstractDetector):
    def __init__(self, csv_file_path):
        self._csv_file_path = csv_file_path

    def is_detected(self, *args, **kwargs):
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
    def is_detected(self):
        if LinuxDetector().is_detected():
            try:
                if 'cartesius' in Path('/etc/hosts').read_text():
                    return True
            except:
                return False
        return False


class BigrClusterDetector(AbstractDetector):
    def is_detected(self):
        if LinuxDetector().is_detected():
            try:
                if 'bigr-cluster' in Path('/etc/hosts').read_text():
                    return True
            except:
                return False
        return False


class HostnameDetector(AbstractDetector):
    def is_detected(self):
        if platform.node() == self._expected_hostname:
            return True
        return False

    def __init__(self, expected_hostname):
        self._expected_hostname = expected_hostname


class LinuxDetector(AbstractDetector):
    def is_detected(self):
        if platform.system().lower().strip() == 'linux':
            return True
        return False
