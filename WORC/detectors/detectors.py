from abc import ABC, abstractmethod
from pathlib import Path

import platform


class AbstractDetector(ABC):
    # noinspection PyBroadException
    def do_detection(self, *args, **kwargs):
        try:
            return self.is_detected(*args, **kwargs)
        except:
            return False

    @abstractmethod
    def is_detected(self, *args, **kwargs):
        pass


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
        if platform.system.lower().strip() == 'linux':
            return True
        return False