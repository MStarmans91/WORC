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

import csv
import string
from abc import ABC, abstractmethod
from pathlib import Path
from os import environ
import platform
import os
import pkg_resources
import site
import sys


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


class WORCDirectoryDetector(AbstractDetector):
    def _is_detected(self):
        # Get directory in which WORC package is installed
        working_set = pkg_resources.working_set
        requirement_spec = pkg_resources.Requirement.parse('WORC')
        egg_info = working_set.find(requirement_spec)
        if egg_info is None:  # Backwards compatibility with WORC2
            try:
                packagedir = site.getsitepackages()[0]
            except AttributeError:
                # Inside virtualenvironment, so getsitepackages doesnt work.
                paths = sys.path
                for p in paths:
                    if os.path.isdir(p) and os.path.basename(p) == 'site-packages':
                        packagedir = p
        else:
            packagedir = egg_info.location

        packagedir = os.path.join(packagedir, 'WORC')
        return packagedir
