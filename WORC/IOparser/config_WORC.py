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


import configparser
import os
import WORC.addexceptions as ae


def load_config(config_file_path):
    """ Parse a WORC configuration file.

    Arguments:
        config_file_path: path to the configuration file to be parsed.

    Returns:
        settings_dict: dictionary containing all parsed settings.
    """
    if not os.path.exists(config_file_path):
        e = f'File {config_file_path} does not exist!'
        raise ae.WORCKeyError(e)

    settings = configparser.ConfigParser()
    settings.read(config_file_path)

    settings_dict = {'ImageFeatures': dict(), 'General': dict(),
                     'SVMFeatures': dict()}

    settings_dict['ImageFeatures']['image_type'] =\
        str(settings['ImageFeatures']['image_type'])

    settings_dict['General']['FeatureCalculators'] =\
        [str(item).strip() for item in
         settings['General']['FeatureCalculators'].split(',')]

    settings_dict['General']['Preprocessing'] =\
        str(settings['General']['Preprocessing'])

    settings_dict['General']['PCE'] =\
        settings['General'].getboolean('PCE')

    settings_dict['General']['Segmentix'] =\
        settings['General'].getboolean('Segmentix')

    return settings_dict
