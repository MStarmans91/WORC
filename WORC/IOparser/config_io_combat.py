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
    """
    Load the config ini, parse settings to WORC

    Args:
        config_file_path (String): path of the .ini config file

    Returns:
        settings_dict (dict): dict with the loaded settings
    """
    if not os.path.exists(config_file_path):
        e = f'File {config_file_path} does not exist!'
        raise ae.WORCKeyError(e)

    settings = configparser.ConfigParser()
    settings.read(config_file_path)

    settings_dict = {'ComBat': dict()}

    # Convert settings
    settings_dict['ComBat']['batch'] =\
        [str(item).strip() for item in
         settings['ComBat']['batch'].split(',')]

    settings_dict['ComBat']['mod'] =\
        [str(item).strip() for item in
         settings['ComBat']['mod'].split(',')]

    settings_dict['ComBat']['par'] =\
        settings['ComBat'].getint('par')

    settings_dict['ComBat']['eb'] =\
        settings['ComBat'].getint('eb')

    settings_dict['ComBat']['language'] =\
        str(settings['ComBat']['language'])

    settings_dict['ComBat']['matlab'] =\
        str(settings['ComBat']['matlab'])

    settings_dict['ComBat']['per_feature'] =\
        int(settings['ComBat']['per_feature'])

    settings_dict['ComBat']['excluded_features'] =\
        [str(item).strip() for item in
         settings['ComBat']['excluded_features'].split(',')]

    return settings_dict
