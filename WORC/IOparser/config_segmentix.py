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
    """ Parse a segmentix configuration file.

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

    settings_dict = {'Segmentix': dict(), 'Preprocessing': dict()}

    # Segmentation settings
    settings_dict['Sementix'] = dict()
    settings_dict['Segmentix']['type'] =\
        str(settings['Segmentix']['segtype'])

    settings_dict['Segmentix']['mask'] =\
        str(settings['Segmentix']['mask'])

    settings_dict['Segmentix']['radius'] =\
        int(settings['Segmentix']['segradius'])

    settings_dict['Segmentix']['N_blobs'] =\
        int(settings['Segmentix']['N_blobs'])

    settings_dict['Segmentix']['fillholes'] =\
        settings['Segmentix'].getboolean('fillholes')

    settings_dict['Segmentix']['remove_small_objects'] =\
        settings['Segmentix'].getboolean('remove_small_objects')

    settings_dict['Segmentix']['min_object_size'] =\
        int(settings['Segmentix']['min_object_size'])

    settings_dict['Segmentix']['AssumeSameImageAndMaskMetadata'] =\
        settings['General'].getboolean('AssumeSameImageAndMaskMetadata')

    # Check spacing
    settings_dict['Preprocessing']['CheckSpacing'] =\
        settings['Preprocessing'].getboolean('CheckSpacing')

    # Re-orientation
    settings_dict['Preprocessing']['CheckOrientation'] =\
        settings['Preprocessing'].getboolean('CheckOrientation')

    settings_dict['Preprocessing']['OrientationPrimaryAxis'] =\
        str(settings['Preprocessing']['OrientationPrimaryAxis'])
        
    # Resampling
    settings_dict['Preprocessing']['Resampling'] =\
        settings['Preprocessing'].getboolean('Resampling')

    settings_dict['Preprocessing']['Resampling_spacing'] =\
        [float(item) for item in
         settings['Preprocessing']['Resampling_spacing'].split(',')]

    if len(settings_dict['Preprocessing']['Resampling_spacing']) != 3:
        s = settings_dict['Preprocessing']['Resampling_spacing']
        raise ae.WORCValueError(f'Resampling spacing should be three elements, got {s}')

    return settings_dict
