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

    settings_dict = {'Preprocessing': dict(), 'ImageFeatures': dict(),
                     'General': dict()}

    # General settings
    settings_dict['ImageFeatures']['image_type'] =\
        [str(item).strip() for item in
         settings['ImageFeatures']['image_type'].split(',')]

    settings_dict['General']['AssumeSameImageAndMaskMetadata'] =\
        settings['General'].getboolean('AssumeSameImageAndMaskMetadata')

    # Detect incorrect spacing
    settings_dict['Preprocessing']['CheckSpacing'] =\
        settings['Preprocessing'].getboolean('CheckSpacing')

    # Clipping
    settings_dict['Preprocessing']['Clipping'] =\
        settings['Preprocessing'].getboolean('Clipping')

    settings_dict['Preprocessing']['Clipping_Range'] =\
        [float(item) for item in
         settings['Preprocessing']['Clipping_Range'].split(',')]

    if len(settings_dict['Preprocessing']['Clipping_Range']) != 2:
        raise ae.WORCValueError(f"Clipping range should be two floats split by a comma, got {settings['Preprocessing']['Clipping_Range']}.")

    # Normalization
    settings_dict['Preprocessing']['Normalize'] =\
        settings['Preprocessing'].getboolean('Normalize')

    settings_dict['Preprocessing']['Normalize_ROI'] =\
        str(settings['Preprocessing']['Normalize_ROI'])

    settings_dict['Preprocessing']['ROIdilate'] =\
        str(settings['Preprocessing']['ROIdilate'])

    settings_dict['Preprocessing']['ROIDetermine'] =\
        str(settings['Preprocessing']['ROIDetermine'])

    settings_dict['Preprocessing']['ROIdilateradius'] =\
        int(settings['Preprocessing']['ROIdilateradius'])

    settings_dict['Preprocessing']['Method'] =\
        str(settings['Preprocessing']['Method'])

    # Bias Correction
    settings_dict['Preprocessing']['BiasCorrection'] =\
        settings['Preprocessing'].getboolean('BiasCorrection')

    settings_dict['Preprocessing']['BiasCorrection_Mask'] =\
        settings['Preprocessing'].getboolean('BiasCorrection_Mask')

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
