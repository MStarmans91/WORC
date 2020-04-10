#!/usr/bin/env python

# Copyright 2011-2017 Biomedical Imaging Group Rotterdam, Departments of
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
import numpy as np
import os
import PREDICT.addexceptions as ae


def load_config(config_file_path):
    if not os.path.exists(config_file_path):
        e = f'File {config_file_path} does not exist!'
        raise ae.PREDICTKeyError(e)

    settings = configparser.ConfigParser()
    settings.read(config_file_path)

    settings_dict = {'ImageFeatures': dict(), 'DataPaths': dict(),
                     'General': dict()}

    # Extract some general settings
    settings_dict['General']['Joblib_ncores'] =\
        settings['General'].getint('Joblib_ncores')

    settings_dict['General']['Joblib_backend'] =\
        str(settings['General']['Joblib_backend'])

    # Extract image feature specific settings
    settings_dict['ImageFeatures']['image_type'] =\
        [str(item).strip() for item in
         settings['ImageFeatures']['image_type'].split(',')]

    settings_dict['ImageFeatures']['shape'] =\
        settings['ImageFeatures'].getboolean('shape')

    settings_dict['ImageFeatures']['histogram'] =\
        settings['ImageFeatures'].getboolean('histogram')

    settings_dict['ImageFeatures']['orientation'] =\
        settings['ImageFeatures'].getboolean('orientation')

    settings_dict['ImageFeatures']['texture_Gabor'] =\
        settings['ImageFeatures'].getboolean('texture_Gabor')

    settings_dict['ImageFeatures']['texture_GLCM'] =\
        settings['ImageFeatures'].getboolean('texture_GLCM')

    settings_dict['ImageFeatures']['texture_GLCMMS'] =\
        settings['ImageFeatures'].getboolean('texture_GLCMMS')

    settings_dict['ImageFeatures']['texture_GLRLM'] =\
        settings['ImageFeatures'].getboolean('texture_GLRLM')

    settings_dict['ImageFeatures']['texture_GLSZM'] =\
        settings['ImageFeatures'].getboolean('texture_GLSZM')

    settings_dict['ImageFeatures']['texture_NGTDM'] =\
        settings['ImageFeatures'].getboolean('texture_NGTDM')

    settings_dict['ImageFeatures']['texture_LBP'] =\
        settings['ImageFeatures'].getboolean('texture_LBP')

    settings_dict['ImageFeatures']['coliage'] =\
        settings['ImageFeatures'].getboolean('coliage')

    settings_dict['ImageFeatures']['vessel'] =\
        settings['ImageFeatures'].getboolean('vessel')

    settings_dict['ImageFeatures']['log'] =\
        settings['ImageFeatures'].getboolean('log')

    settings_dict['ImageFeatures']['phase'] =\
        settings['ImageFeatures'].getboolean('phase')

    # Parameters for computing features
    settings_dict['ImageFeatures']['parameters'] = dict()

    # Gabor settings
    settings_dict['ImageFeatures']['parameters']['gabor_settings'] = dict()

    gabor_frequencies = [str(item).strip() for item in
                         settings['ImageFeatures']['gabor_frequencies']
                         .split(',')]
    gabor_frequencies = np.asarray(gabor_frequencies).astype(np.float)

    gabor_angles = [str(item).strip() for item in
                    settings['ImageFeatures']['gabor_angles']
                    .split(',')]
    gabor_angles = np.asarray(gabor_angles).astype(np.float)
    # Convert gabor angle to radians from angles
    gabor_angles = np.radians(gabor_angles)

    settings_dict['ImageFeatures']['parameters']['gabor_settings']['gabor_frequencies'] =\
        gabor_frequencies

    settings_dict['ImageFeatures']['parameters']['gabor_settings']['gabor_angles'] =\
        gabor_angles

    # GLCM
    settings_dict['ImageFeatures']['parameters']['GLCM'] = dict()

    settings_dict['ImageFeatures']['parameters']['GLCM']['levels'] =\
        int(settings['ImageFeatures']['GLCM_levels'])

    GLCM_angles = [str(item).strip() for item in
                   settings['ImageFeatures']['GLCM_angles']
                   .split(',')]
    settings_dict['ImageFeatures']['parameters']['GLCM']['angles'] =\
        [float(g) for g in GLCM_angles]

    GLCM_distances = [str(item).strip() for item in
                      settings['ImageFeatures']['GLCM_distances']
                      .split(',')]
    settings_dict['ImageFeatures']['parameters']['GLCM']['distances'] =\
        [float(g) for g in GLCM_distances]

    # LBP
    settings_dict['ImageFeatures']['parameters']['LBP'] = dict()

    LBP_radius = [str(item).strip() for item in
                  settings['ImageFeatures']['LBP_radius']
                  .split(',')]
    settings_dict['ImageFeatures']['parameters']['LBP']['radius'] =\
        [int(g) for g in LBP_radius]

    LBP_npoints = [str(item).strip() for item in
                   settings['ImageFeatures']['LBP_npoints']
                   .split(',')]
    settings_dict['ImageFeatures']['parameters']['LBP']['N_points'] =\
        [int(g) for g in LBP_npoints]

    # Phase features
    settings_dict['ImageFeatures']['parameters']['phase'] = dict()

    phase_minwavelength = [str(item).strip() for item in
                           settings['ImageFeatures']['phase_minwavelength']
                           .split(',')]
    settings_dict['ImageFeatures']['parameters']['phase']['minwavelength'] =\
        [int(g) for g in phase_minwavelength]

    phase_nscale = [str(item).strip() for item in
                    settings['ImageFeatures']['phase_nscale']
                    .split(',')]
    settings_dict['ImageFeatures']['parameters']['phase']['nscale'] =\
        [int(g) for g in phase_nscale]

    # log features
    settings_dict['ImageFeatures']['parameters']['log'] = dict()

    log_sigma = [str(item).strip() for item in
                 settings['ImageFeatures']['log_sigma']
                 .split(',')]
    settings_dict['ImageFeatures']['parameters']['log']['sigma'] =\
        [int(g) for g in log_sigma]

    # vessel features
    settings_dict['ImageFeatures']['parameters']['vessel'] = dict()

    vessel_scale_range = [str(item).strip() for item in
                          settings['ImageFeatures']['vessel_scale_range']
                          .split(',')]
    settings_dict['ImageFeatures']['parameters']['vessel']['scale_range'] =\
        [float(g) for g in vessel_scale_range]

    vessel_scale_step = [str(item).strip() for item in
                         settings['ImageFeatures']['vessel_scale_step']
                         .split(',')]
    settings_dict['ImageFeatures']['parameters']['vessel']['scale_step'] =\
        [float(g) for g in vessel_scale_step]

    settings_dict['ImageFeatures']['parameters']['vessel']['radius'] =\
        int(settings['ImageFeatures']['vessel_radius'])

    return settings_dict
