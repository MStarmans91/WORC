#!/usr/bin/env python

# Copyright 2016-2022 Biomedical Imaging Group Rotterdam, Departments of
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

import SimpleITK as sitk
import argparse
from WORC.processing.preprocessing import preprocess
import numpy as np
from WORC.tests import test_helpers as th
from WORC import WORC


def test_preprocessing():
    image = th.create_SimpeITKimage(size=10)
    mask = th.create_SimpeITKimage_mask(size=10, radius=3)
    config = WORC().defaultconfig()
    
    # Test all preprocessing options
    config['ImageFeatures']['image_type'] = 'MRI'
    config['Preprocessing']['CheckSpacing'] = 'False'
    config['Preprocessing']['Clipping'] = 'True'
    config['Preprocessing']['Clipping_Range'] = '-1000.0, 3000.0'
    config['Preprocessing']['Normalize'] = 'True'
    config['Preprocessing']['Normalize_ROI'] = 'Full'
    config['Preprocessing']['ROIdilate'] = 'True'
    config['Preprocessing']['Resampling'] = 'True'
    config['Preprocessing']['BiasCorrection'] = 'True'
    config['Preprocessing']['CheckOrientation'] = 'True'
    config['Preprocessing']['HistogramEqualization'] = 'True'
    configfile = 'testconfig.ini'
    with open(configfile, 'w') as cfile:
        config.write(cfile)
    
    # Perform test    
    preprocess(imagefile=image, config=configfile, mask=mask)
    
    
if __name__ == "__main__":
    test_preprocessing()