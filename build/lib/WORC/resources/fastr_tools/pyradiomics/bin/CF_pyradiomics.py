#!/usr/bin/env python

# Copyright 2017-2018 Biomedical Imaging Group Rotterdam, Departments of
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

import logging
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor


def AllFeatures(image, mask):
    # Define settings for signature calculation
    # These are currently set equal to the respective default values
    kwargs = {}
    kwargs['binWidth'] = 25
    kwargs['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
    kwargs['interpolator'] = sitk.sitkBSpline
    kwargs['verbose'] = True

    # Initialize wrapperClass to generate signature
    extractor = featureextractor.RadiomicsFeaturesExtractor(**kwargs)

    # Disable all classes except firstorder
    extractor.enableAllFeatures()

    # Enable writing out the log using radiomics logger
    radiomics.debug()  # Switch on radiomics logging from level=DEBUG (default level=WARNING)

    # Prevent radiomics logger from printing out log entries with level < WARNING to the console
    logger = logging.getLogger('radiomics')
    logger.handlers[0].setLevel(logging.WARNING)
    logger.propagate = False  # Do not pass log messages on to root logger

    # Write out all log entries to a file
    handler = logging.FileHandler(filename='testLog.txt', mode='w')
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    print("Active features:")
    for cls, features in extractor.enabledFeatures.iteritems():
        if len(features) == 0:
            features = extractor.getFeatureNames(cls)
        for f in features:
            print(f)
            print(eval('extractor.featureClasses["%s"].get%sFeatureValue.__doc__' % (cls, f)))

    print("Calculating features")
    featureVector = extractor.execute(image, mask)

    for featureName in featureVector.keys():
        print("Computed %s: %s" % (featureName, featureVector[featureName]))

    return featureVector
