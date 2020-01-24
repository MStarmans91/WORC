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

import argparse
from PREDICT.CalcFeatures import CalcFeatures
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Feature extraction')
    parser.add_argument('-im', '--im', metavar='image', nargs='+',
                        dest='im', type=str, required=True,
                        help='Images to calculate features on')
    parser.add_argument('-md', '--md', metavar='metadata', dest='md',
                        type=str, required=False, nargs='+',
                        help='Clinical data on patient (DICOM)')
    parser.add_argument('-sem', '--sem', metavar='semantics', dest='sem',
                        type=str, required=False, nargs='+',
                        help='Semantic Features')
    parser.add_argument('-seg', '--seg', metavar='segmentation', dest='seg',
                        type=str, required=True, nargs='+',
                        help='Segmentation to calculate features on')
    parser.add_argument('-para', '--para', metavar='Parameters', nargs='+',
                        dest='para', type=str, required=True,
                        help='Parameters')
    parser.add_argument('-out', '--out', metavar='Features',
                        dest='out', type=str, required=False,
                        help='Patient features output (HDF)')
    args = parser.parse_args()

    if 'Dummy' in str(args.im):
        # Image is a dummy, so we write a feature file without features
        panda_labels = ['image_type', 'parameters', 'feature_values',
                        'feature_labels']
        image_type = 'Dummy'
        parameters = list()
        feature_labels = list()
        feature_values = list()
        panda_data = pd.Series([image_type, parameters, feature_values,
                                feature_labels],
                               index=panda_labels,
                               name='Image features'
                               )

        print('Saving image features')
        panda_data.to_hdf(args.out, 'image_features')
    else:
        if type(args.para) is list:
            # Parameters is a single argument
            args.para = args.para[0]

        CalcFeatures(image=args.im, segmentation=args.seg, parameters=args.para,
                     output=args.out, metadata_file=args.md,
                     semantics_file=args.sem)


if __name__ == '__main__':
    main()
