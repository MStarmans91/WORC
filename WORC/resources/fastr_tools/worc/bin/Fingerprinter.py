#!/usr/bin/env python

# Copyright 2017 - 2022 Biomedical Imaging Group Rotterdam, Departments of
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

import yaml
import argparse
from WORC.tools.fingerprinting import Fingerprinter
import WORC.IOparser.file_io as io


def main():
    parser = argparse.ArgumentParser(description='WORC Fingerprinter')
    parser.add_argument('-images_train', '--images_train', metavar='images_train',
                        nargs='+', dest='images_train', type=str, required=False,
                        help='Input images of one modality (ITK Image)')
    parser.add_argument('-segmentations_train', '--segmentations_train', metavar='segmentations_train',
                        nargs='+', dest='segmentations_train', type=str, required=False,
                        help='Input segmentations of one modality (ITK Image)')
    parser.add_argument('-features_train', '--features_train', metavar='features_train',
                        nargs='+', dest='features_train', type=str, required=False,
                        help='Patient features input of first modality (HDF)')
    parser.add_argument('-pc_train', '--pc_train', metavar='Patientinfo',
                        dest='pc_train',
                        type=str, required=True, nargs='+',
                        help='Classification of patient')
    parser.add_argument('-conf_in', '--conf_in', metavar='conf_in', nargs='+',
                        dest='conf_in', type=str, required=True,
                        help='Configuration input')
    parser.add_argument('-conf_out', '--conf_out', metavar='conf_out', nargs='+',
                        dest='conf_out', type=str, required=True,
                        help='Configuration output')
    parser.add_argument('-conf_out_pyradiomics', '--conf_out_pyradiomics', metavar='conf_out_pyradiomics', nargs='+',
                        dest='conf_out_pyradiomics', type=str, required=False,
                        help='Configuration output for PyRadiomics')
    parser.add_argument('-type', '--type', metavar='type', nargs='+',
                        dest='type', type=str, required=False,
                        help='Type fo fingerprint to be extracted {string}')
    args = parser.parse_args()

    # Convert all arguments that are single elements
    if type(args.pc_train) is list:
        args.pc_train = ''.join(args.pc_train)

    if type(args.type) is list:
        args.type = ''.join(args.type)

    if type(args.conf_in) is list:
        args.conf_in = ''.join(args.conf_in)

    if type(args.conf_out) is list:
        args.conf_out = ''.join(args.conf_out)

    if type(args.conf_out_pyradiomics) is list:
        args.conf_out_pyradiomics = ''.join(args.conf_out_pyradiomics)

    # Configure Fingerprinter
    fp = Fingerprinter()
    fp.images = args.images_train
    fp.segmentations = args.segmentations_train
    fp.features = args.features_train
    fp.labels = args.pc_train
    fp.configuration = args.conf_in
    fp.type = args.type

    # Execute
    conf_out = fp.execute()

    # Save output
    with open(args.conf_out, 'w') as configfile:
        conf_out.write(configfile)

    # Save output also for PyRadiomics if required
    if args.conf_out_pyradiomics:
        config_pyradiomics = io.convert_config_pyradiomics(conf_out)
        with open(args.conf_out_pyradiomics, 'w') as file:
            yaml.safe_dump(config_pyradiomics, file)


if __name__ == '__main__':
    main()
