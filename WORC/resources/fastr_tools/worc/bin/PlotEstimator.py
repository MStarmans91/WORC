#!/usr/bin/env python

# Copyright 2017-2020 Biomedical Imaging Group Rotterdam, Departments of
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

from WORC.plotting.plot_estimator_performance import plot_estimator_performance
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description='Plot the performance of an estimator')
    parser.add_argument('-prediction', '--prediction', metavar='prediction',
                        nargs='+', dest='prediction', type=str, required=True,
                        help='Prediction file (HDF)')
    parser.add_argument('-pinfo', '--pinfo', metavar='pinfo',
                        nargs='+', dest='pinfo', type=str, required=True,
                        help='Patient Info File (txt)')
    parser.add_argument('-ensemble', '--ensemble', metavar='ensemble',
                        nargs='+', dest='ensemble', type=str, required=True,
                        help='Length of ensemble (int)')
    parser.add_argument('-label_type', '--label_type', metavar='label_type',
                        nargs='+', dest='label_type', type=str, required=True,
                        help='Label name that is predicted (string)')
    parser.add_argument('-output_json', '--output_json', metavar='output_json',
                        nargs='+', dest='output_json', type=str, required=False,
                        help='File to write output to (json)')
    args = parser.parse_args()

    # Convert inputs to strings
    if type(args.prediction) is list:
        args.prediction = ''.join(args.prediction)

    if type(args.pinfo) is list:
        args.pinfo = ''.join(args.pinfo)

    if type(args.ensemble) is list:
        args.ensemble = int(args.ensemble[0])
        # ensemble = ''.join(ensemble)

    if type(args.output_json) is list:
        args.output_json = ''.join(args.output_json)

    # if type(args.label_type) is list:
    #     args.label_type = ''.join(args.label_type)

    # Plot the statistics
    stats =\
        plot_estimator_performance(prediction=args.prediction,
                                   label_data=args.pinfo,
                                   ensemble=args.ensemble,
                                   label_type=args.label_type,
                                   output='stats')

    with open(args.output_json, 'w') as fp:
        json.dump(stats, fp, indent=4)


if __name__ == '__main__':
    main()
