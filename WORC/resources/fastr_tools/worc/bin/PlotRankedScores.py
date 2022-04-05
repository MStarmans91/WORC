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

from WORC.plotting.plot_ranked_scores import plot_ranked_scores
import argparse


def main():
    parser = argparse.ArgumentParser(description='Plot the ranked scores of an estimator')
    parser.add_argument('-estimator', '--estimator', metavar='estimator',
                        nargs='+', dest='estimator', type=str, required=True,
                        help='Estimator file (HDF)')
    parser.add_argument('-pinfo', '--pinfo', metavar='pinfo',
                        nargs='+', dest='pinfo', type=str, required=True,
                        help='Patient Info File (txt)')
    parser.add_argument('-ensemble_method', '--ensemble_method', metavar='ensemble_method',
                        nargs='+', dest='ensemble_method', type=str, required=True,
                        help='Method for creating ensemble (string)')
    parser.add_argument('-ensemble_size', '--ensemble_size', metavar='ensemble_size',
                        nargs='+', dest='ensemble_size', type=str, required=False,
                        help='Length of ensemble (int)')
    parser.add_argument('-scores', '--scores', metavar='scores',
                        nargs='+', dest='scores', type=str, required=False,
                        help='Type of scoring used: percentages or posteriors (string)')
    parser.add_argument('-label_type', '--label_type', metavar='label_type',
                        nargs='+', dest='label_type', type=str, required=False,
                        help='Label name that is predicted (string)')
    parser.add_argument('-images', '--images', metavar='images',
                        nargs='+', dest='images', type=str, required=False,
                        help='Paths to images (ITK Image files)')
    parser.add_argument('-segmentations', '--segmentations',
                        metavar='segmentations', nargs='+',
                        dest='segmentations', type=str, required=False,
                        help='Paths to segmentations (ITK Image files)')
    parser.add_argument('-output_csv', '--output_csv', metavar='output_csv',
                        nargs='+', dest='output_csv', type=str, required=False,
                        help='File to write output to (csv)')
    parser.add_argument('-output_zip', '--output_zip', metavar='output_zip',
                        nargs='+', dest='output_zip', type=str, required=False,
                        help='File to write output to (zip)')
    args = parser.parse_args()

    # convert inputs that should be single arguments to lists
    pinfo = args.pinfo
    if type(pinfo) is list:
        pinfo = ''.join(pinfo)

    estimator = args.estimator
    if type(estimator) is list:
        estimator = ''.join(estimator)

    ensemble_method = args.ensemble_method
    if type(ensemble_method) is list:
        ensemble_method = ''.join(ensemble_method)

    ensemble_size = args.ensemble_size
    if type(ensemble_size) is list:
        ensemble_size = int(ensemble_size[0])

    label_type = args.label_type
    if type(label_type) is list:
        label_type = ''.join(label_type)

    scores = args.scores
    if type(scores) is list:
        scores = ''.join(scores)

    output_csv = args.output_csv
    if type(output_csv) is list:
        output_csv = ''.join(output_csv)

    output_zip = args.output_zip
    if type(output_zip) is list:
        output_zip = ''.join(output_zip)

    plot_ranked_scores(estimator=estimator,
                       pinfo=pinfo,
                       label_type=label_type,
                       scores=scores,
                       images=args.images,
                       segmentations=args.segmentations,
                       ensemble_method=ensemble_method,
                       ensemble_size=ensemble_size,
                       output_csv=output_csv,
                       output_zip=output_zip)


if __name__ == '__main__':
    main()
