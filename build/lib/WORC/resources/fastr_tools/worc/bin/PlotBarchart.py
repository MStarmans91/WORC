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

from WORC.plotting.plot_barchart import plot_barchart
import argparse


def main():
    parser = argparse.ArgumentParser(description='Plot the barchart of the hyperparameters of a number of hyperparameter settings')
    parser.add_argument('-prediction', '--prediction', metavar='prediction',
                        nargs='+', dest='prediction', type=str, required=True,
                        help='Prediction file (HDF)')
    parser.add_argument('-estimators', '--estimators', metavar='estimators',
                        nargs='+', dest='estimators', type=str, required=True,
                        help='Number of estimators (int)')
    parser.add_argument('-label_type', '--label_type', metavar='label_type',
                        nargs='+', dest='label_type', type=str, required=True,
                        help='Label name that is predicted (string)')
    parser.add_argument('-output_png', '--output_png', metavar='output_png',
                        nargs='+', dest='output_png', type=str, required=False,
                        help='File to write output to (PNG)')
    parser.add_argument('-output_tex', '--output_tex', metavar='output_tex',
                        nargs='+', dest='output_tex', type=str, required=False,
                        help='File to write output to (tex)')
    args = parser.parse_args()

    # Convert inputs from lists to elements for single inputs
    if type(args.prediction) is list:
        args.prediction = ''.join(args.prediction)

    if type(args.label_type) is list:
        args.label_type = ''.join(args.label_type)

    if type(args.output_png) is list:
        args.output_png = ''.join(args.output_png)

    if type(args.output_tex) is list:
        args.output_tex = ''.join(args.output_tex)

    # Number of estimators should be an integer
    if type(args.estimators) is list:
        args.estimators = int(args.estimators[0])

    # Plot the barchart
    plot_barchart(prediction=args.prediction,
                  estimators=args.estimators,
                  label_type=args.label_type,
                  output_tex=args.output_tex,
                  output_png=args.output_png)


if __name__ == '__main__':
    main()
