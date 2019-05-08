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

from WORC.plotting.plot_ROC import plot_ROC
import argparse


def main():
    parser = argparse.ArgumentParser(description='Plot the ROC Curve of an estimator')
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
    parser.add_argument('-output_png', '--output_png', metavar='output_png',
                        nargs='+', dest='output_png', type=str, required=False,
                        help='File to write output to (PNG)')
    parser.add_argument('-output_csv', '--output_csv', metavar='output_csv',
                        nargs='+', dest='output_csv', type=str, required=False,
                        help='File to write output to (csv)')
    parser.add_argument('-output_tex', '--output_tex', metavar='output_tex',
                        nargs='+', dest='output_tex', type=str, required=False,
                        help='File to write output to (tex)')
    args = parser.parse_args()

    plot_ROC(prediction=args.prediction,
             pinfo=args.pinfo,
             ensemble=args.ensemble,
             label_type=args.label_type,
             output_png=args.output_png,
             output_tex=args.output_tex,
             output_csv=args.output_csv)


if __name__ == '__main__':
    main()
