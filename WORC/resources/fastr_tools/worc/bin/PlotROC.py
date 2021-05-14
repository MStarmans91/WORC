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
    parser.add_argument('-ROC_png', '--ROC_png', metavar='ROC_png',
                        nargs='+', dest='ROC_png', type=str, required=False,
                        help='File to write ROC to (PNG)')
    parser.add_argument('-ROC_csv', '--ROC_csv', metavar='ROC_csv',
                        nargs='+', dest='ROC_csv', type=str, required=False,
                        help='File to write ROC to (csv)')
    parser.add_argument('-ROC_tex', '--ROC_tex', metavar='ROC_tex',
                        nargs='+', dest='ROC_tex', type=str, required=False,
                        help='File to write ROC to (tex)')
    parser.add_argument('-PRC_png', '--PRC_png', metavar='PRC_png',
                        nargs='+', dest='PRC_png', type=str, required=False,
                        help='File to write PR to (PNG)')
    parser.add_argument('-PRC_csv', '--PRC_csv', metavar='PRC_csv',
                        nargs='+', dest='PRC_csv', type=str, required=False,
                        help='File to write PR to (csv)')
    parser.add_argument('-PRC_tex', '--PRC_tex', metavar='PRC_tex',
                        nargs='+', dest='PRC_tex', type=str, required=False,
                        help='File to write PR to (tex)')
    args = parser.parse_args()

    plot_ROC(prediction=args.prediction,
             pinfo=args.pinfo,
             ensemble=args.ensemble,
             label_type=args.label_type,
             ROC_png=args.ROC_png,
             ROC_tex=args.ROC_tex,
             ROC_csv=args.ROC_csv,
             PRC_png=args.PRC_png,
             PRC_tex=args.PRC_tex,
             PRC_csv=args.PRC_csv)


if __name__ == '__main__':
    main()
