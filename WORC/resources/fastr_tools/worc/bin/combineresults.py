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

import pandas as pd
import json
import csv
import argparse
from natsort import natsorted
import glob
import os


def main():
    parser = argparse.ArgumentParser(description='Radiomics results')
    parser.add_argument('-svm', '--svm', metavar='svm',
                        nargs='+', dest='svm', type=str, required=True,
                        help='SVM file (HDF)')
    parser.add_argument('-perf', '--perf', metavar='Performance', nargs='+',
                        dest='perf', type=str, required=True,
                        help='Performance (Parameters + Statistics)')
    parser.add_argument('-ts', '--ts', metavar='ts', nargs='+',
                        dest='ts', type=str, required=False,
                        help='Total Sensitivity from PCE Analysis (.csv)')
    parser.add_argument('-res', '--res', metavar='Results', nargs='+',
                        dest='res', type=str, required=True,
                        help='Results')
    args = parser.parse_args()

    if type(args.svm) is list:
        args.svm = ''.join(args.svm)

    if type(args.ts) is list:
        args.ts = ''.join(args.ts)

    if type(args.perf) is list:
        args.perf = ''.join(args.perf)

    # If input is dir, use glob
    if os.path.isdir(args.svm):
        args.svm = glob.glob(args.svm + '/svm_*.hdf5')
        args.svm = natsorted(args.svm)

    if os.path.isdir(args.perf):
        args.perf = glob.glob(args.perf + '/performance_*.json')
        args.perf = natsorted(args.perf)

    if args.ts is not None:
        if os.path.isdir(args.ts):
            args.ts = glob.glob(args.ts + '/TS_*.csv')
            args.ts = natsorted(args.ts)
    else:
        args.ts = [None] * len(args.svm)

    # Remove provenance
    temp = list()
    for pf in args.perf:
        if 'prov' not in pf:
            temp.append(pf)

    args.perf = temp

    # Read the inputs: labels of SVM, parameters, sensitivity
    feature_labels = list()
    # TODO: Fix "Too Few Features" SVMs
    all_feature_labels = list()
    pftemp = list()
    tstemp = list()
    for svmfile, tsfile, pffile in zip(args.svm, args.ts, args.perf):
        svm = pd.read_hdf(svmfile)
        if 'Too Few Features.' in svm.keys() or ' Too Few Features.' in svm.keys():
            print(("Too few features in {}.").format(svmfile))
        else:
            labels = svm[svm.keys()[0]].ix['feature_labels'] # .tolist()
            feature_labels.append(labels)
            all_feature_labels = all_feature_labels + list(set(labels) - set(all_feature_labels))
            pftemp.append(pffile)
            tstemp.append(tsfile)

    args.perf = pftemp
    args.ts = tstemp
    parameters = list()
    statistics = list()
    alwaysright = list()
    alwayswrong = list()
    ar_labels = list()
    aw_labels = list()
    for parafile in args.perf:
        with open(parafile) as fp:
            parameters_temp = json.load(fp)

        parameters.append(parameters_temp['Parameters'])
        statistics.append(parameters_temp['Statistics'])

        # Extract always right and wrong labels
        ar_temp = parameters_temp['Statistics']['Always right']
        aw_temp = parameters_temp['Statistics']['Always wrong']

        ar = dict()
        aw = dict()
        for k in ar_temp.keys():
            ar[k + ' (Right)'] = ar_temp[k]
            if (k + ' (Right)') not in ar_labels:
                ar_labels.append(k + ' (Right)')

        for k in aw_temp.keys():
            aw[k + ' (Wrong)'] = aw_temp[k]
            if (k + ' (Wrong)') not in aw_labels:
                aw_labels.append(k + ' (Wrong)')

        # Set patient ID labels
        alwaysright.append(ar)
        alwayswrong.append(aw)

    parameter_labels = parameters[0].keys()
    statistics_labels = statistics[0].keys()

    # Fill in the missing patient labels

    # Read in total sensitivity
    if args.ts[0] is not None:
        ts = list()
        for tsfile in args.ts:
            ts_temp = list()
            with open(tsfile, 'rb') as f:
                reader = csv.reader(f)
                for row in reader:
                    ts_temp.append(row[0])
            ts.append(ts_temp)

        # Combine the svm labels with the ts values in one dictionary
        sensitivity = list()
        for label, values in zip(feature_labels, ts):
            sensitivity_temp = dict()
            for key, item in zip(label, values):
                sensitivity_temp[key] = item
            sensitivity.append(sensitivity_temp)

    else:
        sensitivity = [None]

    # Join all labels and values
    # if type(patient_labels) == dict:
    #     patient_labels = patient_labels.keys()
    #
    # patient_labels_right = list()
    # patient_labels_wrong = list()
    # for k in patient_labels:
    #     patient_labels_right.append(k + ' (Right)')
    #     patient_labels_wrong.append(k + ' (Wrong)')

    if sensitivity[0] is not None:
        all_labels = parameter_labels + statistics_labels +\
            natsorted(ar_labels) + natsorted(aw_labels) +\
            natsorted(all_feature_labels)

        parameter_grid = list()
        for para, stat, sens, right, wrong in zip(parameters, statistics,
                                                  sensitivity, alwaysright,
                                                  alwayswrong):
            z = dict()
            for d in [para, stat, sens, right, wrong]:
                for k, v in d.iteritems():
                    z[k] = v

            # Fill in missing values in dict with blanks
            for key in all_labels:
                if key not in z.keys():
                    z[key] = ''

            parameter_grid.append(z)
    else:
        all_labels = parameter_labels + statistics_labels +\
            natsorted(ar_labels) + natsorted(aw_labels)

        parameter_grid = list()
        for para, stat, right, wrong in zip(parameters, statistics,
                                                  alwaysright,
                                                  alwayswrong):
            z = dict()
            for d in [para, stat, right, wrong]:
                for k, v in d.iteritems():
                    z[k] = v

            # Fill in missing values in dict with blanks
            for key in all_labels:
                if key not in z.keys():
                    z[key] = ''

            parameter_grid.append(z)

    # Writing to csv file
    if type(args.res) == list:
        args.res = ''.join(args.res)

    with open(args.res, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=all_labels)
        writer.writeheader()
        for parameter_point in parameter_grid:
            writer.writerow(parameter_point)

    print("Done writing results!")


if __name__ == '__main__':
    main()


# 'python' '/scratch/mstarmans/WORC/fastr_Tools/worc/bin/combineresults.py' '--svm' 'svm_13.hdf5' 'svm_1.hdf5' 'svm_12.hdf5' 'svm_11.hdf5' 'svm_18.hdf5' 'svm_15.hdf5' 'svm_10.hdf5' 'svm_14.hdf5' 'svm_0.hdf5' 'svm_19.hdf5' 'svm_17.hdf5' 'svm_5.hdf5' 'svm_2.hdf5' 'svm_4.hdf5' 'svm_7.hdf5' 'svm_6.hdf5' 'svm_3.hdf5' 'svm_9.hdf5' 'svm_8.hdf5' '--perf' 'performance_13.json' 'performance_1.json' 'performance_12.json' 'performance_11.json' 'performance_18.json' 'performance_15.json' 'performance_10.json' 'performance_14.json' 'performance_0.json' 'performance_19.json' 'performance_17.json' 'performance_5.json' 'performance_2.json' 'performance_4.json' 'performance_7.json' 'performance_6.json' 'performance_3.json' 'performance_9.json' 'performance_8.json' '--ts' 'TS_13.csv' 'TS_1.csv' 'TS_12.csv' 'TS_11.csv' 'TS_18.csv' 'TS_15.csv' 'TS_0.csv' 'TS_14.csv' 'TS_10.csv' 'TS_19.csv' 'TS_17.csv' 'TS_5.csv' 'TS_2.csv' 'TS_4.csv' 'TS_7.csv' 'TS_6.csv' 'TS_3.csv' 'TS_9.csv' 'TS_8.csv' '--res' 'results.csv'

# 'python' '/scratch/mstarmans/WORC/fastr_Tools/worc/bin/combineresults.py' '--svm' 'svm_13.hdf5' 'svm_1.hdf5' 'svm_12.hdf5' 'svm_11.hdf5' 'svm_15.hdf5' 'svm_10.hdf5' 'svm_14.hdf5' 'svm_0.hdf5' 'svm_19.hdf5' 'svm_17.hdf5' 'svm_5.hdf5' 'svm_2.hdf5' 'svm_4.hdf5' 'svm_7.hdf5' 'svm_6.hdf5' 'svm_3.hdf5' 'svm_9.hdf5' 'svm_8.hdf5' '--perf' 'performance_13.json' 'performance_1.json' 'performance_12.json' 'performance_11.json' 'performance_15.json' 'performance_10.json' 'performance_14.json' 'performance_0.json' 'performance_19.json' 'performance_17.json' 'performance_5.json' 'performance_2.json' 'performance_4.json' 'performance_7.json' 'performance_6.json' 'performance_3.json' 'performance_9.json' 'performance_8.json' '--ts' 'TS_13.csv' 'TS_1.csv' 'TS_12.csv' 'TS_11.csv' 'TS_15.csv' 'TS_0.csv' 'TS_14.csv' 'TS_10.csv' 'TS_19.csv' 'TS_17.csv' 'TS_5.csv' 'TS_2.csv' 'TS_4.csv' 'TS_7.csv' 'TS_6.csv' 'TS_3.csv' 'TS_9.csv' 'TS_8.csv' '--res' 'results.csv'

# 'python' '/home/mstarmans/WORC/fastr_Tools/worc/bin/combineresults.py' '--svm' '/archive/mstarmans/Output/CLM_0503_M1_GL' '--perf' '/archive/mstarmans/Output/CLM_0503_M1_GL' '--ts' '/archive/mstarmans/Output/CLM_0503_M1_GL' '--res' 'results_CLM_0503F1_GL.csv'
