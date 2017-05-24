import pandas as pd
import json
import csv
import argparse
from natsort import natsorted
import glob
import os


def segstatistics():
    parser = argparse.ArgumentParser(description='Radiomics results')
    parser.add_argument('-seg1', '--seg1', metavar='seg1',
                        nargs='+', dest='seg1', type=str, required=True,
                        help='SVM file (HDF)')
    parser.add_argument('-seg2', '--seg2', metavar='seg2',
                        nargs='+', dest='seg2', type=str, required=True,
                        help='SVM file (HDF)')
    parser.add_argument('-seg3', '--seg3', metavar='seg3',
                        nargs='+', dest='seg3', type=str, required=True,
                        help='SVM file (HDF)')
    parser.add_argument('-seg4', '--seg4', metavar='seg4',
                        nargs='+', dest='seg1', type=str, required=True,
                        help='SVM file (HDF)')
    parser.add_argument('-res', '--res', metavar='Results', nargs='+',
                        dest='res', type=str, required=True,
                        help='Results')
    args = parser.parse_args()

    if type(args.svm) is list:
        args.svm = ''.join(args.svm)

    if type(args.ts) is list:
        args.ts = ''.join(args.ts)

    if type(args.res) is list:
        args.res = ''.join(args.res)

    # If input is dir, use glob
    if os.path.isdir(args.svm):
        args.svm = glob.glob(args.svm + '/svm_*.hdf5')
        args.svm = natsorted(args.svm)

    if os.path.isdir(args.ts):
        args.ts = glob.glob(args.ts + '/TS_*.csv')
        args.ts = natsorted(args.ts)

    if os.path.isdir(args.perf):
        args.perf = glob.glob(args.perf + '/performance_*.json')
        args.perf = natsorted(args.perf)

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

    # Join all labels and values
    # if type(patient_labels) == dict:
    #     patient_labels = patient_labels.keys()
    #
    # patient_labels_right = list()
    # patient_labels_wrong = list()
    # for k in patient_labels:
    #     patient_labels_right.append(k + ' (Right)')
    #     patient_labels_wrong.append(k + ' (Wrong)')

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
