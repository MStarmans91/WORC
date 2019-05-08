#!/usr/bin/env python

# Copyright 2016-2019 Biomedical Imaging Group Rotterdam, Departments of
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

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("[PREDICT Warning] Cannot use scatterplot function, as _tkinter is not installed")

import pandas as pd
import argparse
import genetics.genetic_processing as gp
import os
import glob
from natsort import natsorted


def main():
    parser = argparse.ArgumentParser(description='Radiomics results')
    parser.add_argument('-feat', '--feat', metavar='feat',
                        nargs='+', dest='feat', type=str, required=True,
                        help='List of patient feature files (HDF)')
    parser.add_argument('-class', '--class', metavar='class',
                        nargs='+', dest='classs', type=str, required=True,
                        help='Classification of patients (text)')
    parser.add_argument('-lab', '--lab', metavar='lab',
                        nargs='+', dest='lab', type=str, required=True,
                        help='Label of two features to plot')
    parser.add_argument('-out', '--out', metavar='out',
                        nargs='+', dest='out', type=str, required=True,
                        help='Output png file')
    args = parser.parse_args()

    if type(args.classs) is list:
        args.classs = ''.join(args.classs)

    if type(args.out) is list:
        args.out = ''.join(args.out)

    if type(args.feat) is list and len(args.feat) == 1:
        args.feat = ''.join(args.feat)

    if os.path.isdir(args.feat):
        args.feat = glob.glob(args.feat + '/features_*.hdf5')
        args.feat = natsorted(args.feat)

    # Read and stack the features
    image_features_temp = list()
    for i_feat in range(len(args.feat)):
        feat = dict()
        feat_temp = pd.read_hdf(args.feat[i_feat])
        feat_temp = feat_temp.image_features

        for feattype in feat_temp.keys():
            feat_type = feat_temp[feattype]
            for subtype in feat_type.keys():
                subfeat = feat_type[subtype]
                for k in subfeat.keys():
                    feat[k] = subfeat[k]

        image_features_temp.append(feat)

    # Get the mutation labels and patient IDs
    mutation_type = [['GP']]
    mutation_data, image_features = gp.findmutationdata(args.classs,
                                                        mutation_type,
                                                        args.feat,
                                                        image_features_temp)

    image_features = image_features.tolist()
    # Select the two relevant features
    feat1_c0 = list()
    feat2_c0 = list()
    feat1_c1 = list()
    feat2_c1 = list()
    mutation_label = mutation_data['mutation_label'].tolist()[0]
    patient_IDs = mutation_data['patient_IDs'].tolist()

    for imfeat, label, pid in zip(image_features, mutation_label, patient_IDs):
        print imfeat[args.lab[0]], pid
        if label[0] == 0:
            feat1_c0.append(imfeat[args.lab[0]])
            feat2_c0.append(imfeat[args.lab[1]])
        else:
            feat1_c1.append(imfeat[args.lab[0]])
            feat2_c1.append(imfeat[args.lab[1]])

    # Make a scatter plot
    f = plt.figure()
    subplot = f.add_subplot(111)
    subplot.plot(feat1_c0, feat2_c0, linestyle='', ms=12, marker='o', color='navy')
    subplot.plot(feat1_c1, feat2_c1, linestyle='', ms=12, marker='x', color='red')
    # NOTE: arbitrary limits!
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.xlabel(args.lab[0])
    plt.ylabel(args.lab[1])
    plt.title('Feature scatter plot')
    plt.legend()
    plt.show()

    f.savefig(args.out)
    print(("Snapshot saved as {} !").format(args.out))


if __name__ == '__main__':
    main()
