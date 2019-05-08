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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import pandas as pd
import argparse
import WORC.processing.label_processing as lp
import os
import glob
from natsort import natsorted
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Radiomics results')
    parser.add_argument('-feat', '--feat', metavar='feat',
                        nargs='+', dest='feat', type=str, required=True,
                        help='List of patient feature files (HDF)')
    parser.add_argument('-class', '--class', metavar='class',
                        nargs='+', dest='classs', type=str, required=True,
                        help='Classification of patients (text)')
    parser.add_argument('-label_type', '--label_type', metavar='label_type',
                        nargs='+', dest='label_type', type=str, required=True,
                        help='Name of the label that was predicted')
    parser.add_argument('-out', '--out', metavar='out',
                        nargs='+', dest='out', type=str, required=True,
                        help='Output png file')
    args = parser.parse_args()

    if type(args.classs) is list:
        args.classs = ''.join(args.classs)

    if type(args.label_type) is list:
        args.label_type = ''.join(args.label_type)

    if type(args.out) is list:
        args.out = ''.join(args.out)

    if type(args.feat) is list and len(args.feat) == 1:
        args.feat = ''.join(args.feat)

    if os.path.isdir(args.feat):
        args.feat = glob.glob(args.feat + '/features_*.hdf5')
        args.feat = natsorted(args.feat)

    # Read and stack the features
    print("Reading features.")
    image_features_temp = list()
    for i_feat in range(len(args.feat)):
        feat_temp = pd.read_hdf(args.feat[i_feat])
        feat_values = feat_temp.feature_values
        feat_labels = feat_temp.feature_labels

        feat = {k: v for k, v in zip(feat_labels, feat_values)}

        image_features_temp.append(feat)

    # Get the labels and patient IDs
    print("Reading class labels.")
    label_type = args.label_type
    label_data, image_features = lp.findlabeldata(args.classs,
                                                  label_type,
                                                  args.feat,
                                                  image_features_temp)

    generate_boxplots(image_features, label_data, args.out)


def generate_boxplots(image_features, label_data, outputfolder):
    '''
    Generate boxplots of the feature values among different objects.

    Parameters
    ----------
    features: list, mandatory
        List with a dictionary of the feature labels and values for each patient.

    label_data: pandas dataframe, mandatory
        Dataframe containing the labels of the objects.

    outputfolder: path, mandatory
        Folder to which the output boxplots should be written.


    '''
    labels = image_features[0].keys()
    featvect = dict()
    flab = dict()
    for l in labels:
        featvect[l] = {"all": [], "1": [], "0": []}
        flab[l] = {"all": [], "1": [], "0": []}

    # Stack per feature type and class
    print("Stacking features.")
    label = label_data['label'].tolist()[0]
    patient_IDs = label_data['patient_IDs'].tolist()
    for imfeat, label, pid in zip(image_features, label, patient_IDs):
        for fl in labels:
            featvect[fl]['all'].append(imfeat[fl])
            flab[fl]['all'].append(pid)
            if label[0] == 0:
                featvect[fl]['0'].append(imfeat[fl])
                flab[fl]['0'].append(pid)
            else:
                featvect[fl]['1'].append(imfeat[fl])
                flab[fl]['1'].append(pid)

    # Create the boxplots
    print("Generating boxplots.")

    # Split in 5x5 figures.
    nfig = np.ceil(len(labels) / 25.0)

    labels = sorted(labels)
    for fi in range(0, int(nfig)):
        f = plt.figure()
        fignum = 1
        for i in range(fi*25, min((fi+1)*25, len(labels))):
            ax = plt.subplot(5, 5, fignum)
            lab = labels[i]
            plt.subplots_adjust(hspace=0.3, wspace=0.2)
            ax.scatter(np.ones(len(featvect[lab]['all'])),
                       featvect[lab]['all'],
                       color='blue')
            ax.scatter(np.ones(len(featvect[lab]['1']))*2.0,
                       featvect[lab]['1'],
                       color='red')
            ax.scatter(np.ones(len(featvect[lab]['0']))*3.0,
                       featvect[lab]['0'],
                       color='green')

            plt.boxplot([featvect[lab]['all'], featvect[lab]['1'], featvect[lab]['0']])

            fz = 5  # Works best after saving
            ax.set_title(lab, fontsize=fz)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fz)

            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fz)

            fignum += 1

        # Maximize figure to get correct spacings
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        # plt.show()

        # High DTI to  make sure we save the maximized image
        fname = ('boxplot_{}.png').format(str(fi))
        outputname = os.path.join(outputfolder, fname)
        f.savefig(outputname, dpi=600)
        print(("Boxplot saved as {} !").format(outputname))


if __name__ == '__main__':
    main()
