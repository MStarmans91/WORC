#!/usr/bin/env python

# Copyright 2016-2020 Biomedical Imaging Group Rotterdam, Departments of
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

import WORC.IOparser.config_io_classifier as config_io
from WORC.IOparser.file_io import load_features
import os
import numpy as np
import zipfile
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def plot_boxplot_features(features, label_data, config, output_zip,
                          label_type=None, verbose=False):
    # Load variables from the config file
    config = config_io.load_config(config)

    # Create output folder if required
    if not os.path.exists(os.path.dirname(output_zip)):
        os.makedirs(os.path.dirname(output_zip))

    if label_type is None:
        label_type = config['Labels']['label_names']

    # Read and stack the features
    if verbose:
        print("Reading features and label data.")
    label_data, image_features =\
        load_features(features, label_data, label_type)

    # Generate the actual boxplots
    generate_feature_boxplots(image_features, label_data, output_zip,
                              verbose=verbose)


def generate_feature_boxplots(image_features, label_data, output_zip, dpi=500,
                              verbose=False):
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
    labels = image_features[0][1]
    featvect = dict()
    flab = dict()
    for l in labels:
        featvect[l] = {"all": [], "1": [], "0": []}
        flab[l] = {"all": [], "1": [], "0": []}

    # Stack per feature type and class
    if verbose:
        print("Stacking features.")
    label = label_data['label'].tolist()[0]
    patient_IDs = label_data['patient_IDs'].tolist()
    for imfeat, label, pid in zip(image_features, label, patient_IDs):
        imfeat = imfeat[0]
        for flnum, fl in enumerate(labels):
            featvect[fl]['all'].append(imfeat[flnum])
            flab[fl]['all'].append(pid)
            if label[0] == 0:
                featvect[fl]['0'].append(imfeat[flnum])
                flab[fl]['0'].append(pid)
            else:
                featvect[fl]['1'].append(imfeat[flnum])
                flab[fl]['1'].append(pid)

    # Generate the output zip file
    zipf = zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
    outputfolder_temp = os.path.join(os.path.dirname(output_zip))

    # Create the boxplots
    if verbose:
        print("Generating boxplots.")

    # Split in 5x5 figures.
    nfig = np.ceil(len(labels) / 25.0)

    labels = sorted(labels)
    for fi in range(0, int(nfig)):
        f = plt.figure(figsize=(13, 10))
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

            # Alter the label, remove redundant parts
            lab = lab.replace('featureconverter_', '')
            lab = lab.replace('train_', '')
            lab = lab.replace('test_', '')
            lab = lab.replace('CalcFeatures', '')
            lab = lab.replace('predict', '')
            lab = lab.replace('pyradiomics', '')
            
            ax.set_title(lab, fontsize=fz)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fz)

            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fz)

            fignum += 1

        # Maximize figure to get correct spacings
        # mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())
        # plt.show()

        # High DTI to  make sure we save the maximized image
        fname = ('boxplot_{}.png').format(str(fi))
        outputname = os.path.join(outputfolder_temp, fname)
        f.savefig(outputname, bbox_inches='tight', pad_inches=0, dpi=dpi)
        if verbose:
            print(("Boxplot saved as {} !").format(outputname))

        # Copy the image to the zipfile and remove image
        zipf.write(outputname, os.path.basename(outputname))
        plt.close()
        os.remove(outputname)
