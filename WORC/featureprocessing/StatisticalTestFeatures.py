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

import os
import csv
import numpy as np
from scipy.stats import ttest_ind, ranksums, mannwhitneyu, chi2_contingency
import WORC.IOparser.config_io_classifier as config_io
from WORC.IOparser.file_io import load_features
from WORC.detectors.detectors import DebugDetector


def StatisticalTestFeatures(features, patientinfo, config, output_csv=None,
                            output_png=None, plot_test='MWU', Bonferonni=True,
                            fontsize='small', yspacing=1,
                            threshold=0.05, verbose=True, label_type=None):
    """Perform several statistical tests on features, such as a student t-test.

    Parameters
    ----------
    features: string, mandatory
            contains the paths to all .hdf5 feature files used.
            modalityname1=file1,file2,file3,... modalityname2=file1,...
            Thus, modalities names are always between a space and a equal
            sign, files are split by commas. We assume that the lists of
            files for each modality has the same length. Files on the
            same position on each list should belong to the same patient.

    patientinfo: string, mandatory
            Contains the path referring to a .txt file containing the
            patient label(s) and value(s) to be used for learning. See
            the Github Wiki for the format.

    config: string, mandatory
            path referring to a .ini file containing the parameters
            used for feature extraction. See the Github Wiki for the possible
            fields and their description.

    # TODO: outputs

    verbose: boolean, default True
            print final feature values and labels to command line or not.

    """
    # Load variables from the config file
    config = config_io.load_config(config)

    if type(patientinfo) is list:
        patientinfo = ''.join(patientinfo)

    if type(config) is list:
        config = ''.join(config)

    if type(output_csv) is list:
        output_csv = ''.join(output_csv)

    if type(output_png) is list:
        output_png = ''.join(output_png)

    # Create output folder if required
    if not os.path.exists(os.path.dirname(output_csv)):
        os.makedirs(os.path.dirname(output_csv))

    if label_type is None:
        label_type = config['Labels']['label_names']

    # Read the features and classification data
    print("Reading features and label data.")
    label_data, image_features =\
        load_features(features, patientinfo, label_type)

    # Extract feature labels and put values in an array
    feature_labels = image_features[0][1]
    feature_values = np.zeros([len(image_features), len(feature_labels)])
    for num, x in enumerate(image_features):
        feature_values[num, :] = x[0]

    # -----------------------------------------------------------------------
    # Perform statistical tests
    print("Performing statistical tests.")
    label_value = label_data['label']
    label_name = label_data['label_name']

    header = list()
    subheader = list()
    for i_name in label_name:
        header.append(str(i_name[0]))
        header.append('')
        header.append('')
        header.append('')
        header.append('')
        header.append('')

        subheader.append('Label')
        subheader.append('Ttest')
        subheader.append('Welch')
        subheader.append('Wilcoxon')
        subheader.append('Mann-Whitney')
        subheader.append('Chi2')
        subheader.append('')

    # Open the output_csv file
    if output_csv is not None:
        myfile = open(output_csv, 'w')
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(header)
        wr.writerow(subheader)

    savedict = dict()
    for i_class, i_name in zip(label_value, label_name):
        savedict[i_name[0]] = dict()
        pvalues = list()
        pvalueswelch = list()
        pvalueswil = list()
        pvaluesmw = list()
        pvalueschi2 = list()
        classlabels = i_class.ravel()

        for num, fl in enumerate(feature_labels):
            fv = feature_values[:, num]

            # Remove NaN values
            fv = fv[~np.isnan(fv)]

            class1 = [i for j, i in enumerate(fv) if classlabels[j] == 1]
            class2 = [i for j, i in enumerate(fv) if classlabels[j] == 0]

            pvalues.append(ttest_ind(class1, class2)[1])
            pvalueswelch.append(ttest_ind(class1, class2, equal_var=False)[1])
            pvalueswil.append(ranksums(class1, class2)[1])
            try:
                pvaluesmw.append(mannwhitneyu(class1, class2)[1])
            except ValueError as e:
                print("[WORC Warning] " + str(e) + '. Replacing metric value by NaN.')
                pvaluesmw.append(np.nan)

            # Optional: perform chi2 test. Only do this when categorical, which we define as less than 20 options.
            unique_values = list(set(fv))
            unique_values.sort()
            if len(unique_values) == 1:
                print("[WORC Warning] " + fl + " has only one value. Replacing chi2 metric value by NaN.")
                pvalueschi2.append(np.nan)
            elif len(unique_values) <= 20:
                class1_count = [class1.count(i) for i in unique_values]
                class2_count = [class2.count(i) for i in unique_values]
                obs = np.array([class1_count, class2_count])

                _, p, _, _ = chi2_contingency(obs)
                pvalueschi2.append(p)
            else:
                print("[WORC Warning] " + fl + " is no categorical variable. Replacing chi2 metric value by NaN.")
                pvalueschi2.append(np.nan)

        # Sort based on p-values:
        indices = np.argsort(np.asarray(pvaluesmw))
        feature_labels_o = np.asarray(feature_labels)[indices].tolist()

        pvalues = np.asarray(pvalues)[indices].tolist()
        pvalueswelch = np.asarray(pvalueswelch)[indices].tolist()
        pvalueswil = np.asarray(pvalueswil)[indices].tolist()
        pvaluesmw = np.asarray(pvaluesmw)[indices].tolist()
        pvalueschi2 = np.asarray(pvalueschi2)[indices].tolist()

        savedict[i_name[0]]['ttest'] = pvalues
        savedict[i_name[0]]['welch'] = pvalueswelch
        savedict[i_name[0]]['wil'] = pvalueswil
        savedict[i_name[0]]['mw'] = pvaluesmw
        savedict[i_name[0]]['chi2'] = pvalueschi2
        savedict[i_name[0]]['labels'] = feature_labels_o

    if output_csv is not None:
        for num in range(0, len(savedict[i_name[0]]['ttest'])):
            writelist = list()
            for i_name in savedict.keys():
                labeldict = savedict[i_name]
                writelist.append(labeldict['labels'][num])
                writelist.append(labeldict['ttest'][num])
                writelist.append(labeldict['welch'][num])
                writelist.append(labeldict['wil'][num])
                writelist.append(labeldict['mw'][num])
                writelist.append(labeldict['chi2'][num])
                writelist.append('')

            wr.writerow(writelist)

        print("Saved data to CSV!")

    if output_png is not None:
        # Initialize objects
        objects_temp = labeldict['labels']
        if plot_test == 'MWU':
            p_values_temp = labeldict['mw']

        # remove the nan
        objects_nonan = list()
        p_values_nonan = list()
        for o, p in zip(objects_temp, p_values_temp):
            if not np.isnan(p):
                objects_nonan.append(o)
                p_values_nonan.append(p)

        # Debug defaults
        if DebugDetector().do_detection():
            # No correction
            Bonferonni = False

            # Just select ~10 features
            sorted_p = p_values_nonan[:]
            sorted_p.sort()
            threshold = sorted_p[10]

        if Bonferonni:
            p_values_nonan = [p * len(p_values_nonan) for p in p_values_nonan]

        # Threshold
        objects = list()
        p_values = list()
        for o, p in zip(objects_nonan, p_values_nonan):
            if p < threshold:
                objects.append(o)
                p_values.append(p)

        # Replace several labels
        objects = [o.replace('CalcFeatures_', '') for o in objects]
        objects = [o.replace('featureconverter_', '') for o in objects]
        objects = [o.replace('PREDICT_', '') for o in objects]
        objects = [o.replace('PyRadiomics_', '') for o in objects]
        objects = [o.replace('original_', '') for o in objects]
        objects = [o.replace('train_', '') for o in objects]
        objects = [o.replace('test_', '') for o in objects]
        objects = [o.replace('1_0_', '') for o in objects]

        # Create positions
        y_pos = np.arange(len(objects)) * yspacing
        y_pos = y_pos[::-1]

        # Create colors
        colors = list()
        for o in objects:
            if 'tf_' in o or 'vf_' in o or 'pf_' in o:
                colors.append('darkblue')
            elif 'hf_' in o:
                colors.append('magenta')
            elif 'sf_' in o:
                colors.append('cyan')
            elif 'of_' in o:
                colors.append('gray')
            elif 'semf_' in o:
                colors.append('black')
            else:
                raise KeyError(o)

        # Plotting
        # fig = plt.figure(figsize=(20, 5))
        plt.barh(y_pos, p_values, align='center', alpha=0.5, color=colors)
        plt.yticks(y_pos, objects, rotation=00, fontsize=fontsize)
        plt.xscale('log')
        plt.xticks(fontsize=fontsize)
        plt.xticks([1E-3, 1E-2, 1E-1], ['10-3', '10-2', '10-1'])

        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.30)

        plt.xlabel('Mann-Whitney-U Corrected P-value', fontsize=fontsize)

        # Adjust such that labels fit in figure
        plt.gcf().subplots_adjust(left=0.35)
        plt.gcf().subplots_adjust(right=0.98)
        plt.gcf().subplots_adjust(top=0.99)
        plt.gcf().subplots_adjust(bottom=0.1)

        # Make figure fullscreen and save
        fig = plt.gcf()
        fig.set_size_inches((15, 11), forward=False)
        fig.savefig(output_png, dpi=600)
        print("Saved data to PNG!")

    return savedict
