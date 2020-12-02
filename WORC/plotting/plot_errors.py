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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
import numpy as np
from WORC.IOparser.file_io import load_features
import pandas as pd
import WORC.addexceptions as ae
import tikzplotlib


def plot_errors(featurefiles, patientinfo, label_type, featurenames,
                posteriors_csv=None, agesex=True, output_png=None,
                output_tex=None):
    """Scatterplot of all objects with marking of errors."""
    # check some input
    if len(featurenames) != 2:
        raise ae.WORCValueError(f'Featurenames should be list with two strings, got {featurenames}.')

    # Read the features and classification data
    print("Reading features and label data.")
    label_data, image_features =\
        load_features(featurefiles, patientinfo, label_type)

    # Read in the scores and labels
    if posteriors_csv is not None:
        data = pd.read_csv(posteriors_csv)
        PIDs = data['PatientID'].values
        labels = data['TrueLabel'].values.tolist()
        probabilities = data['Probability'].values

    # Convert probabilities to correct / not
    correct = list()
    for label, prob in zip(labels, probabilities):
        if prob >= 0.5 and label == 1.0:
            correct.append(True)
        elif prob < 0.5 and label == 0.0:
            correct.append(True)
        else:
            # incorrect prediction
            correct.append(False)

    # Select indices of features we need
    feature_labels = image_features[0][1]
    if featurenames[0] not in feature_labels:
        raise ae.WORCKeyError(f'Feature {featurenames[0]} not in feature names.')
    else:
        feature_1_index = feature_labels.index(featurenames[0])

    if featurenames[1] not in feature_labels:
        raise ae.WORCKeyError(f'Feature {featurenames[1]} not in feature names.')
    else:
        feature_2_index = feature_labels.index(featurenames[1])

    # Match probabilities to features
    feature_1 = list()
    feature_2 = list()
    for pid, prob in zip(PIDs, probabilities):
        if pid not in label_data['patient_IDs']:
            raise ae.WORCKeyError(f'Features for {pid} not provided.')
        else:
            index = label_data['patient_IDs'].tolist().index(pid)
            feature_1.append(image_features[index][0][feature_1_index])
            feature_2.append(image_features[index][0][feature_2_index])

    # Resort based on PID
    order = np.argsort(PIDs)
    feature_1 = [feature_1[index] for index in order]
    feature_2 = [feature_2[index] for index in order]
    correct = [correct[index] for index in order]
    labels = [labels[index] for index in order]

    # Actual plotting
    f = plt.figure(figsize=(20, 15))
    ax = plt.subplot(1, 1, 1)
    coordinates = list()
    for index, label in enumerate(labels):
        # Check if coordinate has already been plotted
        coordinate = (feature_1[index], feature_2[index])
        addoffset = 1
        sign = 1
        while coordinate in coordinates:
            # Coordinate plotted, add an x-offset
            offset = sign * 0.01 * addoffset
            coordinate = (feature_1[index] + offset, feature_2[index])
            if sign == 1:
                sign = -1
            else:
                sign = 1
                addoffset += 1
        coordinates.append(coordinate)

        # Red border if classification is incorrect
        if not correct[index]:
            ax.scatter(coordinate[0], coordinate[1], s=80, marker='o', color='red')
            s = 30
        else:
            s = 50

        # Plot point in feature space
        if label == 1.0:
            ax.scatter(coordinate[0], coordinate[1], s=s, marker='o', color='#7dcfe2')
        else:
            ax.scatter(coordinate[0], coordinate[1], s=s, marker='o', color='blue')

    # Add some labelling etc to the plot
    if agesex:
        plt.xlabel('Sex', size=12)
        plt.ylabel('Age', size=12)
        plt.xticks([0, 1], ['Female', 'Male'], size=8)
    else:
        plt.xlabel(featurenames[0], size=12)
        plt.ylabel(featurenames[1], size=12)

    # Save output
    if output_png is not None:
        plt.savefig(output_png, bbox_inches='tight', pad_inches=0)
        print(f"Plot saved as {output_png}!")

    if output_tex is not None:
        tikzplotlib.save(output_tex)
        print(f"Plot saved as {output_tex}!")
