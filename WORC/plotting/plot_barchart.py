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

import tikzplotlib
import numpy as np
import pandas as pd
from collections import Counter
import argparse


def plot_barchart(prediction, estimators=10, label_type=None, output_tex=None,
                  output_png=None):
    '''
    Make a barchart of the top X hyperparameters settings of the ranked
    estimators in all cross validation iterations.

    Parameters
    ----------
    prediction: filepath, mandatory
        Path pointing to the .hdf5 file which was is the output of the
        trainclassifier function.

    estimators: integer, default 10
        Number of hyperparameter settings/estimators used in each cross
        validation. The settings are ranked, so when supplying e.g. 10,
        the best 10 settings in each cross validation setting will be used.

    label_type: string, default None
        The name of the label predicted by the estimator. If None,
        the first label from the prediction file will be used.

    output_tex: filepath, optional
        If given, the barchart will be written to this tex file.

    output_png: filepath, optional
        If given, the barchart will be written to this png file.

    Returns
    ----------
    fig: matplotlib figure
        The figure in which the barchart is plotted.

    '''
    # Load input prediction
    prediction = pd.read_hdf(prediction)

    # Determine for which label we extract the estimator
    keys = prediction.keys()
    if label_type is None:
        label_type = keys[0]

    prediction = prediction[label_type]

    # Extract the parameter settings:
    parameters = dict()
    for n_crossval, est in enumerate(prediction.classifiers):
        for n_setting in range(0, estimators):
            # Extract parameter settings of nth estimator
            parameters_all = est.cv_results_['params'][n_setting]

            # Stack settings in parameters dictionary
            for k in parameters_all.keys():
                if k not in parameters.keys():
                    parameters[k] = list()
                parameters[k].append(parameters_all[k])

    # Count for every parameter how many times a setting occurs
    counts = count_parameters(parameters)

    # Normalize the values
    normalization_factor = len(prediction.classifiers) * estimators

    # Make the barplot
    fig = plot_bars(counts, normalization_factor)

    # Try making it fullscreen

    # Save the output
    if output_tex is not None:
        print(f'Saving barchart to {output_tex}.')
        tikzplotlib.save(output_tex)

    if output_png is not None:
        print(f'Saving barchart to {output_png}.')
        fig.savefig(output_png, bbox_inches='tight', pad_inches=0, dpi=500)


def plot_bars(params, normalization_factor=None, figwidth=40, fontsize=30,
              spacing=2):

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # Count how often feature groups are used
    ntimes_groups = list()
    groups = list()
    for key in params.keys():
        # Check if parameter is a boolean
        if 'True' in params[key].keys() or 'False' in params[key].keys():
            if 'True' in params[key].keys():
                ntimes_groups.append(params[key]['True'])
                groups.append(key)
            else:
                # Only False
                ntimes_groups.append(0)
                groups.append(key)

    # Normalize the values in order to not make figure to large
    if normalization_factor is None:
        normalization_factor = max(ntimes_groups)
    normalization_factor = float(normalization_factor)  # Needed for percentages
    ntimes_groups = [x / normalization_factor for x in ntimes_groups]

    # Create the figure for the barchart
    plt.rcdefaults()
    fig, ax = plt.subplots()
    fig.set_figwidth(figwidth)
    fig.set_figheight(figwidth)
    ax.set_xlim(0, 1)

    # Determine positions of all the labels
    y_pos = np.arange(len(groups) * spacing)
    ntimes_groups_plot = list()
    groups_plot = list()
    num = 0
    for i in range(len(groups) * spacing):
        if i % spacing == 0:
            ntimes_groups_plot.append(ntimes_groups[num])
            groups_plot.append(groups[num])
            num += 1
        else:
            # empty entry to fill up spacing
            ntimes_groups_plot.append(0.0)
            groups_plot.append('')

    # Normal features
    colors = ['steelblue', 'lightskyblue']
    ax.barh(y_pos, ntimes_groups_plot, align='center',
            color=colors[0], ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(groups_plot)

    ax.tick_params(axis='both', labelsize=fontsize)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Percentage', fontsize=fontsize)

    return fig


def count_parameters(parameters):
    # Count for every parameter how many times a setting occurs
    output = dict()
    for setting, values in parameters.items():
        output[setting] = dict()
        try:
            c = Counter(values)
            for k, v in zip(c.keys(), c.values()):
                output[setting][k] = v
        except TypeError:
            # Not possible to count parameters, remove
            del output[setting]

    return output


def paracheck(parameters):
    # NOTE: Deprecated
    output = dict()
    # print parameters

    f = parameters['semantic_features']
    total = float(len(f))
    count_semantic = sum([i == 'True' for i in f])
    ratio_semantic = count_semantic/total
    print("Semantic: " + str(ratio_semantic))
    output['semantic_features'] = ratio_semantic

    f = parameters['patient_features']
    count_patient = sum([i == 'True' for i in f])
    ratio_patient = count_patient/total
    print("patient: " + str(ratio_patient))
    output['patient_features'] = ratio_patient

    f = parameters['orientation_features']
    count_orientation = sum([i == 'True' for i in f])
    ratio_orientation = count_orientation/total
    print("orientation: " + str(ratio_orientation))
    output['orientation_features'] = ratio_orientation

    f = parameters['histogram_features']
    count_histogram = sum([i == 'True' for i in f])
    ratio_histogram = count_histogram/total
    print("histogram: " + str(ratio_histogram))
    output['histogram_features'] = ratio_histogram

    f = parameters['shape_features']
    count_shape = sum([i == 'True' for i in f])
    ratio_shape = count_shape/total
    print("shape: " + str(ratio_shape))
    output['shape_features'] = ratio_shape

    if 'coliage_features' in parameters.keys():
        f = parameters['coliage_features']
        count_coliage = sum([i == 'True' for i in f])
        ratio_coliage = count_coliage/total
        print("coliage: " + str(ratio_coliage))
        output['coliage_features'] = ratio_coliage

    if 'phase_features' in parameters.keys():
        f = parameters['phase_features']
        count_phase = sum([i == 'True' for i in f])
        ratio_phase = count_phase/total
        print("phase: " + str(ratio_phase))
        output['phase_features'] = ratio_phase

    if 'vessel_features' in parameters.keys():
        f = parameters['vessel_features']
        count_vessel = sum([i == 'True' for i in f])
        ratio_vessel = count_vessel/total
        print("vessel: " + str(ratio_vessel))
        output['vessel_features'] = ratio_vessel

    if 'log_features' in parameters.keys():
        f = parameters['log_features']
        count_log = sum([i == 'True' for i in f])
        ratio_log = count_log/total
        print("log: " + str(ratio_log))
        output['log_features'] = ratio_log

    f = parameters['texture_features']
    count_texture_all = sum([i == 'True' for i in f])
    ratio_texture_all = count_texture_all/total
    print("texture_all: " + str(ratio_texture_all))
    output['texture_all_features'] = ratio_texture_all

    count_texture_no = sum([i == 'False' for i in f])
    ratio_texture_no = count_texture_no/total
    print("texture_no: " + str(ratio_texture_no))
    output['texture_no_features'] = ratio_texture_no

    count_texture_Gabor = sum([i == 'Gabor' for i in f])
    ratio_texture_Gabor = count_texture_Gabor/total
    print("texture_Gabor: " + str(ratio_texture_Gabor))
    output['texture_Gabor_features'] = ratio_texture_Gabor

    count_texture_LBP = sum([i == 'LBP' for i in f])
    ratio_texture_LBP = count_texture_LBP/total
    print("texture_LBP: " + str(ratio_texture_LBP))
    output['texture_LBP_features'] = ratio_texture_LBP

    count_texture_GLCM = sum([i == 'GLCM' for i in f])
    ratio_texture_GLCM = count_texture_GLCM/total
    print("texture_GLCM: " + str(ratio_texture_GLCM))
    output['texture_GLCM_features'] = ratio_texture_GLCM

    count_texture_GLRLM = sum([i == 'GLRLM' for i in f])
    ratio_texture_GLRLM = count_texture_GLRLM/total
    print("texture_GLRLM: " + str(ratio_texture_GLRLM))
    output['texture_GLRLM_features'] = ratio_texture_GLRLM

    count_texture_GLSZM = sum([i == 'GLSZM' for i in f])
    ratio_texture_GLSZM = count_texture_GLSZM/total
    print("texture_GLSZM: " + str(ratio_texture_GLSZM))
    output['texture_GLSZM_features'] = ratio_texture_GLSZM

    count_texture_NGTDM = sum([i == 'NGTDM' for i in f])
    ratio_texture_NGTDM = count_texture_NGTDM/total
    print("texture_NGTDM: " + str(ratio_texture_NGTDM))
    output['texture_NGTDM_features'] = ratio_texture_NGTDM

    if 'degree' in parameters.keys():
        f = parameters['degree']
        print("Polynomial Degree: " + str(np.mean(f)))
        output['polynomial_degree'] = np.mean(f)

    return output


def main():
    parser = argparse.ArgumentParser(description='Plot a Barchart.')
    parser.add_argument('-prediction', '--prediction', metavar='prediction',
                        nargs='+', dest='prediction', type=str, required=True,
                        help='Prediction file (HDF)')
    parser.add_argument('-estimators', '--estimators', metavar='estimator',
                        nargs='+', dest='estimators', type=str, required=False,
                        help='Number of estimators to evaluate in each cross validation.')
    parser.add_argument('-label_type', '--label_type', metavar='label_type',
                        nargs='+', dest='label_type', type=str, required=False,
                        help='Key of the label which was predicted.')
    parser.add_argument('-output_tex', '--output_tex', metavar='output_tex',
                        nargs='+', dest='output_tex', type=str, required=True,
                        help='Output file path (.tex)')
    parser.add_argument('-output_png', '--output_png', metavar='output_png',
                        nargs='+', dest='output_png', type=str, required=True,
                        help='Output file path (.png)')
    args = parser.parse_args()

    # Convert the inputs to the correct format
    if type(args.prediction) is list:
        args.prediction = ''.join(args.prediction)

    if type(args.output) is list:
        args.output = ''.join(args.output)

    if type(args.estimators) is list:
        args.estimators = int(args.estimators[0])

    if type(args.label_type) is list:
        args.label_type = ''.join(args.label_type)

    plot_barchart(prediction=args.prediction,
                  estimators=args.estimators,
                  label_type=args.label_type,
                  output_tex=args.output_tex,
                  output_png=args.output_png)


if __name__ == '__main__':
    main()
