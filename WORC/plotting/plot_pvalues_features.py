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

import numpy as np
import tikzplotlib


def manhattan_importance(values, labels, feature_labels,
                         output_png=None, output_tex=None,
                         mapping=None, threshold_annotated=0.05):
    # Assume labels are numeric and sorted
    f = plt.figure(figsize=(20, 10))

    # Generate X-positions
    positions = np.arange(len(values))

    # Initialize several objects
    values = np.asarray(values)
    unique_labels = set(list(labels))
    n_labels = len(unique_labels)
    colormap = ['#7dcfe2', '#4b78b5', 'darkgrey', 'dimgray'] * n_labels

    for lnum, i in enumerate(unique_labels):
        # Shift positions for each class
        for pnum in range(len(positions)):
            if labels[pnum] == i:
                positions[pnum] += lnum

        # NOTE: use lnum to leave space between labels for vlines
        # Only take first set of points corresponding to first label
        plot_positions = [p for p, l in zip(positions, labels) if l == i]
        plot_values = [v for v, l in zip(values, labels) if l == i]
        plt.scatter(plot_positions, plot_values, c=colormap[lnum])

    # Set a line after each label ends
    label_previous = labels[0]
    pos_previous = positions[0]
    color_end = list()
    vlines = list()
    # NOTE: leave space between groups to plot vline
    for i, p in zip(labels, positions):
        if i != label_previous:
            # New color starts here
            color_end.append((pos_previous + p) / 2.0)
            label_previous = i
            pos_previous = p

            # Plot vlines just between classes
            vlines.append(p - 1)

    # Add the last color and line
    color_end.append((pos_previous + p) / 2.0)

    # Decide y-limits
    ymax = np.max(values)
    for i in range(0, 100):
        if 10**(-i) < ymax:
            ymaxlim = i - 1
            break

    ymin = np.min(values)
    for i in range(0, 100):
        if 10**(-i) < ymin:
            yminlim = i
            break

    # Set several figure lay-out options
    plt.gca().invert_yaxis()
    plt.yscale('log')
    plt.ylim((10**-ymaxlim, 10**-yminlim))
    plt.xlim((0, max(positions)))

    plt.yticks([10**-i for i in range(ymaxlim, yminlim + 1)],
               [f'10-{i}' for i in range(ymaxlim, yminlim + 1)])
    if mapping is None:
        # Use raw labels (=numbers) as ticks
        plt.xticks(color_end, np.arange(len(color_end)) + 1, size=16)
    else:
        xticks = [mapping[i] for i in unique_labels]
        plt.xticks(color_end, xticks, size=8)

    plt.vlines(vlines, 10**-ymaxlim, 10**-yminlim,  linestyles='dotted', linewidth=0.3)

    if threshold_annotated > 10**-yminlim:
        y_value_annotated = threshold_annotated
        plt.hlines(threshold_annotated, 0, max(positions),  linestyles='dashed', linewidth=1, color='magenta')
    else:
        y_value_annotated = 10**-yminlim
        plt.hlines(10**-yminlim, 0, max(positions),  linestyles='dashed', linewidth=1, color='magenta')

    plt.annotate(f'p={round(threshold_annotated, 5)}',
                 (1, y_value_annotated),
                 xytext=(1, y_value_annotated*0.95), size=8, color='magenta')

    plt.xlabel("Feature groups", size=12)
    plt.ylabel("P-value Mann-Whitney U", size=12)

    # Annotate points above the threshold
    offset = len(values) / 200
    offset = np.clip(offset, 0.1, 100)

    annotated_values = [v for v in values if v < threshold_annotated]
    annotated_pos = [p for p, v in zip(positions, values) if v < threshold_annotated]
    annotated_labels = [p for p, v in zip(feature_labels, values) if v < threshold_annotated]

    y_offset = -0.1
    for x, y, text in zip(annotated_pos, annotated_values, annotated_labels):
        plt.annotate(text,
                     (x, y),
                     xytext=(x + offset, y * (1 - y_offset)), size=6)
        y_offset = y_offset * -1

    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')

    if output_png is not None:
        plt.savefig(output_png, bbox_inches='tight', pad_inches=0)
        print(f"Plot saved as {output_png}!")

    if output_tex is not None:
        tikzplotlib.save(output_tex)
        print(f"Plot saved as {output_tex}!")

    return f
