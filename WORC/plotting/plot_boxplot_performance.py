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

import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tikzplotlib


def generate_performance_boxplots(performances, metrics, outputfolder,
                                  colors=None):
    """Generate boxplots for performance of various models."""
    if colors is None:
        colors = list()
        for p in performances:
            colors.append(np.random.rand(3,))

    linecolor = [0.1, 0.1, 0.1]

    nperf = len(performances)
    f = plt.figure(figsize=(nperf*2.5, nperf))
    ax = plt.subplot(1,1,1)

    # Define the middle label position
    if np.mod(nperf, 2) == 0:
        # Even, pick a middle position
        middle = None
    else:
        middle = (nperf - 1) / 2.0

    xdist = 0
    markersize = 0.3
    xmaindelta = nperf / 2.5
    xdelta = 1.5 / nperf

    y_all = list()
    linepositions = list()
    labelpositions = list()
    labelnames = list()
    for m in metrics:
        xdist += xmaindelta
        if middle is None:
            labelpositions.append(xdist + nperf / 2.0 * xdelta - xdelta / 2.0)
            labelnames.append('\n' + m)

        for pnum, p in enumerate(performances):
            y = p[m]
            x = xdist + xdelta*pnum
            color = colors[pnum]
            ax.scatter([x] * len(y),
                       y,
                       color=color,
                       s=markersize)

            y_all.append(y)

            color = [color[0], color[1], color[2], 0.4]
            facecolor = [color[0], color[1], color[2], 0.2]
            plt.boxplot(y, positions=[x], widths=xdelta*0.75, patch_artist=True,
                        boxprops=dict(facecolor=facecolor, color=color),
                        capprops=dict(color=color),
                        whiskerprops=dict(color=color),
                        flierprops=dict(color=color, markeredgecolor=color),
                        medianprops=dict(color=color))

            labelpositions.append(x)

            if middle is not None:
                if pnum == middle:
                    labelnames.append(str(pnum + 1) + '\n' + m)
                    continue

            labelnames.append(str(pnum + 1))

        linepositions.append((x + xdist + xmaindelta)/2.0)

    # Draw vertical lines
    for x in linepositions[0:-1]:
        ax.axvline(x=x, linestyle='dashed', color=linecolor, linewidth=0.5)

    # Put in the labels / xticks
    plt.xticks(labelpositions, labelnames)

    # Changes all colors of the axes
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
        ax.spines[axis].set_color(linecolor)

    ax.tick_params(axis='x', colors=linecolor)
    ax.tick_params(axis='y', colors=linecolor)
    ax.yaxis.label.set_color(linecolor)
    ax.xaxis.label.set_color(linecolor)

    # High DTI to  make sure we save the maximized image
    fname = 'boxplot_test.png'
    outputname_png = os.path.join(outputfolder, fname)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    f.savefig(outputname_png, dpi=600, bbox_inches='tight',
              pad_inches=0)
    print(("Boxplot saved as {} !").format(outputname_png))

    fname = 'boxplot_test.tex'
    outputname_tex = os.path.join(outputfolder, fname)
    tikzplotlib.save(outputname_tex)


def test():
    """Test functionality with synthetic data."""
    perf1 = dict()
    perf1['AUC'] = np.random.randint(low=50, high=100, size=100) / 100.0
    perf1['Accuracy'] = np.random.randint(low=50, high=100, size=100) / 100.0
    perf1['Sensitivity'] = np.random.randint(low=70, high=90, size=100) / 100.0
    perf1['Specificity'] = np.random.randint(low=70, high=90, size=100) / 100.0

    perf2 = dict()
    perf2['AUC'] = np.random.randint(low=75, high=100, size=100) / 100.0
    perf2['Accuracy'] = np.random.randint(low=75, high=100, size=100) / 100.0
    perf2['Sensitivity'] = np.random.randint(low=80, high=100, size=100) / 100.0
    perf2['Specificity'] = np.random.randint(low=80, high=100, size=100) / 100.0

    perf3 = dict()
    perf3['AUC'] = np.random.randint(low=50, high=75, size=100) / 100.0
    perf3['Accuracy'] = np.random.randint(low=50, high=75, size=100) / 100.0
    perf3['Sensitivity'] = np.random.randint(low=60, high=80, size=100) / 100.0
    perf3['Specificity'] = np.random.randint(low=60, high=80, size=100) / 100.0

    perf4 = dict()
    perf4['AUC'] = np.random.randint(low=50, high=75, size=100) / 100.0
    perf4['Accuracy'] = np.random.randint(low=50, high=75, size=100) / 100.0
    perf4['Sensitivity'] = np.random.randint(low=60, high=80, size=100) / 100.0
    perf4['Specificity'] = np.random.randint(low=60, high=80, size=100) / 100.0

    perf5 = dict()
    perf5['AUC'] = np.random.randint(low=50, high=75, size=100) / 100.0
    perf5['Accuracy'] = np.random.randint(low=50, high=75, size=100) / 100.0
    perf5['Sensitivity'] = np.random.randint(low=60, high=80, size=100) / 100.0
    perf5['Specificity'] = np.random.randint(low=60, high=80, size=100) / 100.0

    performances = [perf1, perf2, perf3, perf4]
    metrics = list(perf1.keys())
    out = os.getcwd()
    colors = [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0, 0.5, 0.5], [0.5, 0, 0.5]]
    generate_performance_boxplots(performances, metrics, out, colors)


if __name__ == '__main__':
    test()
