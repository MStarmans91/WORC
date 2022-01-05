#!/usr/bin/env python

# Copyright 2016-2021 Biomedical Imaging Group Rotterdam, Departments of
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

import argparse
import pandas as pd
from WORC.plotting.plot_estimator_performance import plot_estimator_performance
import WORC.processing.label_processing as lp
import glob
import numpy as np
import csv
import os
from WORC.plotting import plot_images as pi
import SimpleITK as sitk
from WORC.addexceptions import WORCKeyError
import zipfile
from sys import platform


# NOTE: Need to add thresholds of plot_ranked_images to arguments


def main():
    parser = argparse.ArgumentParser(description='Plot distances to hyperplane')
    parser.add_argument('-estimator', '--estimator', metavar='est',
                        nargs='+', dest='est', type=str, required=True,
                        help='Path to HDF5 file containing a fitted estimator.')
    parser.add_argument('-pinfo', '--pinfo', metavar='pinfo',
                        nargs='+', dest='pinfo', type=str, required=True,
                        help='Patient Info File (txt)')
    parser.add_argument('-images', '--images', metavar='images',
                        nargs='+', dest='ims', type=str, required=True,
                        help='Images of patients (ITK Image files)')
    parser.add_argument('-segmentations', '--segmentations', metavar='segmentations',
                        nargs='+', dest='segs', type=str, required=True,
                        help='Segmentations of patients (ITK Image files)')
    parser.add_argument('-ensemble_method', '--ensemble_method', metavar='ensemble_method',
                        nargs='+', dest='ens_method', type=str, required=True,
                        help='Method to be used for ensembling (string)')
    parser.add_argument('-ensemble_size', '--ensemble_size', metavar='ensemble_size',
                        nargs='+', dest='ens_size', type=str, required=True,
                        help='If ensembling method is top_N, size to be used (int)')
    parser.add_argument('-label_type', '--label_type', metavar='label_type',
                        nargs='+', dest='label_type', type=str, required=True,
                        help='Label name that is predicted by estimator (string)')
    parser.add_argument('-scores', '--scores', metavar='scores',
                        nargs='+', dest='scores', type=str, required=True,
                        help='Type of scoring: percentages or posteriors (string)')
    parser.add_argument('-output_csv', '--output_csv', metavar='output_csv',
                        nargs='+', dest='output_csv', type=str, required=True,
                        help='Output file for scores (CSV).')
    parser.add_argument('-output_zip', '--output_zip', metavar='output_zip',
                        nargs='+', dest='output_zip', type=str, required=True,
                        help='Output file for images (zip).')
    args = parser.parse_args()

    # convert inputs that should be single arguments to lists
    pinfo = args.pinfo
    if type(pinfo) is list:
        pinfo = ''.join(pinfo)

    estimator = args.est
    if type(estimator) is list:
        estimator = ''.join(estimator)

    ensemble_method = args.ens_method
    if type(ensemble_method) is list:
        ensemble_method = ''.join(ensemble_method)

    ensemble_size = args.ens_size
    if type(ensemble_size) is list:
        ensemble_size = int(ensemble_size[0])

    label_type = args.label_type
    if type(label_type) is list:
        label_type = ''.join(label_type)

    scores = args.scores
    if type(scores) is list:
        scores = ''.join(scores)

    output_csv = args.output_csv
    if type(output_csv) is list:
        output_csv = ''.join(output_csv)

    output_zip = args.output_zip
    if type(output_zip) is list:
        output_zip = ''.join(output_zip)

    plot_ranked_scores(estimator=estimator,
                       pinfo=pinfo,
                       label_type=label_type,
                       scores=scores,
                       images=args.ims,
                       segmentations=args.segs,
                       ensemble_method=ensemble_method,
                       ensemble_size=ensemble_size,
                       output_csv=output_csv,
                       output_zip=output_zip)


def flatten_object(input):
    """Flatten various objects to a 1D list."""
    out = []
    for x in input:
        if isinstance(x, (list, pd.core.series.Series, np.ndarray)):
            out += flatten_object(x)
        else:
            out.append(x)

    return out


def plot_ranked_percentages(estimator, pinfo, label_type=None,
                            ensemble_method='top_N', ensemble_size=100,
                            output_csv=None):

    # Read the inputs
    prediction = pd.read_hdf(estimator)

    # Determine the predicted score per patient
    print('Determining score per patient.')
    stats =\
        plot_estimator_performance(prediction,
                                   pinfo,
                                   [label_type],
                                   alpha=0.95,
                                   ensemble_method=ensemble_method,
                                   ensemble_size=ensemble_size,
                                   output='stats')

    percentages = stats['Rankings']['Percentages']
    ranking = np.argsort(list(percentages.values()))
    ranked_percentages_temp = [list(percentages.values())[r] for r in ranking]
    ranked_PIDs = [list(percentages.keys())[r] for r in ranking]

    ranked_percentages = list()
    ranked_truths = list()
    for r in ranked_percentages_temp:
        id = r.index(':')
        ranked_truths.append(float(r[0:id]))
        ranked_percentages.append(float(r[id+2:-1]))

    # Write output to csv
    if output_csv is not None:
        print("Writing output scores to CSV.")
        header = ['PatientID', 'TrueLabel', 'Percentage']
        with open(output_csv, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)

            for pid, truth, perc in zip(ranked_PIDs, ranked_truths, ranked_percentages):
                towrite = [str(pid), str(truth), str(perc)]
                writer.writerow(towrite)

    return ranked_percentages, ranked_truths, ranked_PIDs


def plot_ranked_images(pinfo, label_type, images, segmentations, ranked_truths,
                       ranked_scores, ranked_PIDs, output_zip=None,
                       output_itk=None, zoomfactor=4, scores='percentages'):
    # Match the images to the label data
    print('Matching image and segmentation data to labels.')
    label_data, images =\
        lp.findlabeldata(pinfo,
                         [label_type],
                         images,
                         objects=images)
    _, segmentations =\
        lp.findlabeldata(pinfo,
                         [label_type],
                         segmentations,
                         objects=segmentations)

    PIDs_images = label_data['patient_IDs'].tolist()
    PIDs_images = [i.lower() for i in PIDs_images]
    del label_data

    # Order the images and segmentations in the scores ordering
    ordering = list()
    for pid in ranked_PIDs:
        if pid.lower() in PIDs_images:
            ordering.append(PIDs_images.index(pid))
        else:
            print(f'[WORC Warning] Patient {pid} not in images list!')

    PIDs_images = [PIDs_images[i] for i in ordering]
    images = [images[i] for i in ordering]
    segmentations = [segmentations[i] for i in ordering]

    # Print the middle segmented slice from each patient based on ranking
    print('Print the middle segmented slice from each patient based on ranking.')
    if output_zip is not None:
        zipf = zipfile.ZipFile(output_zip,
                               'w', zipfile.ZIP_DEFLATED, allowZip64=True)

    if output_itk is not None:
        # Determine spacing factor
        print("Determining spacings factor.")
        spacings_x = list()
        spacings_y = list()
        for idx, im in enumerate(images):
            print(('Processing patient {} / {}: {}.').format(str(idx + 1), str(len(images)), PIDs_images[idx]))
            im = sitk.ReadImage(im)
            spacings_x.append(im.GetSpacing()[0])
            spacings_y.append(im.GetSpacing()[1])
        # NOTE: Used in future feature
        resample = [min(spacings_x), min(spacings_y)]

    for idx in range(0, len(images)):
        print(('Processing patient {} / {}: {}.').format(str(idx + 1), str(len(images)), PIDs_images[idx]))
        im = sitk.ReadImage(images[idx])
        seg = sitk.ReadImage(segmentations[idx])
        pid = PIDs_images[idx]
        score = ranked_scores[idx]
        if scores == 'percentages':
            score = abs(int(score))

        fname = str(score) + '_' + pid + '_TrueLabel_' + str(ranked_truths[idx]) + '_slice.png'
        if 'win' in platform:
            # Windows has issues with sorting based on '-', so replace with
            # 'min'
            if float(ranked_scores[idx]) < 0:
                fname = 'min' + fname[1:]

        if output_zip is not None:
            output_name = os.path.join(os.path.dirname(output_zip), fname)
            output_name_zoom = os.path.join(os.path.dirname(output_zip), 'zoom' + str(zoomfactor) + '_' + fname)
        else:
            output_name = None
            output_name_zoom = None

        imslice, maskslice = pi.slicer(image=im,
                                       mask=seg,
                                       output_name=output_name,
                                       output_name_zoom=output_name_zoom)

        if output_zip is not None:
            # Print PNGs and comine in ZIP
            zipf.write(output_name, fname)
            zipf.write(output_name_zoom, 'zoom_' + fname)
            os.remove(output_name)
            os.remove(output_name_zoom)

        if output_itk is not None:
            # Combine slices in 3D image
            print('WIP')

        del im, seg, imslice, maskslice


def plot_ranked_posteriors(estimator, pinfo, label_type=None,
                           ensemble_method='top_N', ensemble_size=100,
                           output_csv=None):
    # Read the inputs
    prediction = pd.read_hdf(estimator)
    if label_type is None:
        # Assume we want to have the first key
        label_type = prediction.keys()[0]

    # Determine the predicted score per patient
    print('Determining posterior per patient.')
    y_truths, y_scores, y_predictions, PIDs_scores =\
        plot_estimator_performance(prediction,
                                   pinfo,
                                   [label_type],
                                   alpha=0.95,
                                   ensemble_method=ensemble_method,
                                   ensemble_size=ensemble_size,
                                   output='scores')

    # Extract all scores for each patient
    print('Aggregating scores per patient over all crossval iterations.')
    scores = dict()
    truths = dict()

    def aggregate_scores(y_truths_in, y_scores_in, PIDs_scores_in):
        y_truths_flat = flatten_object(y_truths_in)
        y_scores_flat = flatten_object(y_scores_in)
        PIDs_scores_flat = flatten_object(PIDs_scores_in)

        for yt, ys, pid in zip(y_truths_flat, y_scores_flat, PIDs_scores_flat):
            if pid not in scores.keys():
                # No scores yet for patient, create list
                scores[pid] = list()
                truths[pid] = yt
            scores[pid].append(ys)

        # Take the mean for each patient and rank them
        scores_means = {pid: np.mean(scores[pid]) for pid in scores.keys()}

        # Rank according to mean scores
        ranking = np.argsort(list(scores_means.values()))
        ranked_PIDs = [list(scores_means.keys())[r] for r in ranking]

        ranked_mean_scores = [scores_means[r] for r in ranked_PIDs]
        ranked_scores = [scores[r] for r in ranked_PIDs]
        ranked_truths = [truths[r] for r in ranked_PIDs]
        return ranked_PIDs, ranked_truths, ranked_mean_scores, ranked_scores

    # Gather ground truth for each pid
    pid_truths = dict()
    for y, p in zip(y_truths, PIDs_scores):
        for k, v in zip(p, y):
            pid_truths[k] = v

    if len(label_type.split(',')) != 1:
        # Multiclass
        ranked_PIDs = dict()
        ranked_truths = dict()
        ranked_mean_scores = dict()
        ranked_scores = dict()
        total_scores = list()
        means = list()
        for lnum, label in enumerate(label_type.split(',')):
            # Select only values for this label
            y_truths_thislabel = np.asarray(y_truths)[:, :, lnum]
            y_scores_thislabel = np.asarray(y_scores)[:, :, lnum]

            # Rank the patients and scores
            ranked_PIDs_label, ranked_truths_label, ranked_mean_scores_label, ranked_scores_label =\
                aggregate_scores(y_truths_thislabel, y_scores_thislabel, PIDs_scores)

            ranked_PIDs[label] = ranked_PIDs_label
            ranked_truths[label] = ranked_truths_label
            ranked_mean_scores[label] = ranked_mean_scores_label
            ranked_scores[label] = ranked_scores_label

            means.append(f"Mean_{label}")
            total_scores.extend([f"Score_{label}_{i}" for i in range(max([len(score) for score in ranked_scores_label]))])

        # Write output to csv
        unique_pids = list(set(flatten_object(PIDs_scores)))
        # FIXME: bug in scores, so only give the means
        if output_csv is not None:
            print("Writing output scores to CSV.")
            header = ['PatientID', 'TrueLabel', 'Predicted'] + means   #+ total_scores

            with open(output_csv, 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(header)

                for pid in unique_pids:
                    pid_means = list()
                    pid_truth = pid_truths[pid]
                    pid_scores = list()
                    for lnum, label in enumerate(label_type.split(',')):
                        p_index = ranked_PIDs[label].index(pid)
                        pid_means.append(ranked_mean_scores[label][p_index])
                        pid_scores.extend(ranked_scores[label][p_index])

                    towrite = [pid, str(pid_truth), np.argmax(pid_means)] + pid_means   #+ pid_scores
                    writer.writerow(towrite)

    else:
        # Single Label
        ranked_PIDs, ranked_truths, ranked_mean_scores, ranked_scores =\
            aggregate_scores(y_truths, y_scores, PIDs_scores)

        # Write output to csv
        maxlen = max([len(score) for score in scores.values()])
        if output_csv is not None:
            print("Writing output scores to CSV.")
            header = ['PatientID', 'TrueLabel', 'Probability']
            for i in range(0, maxlen):
                header.append('Score' + str(i+1))

            with open(output_csv, 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(header)

                for pid, truth, smean, scores in zip(ranked_PIDs, ranked_truths, ranked_mean_scores, ranked_scores):
                    towrite = [str(pid), str(truth), str(smean)]
                    for s in scores:
                        towrite.append(str(s))

                    writer.writerow(towrite)

    return ranked_mean_scores, ranked_truths, ranked_PIDs


def plot_ranked_scores(estimator, pinfo, label_type, scores='percentages',
                       images=[], segmentations=[], ensemble_method='top_N',
                       ensemble_size=100,
                       output_csv=None, output_zip=None, output_itk=None):
    '''
    Rank the patients according to their average score. The score can either
    be the average posterior or the percentage of times the patient was
    classified correctly in the cross validations. Additionally,
    the middle slice of each patient is plot and saved according to the ranking.

    Parameters
    ----------
    estimator: filepath, mandatory
        Path pointing to the .hdf5 file which was is the output of the
        trainclassifier function.

    pinfo: filepath, mandatory
        Path pointint to the .txt file which contains the patient label
        information.

    label_type: string, default None
        The name of the label predicted by the estimator. If None,
        the first label from the prediction file will be used.

    scores: string, default percentages
        Type of scoring to be used. Either 'posteriors' or 'percentages'.

    images: list, optional
        List containing the filepaths to the ITKImage image files of the
        patients.

    segmentations: list, optional
        List containing the filepaths to the ITKImage segmentation files of
        the patients.

    ensemble_method: string, optional
        Method to be used for ensembling.

    ensemble_size: int, optional
        If top_N method is used, number of workflows to be included in ensemble.

    output_csv: filepath, optional
        If given, the scores will be written to this csv file.

    output_zip: filepath, optional
        If given, the images will be plotted and the pngs saved to this
        zip file.

    output_itk: filepath, optional
        WIP

    '''
    prediction = pd.read_hdf(estimator)
    if label_type is None:
        # Assume we want to have the first key
        label_type = prediction.keys()[0]

    if scores == 'posteriors':
        ranked_scores, ranked_truths, ranked_PIDs =\
            plot_ranked_posteriors(estimator=estimator,
                                   pinfo=pinfo,
                                   label_type=label_type,
                                   ensemble_method=ensemble_method,
                                   ensemble_size=ensemble_size,
                                   output_csv=output_csv)

    elif scores == 'percentages':
        if prediction[prediction.keys()[0]].config['CrossValidation']['Type'] == 'LOO':
            print('Cannot rank percentages for LOO, returning dummies.')
            ranked_scores = ranked_truths = ranked_PIDs = []
            with open(output_csv, 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['LOO: Cannot rank percentages.'])
        else:
            ranked_scores, ranked_truths, ranked_PIDs =\
                plot_ranked_percentages(estimator=estimator,
                                        pinfo=pinfo,
                                        label_type=label_type,
                                        ensemble_method=ensemble_method,
                                        ensemble_size=ensemble_size,
                                        output_csv=output_csv)
    else:
        message = ('{} is not a valid scoring method!').format(str(scores))
        raise WORCKeyError(message)

    if output_zip is not None or output_itk is not None:
        # FIXME: check for multilabel by checking type
        if isinstance(ranked_scores, list):
            if images:
                # Rerank the scores split per ground truth class: negative for 0, positive for 1
                ranked_scores_temp = list()
                for l, p in zip(ranked_truths, ranked_scores):
                    if l == 0:
                        ranked_scores_temp.append(-p)
                    else:
                        ranked_scores_temp.append(p)

                ranked_scores = ranked_scores_temp
                ranking = np.argsort(ranked_scores)
                ranked_scores = [ranked_scores[r] for r in ranking]
                ranked_truths = [ranked_truths[r] for r in ranking]
                ranked_PIDs = [ranked_PIDs[r] for r in ranking]

                # Convert to lower to later on overcome matching errors
                ranked_PIDs = [i.lower() for i in ranked_PIDs]

                # Plot the ranked images
                plot_ranked_images(pinfo=pinfo,
                                   label_type=label_type,
                                   images=images,
                                   segmentations=segmentations,
                                   ranked_truths=ranked_truths,
                                   ranked_scores=ranked_scores,
                                   ranked_PIDs=ranked_PIDs,
                                   output_zip=output_zip,
                                   output_itk=output_itk,
                                   scores=scores)

            else:
                # Make dummy
                if output_zip is not None:
                    zipfile.ZipFile(output_zip,
                                    'w', zipfile.ZIP_DEFLATED, allowZip64=True)

        else:
            # Make dummy
            if output_zip is not None:
                zipfile.ZipFile(output_zip,
                                'w', zipfile.ZIP_DEFLATED, allowZip64=True)


if __name__ == '__main__':
    main()
