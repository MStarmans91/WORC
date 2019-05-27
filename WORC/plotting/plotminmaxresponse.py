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

import pandas as pd
import argparse
import WORC.processing.label_processing as lp
import os
import glob
from natsort import natsorted
import numpy as np
from PREDICT.plotting.getfeatureimages import getfeatureimages
import scipy


def main():
    parser = argparse.ArgumentParser(description='Radiomics results')
    parser.add_argument('-im', '--im', metavar='im',
                        nargs='+', dest='im', type=str, required=False,
                        help='List of patient image files (nii)')
    parser.add_argument('-seg', '--seg', metavar='seg',
                        nargs='+', dest='seg', type=str, required=False,
                        help='List of patient segmentation files (nii)')
    parser.add_argument('-imtest', '--imtest', metavar='imtest',
                        nargs='+', dest='imtest', type=str, required=False,
                        help='List of patient image files of test database (nii)')
    parser.add_argument('-segtest', '--segtest', metavar='segtest',
                        nargs='+', dest='segtest', type=str, required=False,
                        help='List of patient segmentation files of test database (nii)')
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
                        nargs='+', dest='out', type=str, required=False,
                        help='Output folder')
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

    if type(args.im) is list:
        args.im = ''.join(args.im)

    if type(args.seg) is list:
        args.seg = ''.join(args.seg)

    if type(args.imtest) is list:
        args.imtest = ''.join(args.imtest)

    if type(args.segtest) is list:
        args.segtest = ''.join(args.segtest)

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

    # Save image of min and max response per feature
    image_type = 'CT'
    # imname = '/*/*/image.nii.gz'
    # segname = '/*/*/seg*.nii.gz'
    imname = '/*preop_Tumor.nii.gz'
    segname = '/*Tumor_mask.nii.gz'
    for fl in labels:
        if 'cf_' not in fl:
            features = featvect[fl]['all']
            maxind = np.argmax(features)
            minind = np.argmin(features)

            if args.im is not None:
                im_min = glob.glob(os.path.join(args.im, patient_IDs[minind]) + imname)
                if len(im_min) == 0:
                    # Search in testing folder
                    im_min = glob.glob(os.path.join(args.imtest, patient_IDs[minind]) + imname)[0]
                else:
                    im_min = im_min[0]

                seg_min = glob.glob(os.path.join(args.seg, patient_IDs[minind]) + segname)
                if len(seg_min) == 0:
                    # Search in testing folder
                    seg_min = glob.glob(os.path.join(args.segtest, patient_IDs[minind]) + segname)[0]
                else:
                    seg_min = seg_min[0]

                im_max = glob.glob(os.path.join(args.im, patient_IDs[maxind]) + imname)
                if len(im_max) == 0:
                    # Search in testing folder
                    im_max = glob.glob(os.path.join(args.imtest, patient_IDs[maxind]) + imname)[0]
                else:
                    im_max = im_max[0]

                seg_max = glob.glob(os.path.join(args.seg, patient_IDs[maxind]) + segname)
                if len(seg_max) == 0:
                    # Search in testing folder
                    seg_max = glob.glob(os.path.join(args.segtest, patient_IDs[maxind]) + segname)[0]
                else:
                    seg_max = seg_max[0]

                if 'LBP' in fl:
                    # Save LBP image
                    LBPim = getfeatureimages(im_min, seg_min,
                                             image_type=image_type,
                                             types=['LBP'])[0]
                    filename = fl + '_min_' + patient_IDs[minind] + '.png'
                    savename = os.path.join(args.out, filename)
                    scipy.misc.imsave(savename, np.fliplr(np.rot90(LBPim, 3)))

                    LBPim = getfeatureimages(im_max, seg_max,
                                             image_type=image_type,
                                             types=['LBP'])[0]
                    filename = fl + '_max_' + patient_IDs[maxind] + '.png'
                    savename = os.path.join(args.out, filename)
                    scipy.misc.imsave(savename, np.fliplr(np.rot90(LBPim, 3)))
                elif 'Gabor' in fl:
                    # Save Gabor image
                    Gind = fl.index('Gabor')
                    Aind = fl.index('A')
                    gabor_settings = dict()
                    gabor_settings['gabor_frequencies'] = [float(fl[Gind + 6:Aind])]
                    try:
                        gabor_settings['gabor_angles'] = [float(fl[Aind + 1:Aind +1 + 4])]
                    except ValueError:
                        # 0.0: two numbers
                        gabor_settings['gabor_angles'] = [float(fl[Aind + 1:Aind +1 + 3])]

                    Gaborim = getfeatureimages(im_min, seg_min,
                                               image_type=image_type,
                                               gabor_settings=gabor_settings,
                                               types=['Gabor'])[0]
                    filename = fl + '_min_' + patient_IDs[minind] + '.png'
                    savename = os.path.join(args.out, filename)
                    scipy.misc.imsave(savename, np.fliplr(np.rot90(Gaborim, 3)))

                    Gaborim = getfeatureimages(im_max, seg_max,
                                               image_type=image_type,
                                               gabor_settings=gabor_settings,
                                               types=['Gabor'])[0]
                    filename = fl + '_max_' + patient_IDs[maxind] + '.png'
                    savename = os.path.join(args.out, filename)
                    scipy.misc.imsave(savename, np.fliplr(np.rot90(Gaborim, 3)))
                elif 'sf_' in fl or 'hf_' in fl or 'tf_GL' in fl:
                    # Save segmentation
                    Shapeim = getfeatureimages(im_min, seg_min,
                                               image_type=image_type,
                                               types=['Shape'])[0]
                    filename = fl + '_min_' + patient_IDs[minind] + '_seg.png'
                    savename = os.path.join(args.out, filename)
                    scipy.misc.imsave(savename, np.fliplr(np.rot90(Shapeim, 3)))

                    Shapeim = getfeatureimages(im_max, seg_max,
                                               image_type=image_type,
                                               types=['Shape'])[0]
                    filename = fl + '_max_' + patient_IDs[maxind] + '_seg.png'
                    savename = os.path.join(args.out, filename)
                    scipy.misc.imsave(savename, np.fliplr(np.rot90(Shapeim, 3)))

                # Save images
                Histogramim = getfeatureimages(im_min, seg_min,
                                           image_type=image_type,
                                           types=['Histogram'])[0]
                Histogramim[Histogramim == -1000] = 0
                filename = fl + '_min_' + patient_IDs[minind] + '_im.png'
                savename = os.path.join(args.out, filename)
                scipy.misc.imsave(savename, np.fliplr(np.rot90(Histogramim, 3)))

                Histogramim = getfeatureimages(im_max, seg_max,
                                           image_type=image_type,
                                           types=['Histogram'])[0]
                Histogramim[Histogramim == -1000] = 0
                filename = fl + '_max_' + patient_IDs[maxind] + '_im.png'
                savename = os.path.join(args.out, filename)
                scipy.misc.imsave(savename, np.fliplr(np.rot90(Histogramim, 3)))


if __name__ == '__main__':
    main()
