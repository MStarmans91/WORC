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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
import numpy as np
from sklearn.decomposition import PCA, SparsePCA, KernelPCA
from sklearn.manifold import TSNE
from WORC.IOparser.file_io import load_features
import WORC.IOparser.config_io_classifier as config_io
from WORC.featureprocessing.Imputer import Imputer


def Decomposition(features, patientinfo, config, output, label_type=None,
                  verbose=True):
    """
    Perform decompositions to two components of the feature space.

    Useage is similar to StatisticalTestFeatures.

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

    # Create output folder if required
    if not os.path.exists(os.path.dirname(output)):
        os.makedirs(os.path.dirname(output))

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

    # Detect NaNs, otherwise first feature imputation is required
    if any(np.isnan(a) for a in np.asarray(feature_values).flatten()):
        print('\t [WARNING] NaNs detected, applying median imputation')
        imputer = Imputer(missing_values=np.nan, strategy='median')
        imputer.fit(feature_values)
        feature_values = imputer.transform(feature_values)

    # -----------------------------------------------------------------------
    # Perform decomposition
    print("Performing decompositions.")
    label_value = label_data['label']
    label_name = label_data['label_name']

    # Reduce to two components for plotting
    n_components = 2

    for i_class, i_name in zip(label_value, label_name):
        classlabels = i_class.ravel()

        class1 = [i for j, i in enumerate(feature_values) if classlabels[j] == 1]
        class2 = [i for j, i in enumerate(feature_values) if classlabels[j] == 0]

        f = plt.figure(figsize=(20, 15))

        # -------------------------------------------------------
        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(feature_values)
        explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
        class1_pca = pca.transform(class1)
        class2_pca = pca.transform(class2)

        # Plot PCA
        ax = plt.subplot(2, 3, 1)

        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        ax.scatter(class1_pca[:, 0], class1_pca[:, 1], color='blue')
        ax.scatter(class2_pca[:, 0], class2_pca[:, 1], color='green')
        ax.set_title(f'PCA: {round(explained_variance_ratio, 3)} variance.')

        # -------------------------------------------------------
        # Fit Sparse PCA
        pca = SparsePCA(n_components=n_components)
        pca.fit(feature_values)
        class1_pca = pca.transform(class1)
        class2_pca = pca.transform(class2)

        # Plot Sparse PCA
        ax = plt.subplot(2, 3, 2)

        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        ax.scatter(class1_pca[:, 0], class1_pca[:, 1], color='blue')
        ax.scatter(class2_pca[:, 0], class2_pca[:, 1], color='green')
        ax.set_title('Sparse PCA.')

        # -------------------------------------------------------
        # Fit Kernel PCA
        fnum = 3
        for kernel in ['linear', 'poly', 'rbf']:
            try:
                pca = KernelPCA(n_components=n_components, kernel=kernel)
                pca.fit(feature_values)
                class1_pca = pca.transform(class1)
                class2_pca = pca.transform(class2)

                # Plot Sparse PCA
                ax = plt.subplot(2, 3, fnum)

                plt.subplots_adjust(hspace=0.3, wspace=0.2)
                ax.scatter(class1_pca[:, 0], class1_pca[:, 1], color='blue')
                ax.scatter(class2_pca[:, 0], class2_pca[:, 1], color='green')
                ax.set_title(('Kernel PCA: {} .').format(kernel))
                fnum += 1
            except ValueError as e:
                # Sometimes, a specific kernel does not work, just continue
                print(f'[Error] {e}: skipping kernel {kernel}.')
                continue

        # -------------------------------------------------------
        # Fit t-SNE
        tSNE = TSNE(n_components=n_components)
        class_all = class1 + class2
        class_all_tsne = tSNE.fit_transform(class_all)

        class1_tSNE = class_all_tsne[0:len(class1)]
        class2_tSNE = class_all_tsne[len(class1):]

        # Plot Sparse tSNE
        ax = plt.subplot(2, 3, 6)

        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        ax.scatter(class1_tSNE[:, 0], class1_tSNE[:, 1], color='blue')
        ax.scatter(class2_tSNE[:, 0], class2_tSNE[:, 1], color='green')
        ax.set_title('t-SNE.')

        # -------------------------------------------------------
        # Maximize figure to get correct spacings
        # mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())

        # High DTI to  make sure we save the maximized image
        f.savefig(output, dpi=600)
        print(("Decomposition saved as {} !").format(output))
