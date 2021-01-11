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

from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
import numpy as np
import sklearn.neighbors as nn
# from skrebate import ReliefF


class SelectMulticlassRelief(BaseEstimator, SelectorMixin):
    '''
    Object to fit feature selection based on the type group the feature belongs
    to. The label for the feature is used for this procedure.
    '''
    def __init__(self, n_neighbours=3, sample_size=1, distance_p=2, numf=None,
                 random_state=None):
        '''
        Parameters
        ----------
        n_neightbors: integer
            Number of nearest neighbours used.

        sample_size: float
            Percentage of samples used to calculate score

        distance_p: integer
            Parameter in minkov distance usde for nearest neighbour calculation

        numf: integer, default None
            Number of important features to be selected with respect to their
            ranking. If None, all are used.

        '''
        self.no_neighbours = n_neighbours
        self.sample_size = sample_size
        self.distance_p = distance_p
        self.numf = numf
        self.random_state = random_state

    def fit(self, X, y, random_state=None):
        '''
        Select only features specificed by parameters per patient.

        Parameters
        ----------
        feature_values: numpy array, mandatory
                Array containing feature values used for model_selection.
                Number of objects on first axis, features on second axis.

        feature_labels: list, mandatory
                Contains the labels of all features used. The index in this
                list will be used in the transform funtion to select features.
        '''
        # Multiclass relief function
        if len(y.shape) > 1:
            indices, _ = self.multi_class_relief(X, y,
                                                 nb=self.no_neighbours,
                                                 sample_size=self.sample_size,
                                                 distance_p=self.distance_p,
                                                 numf=self.numf,
                                                 random_state=self.random_state)
        else:
            indices, _ = self.single_class_relief(X, y,
                                                  nb=self.no_neighbours,
                                                  sample_size=self.sample_size,
                                                  distance_p=self.distance_p,
                                                  numf=self.numf,
                                                  random_state=self.random_state)

        self.selectrows = indices

    def transform(self, inputarray):
        '''
        Transform the inputarray to select only the features based on the
        result from the fit function.

        Parameters
        ----------
        inputarray: numpy array, mandatory
                Array containing the items to use selection on. The type of
                item in this list does not matter, e.g. floats, strings etc.
        '''
        return np.asarray([np.asarray(x)[self.selectrows].tolist() for x in inputarray])
        # return self.ReliefF.transform(inputarray)

    def _get_support_mask(self):
        # NOTE: Method is required for the Selector class, but can be empty
        pass

    def multi_class_relief(self, feature_set, label_set, nb=3, sample_size=1,
                           distance_p=2, numf=None, random_state=None):

        nrow, ncol = feature_set.shape
        nlabel = label_set.shape[1]
        np.random.seed(random_state)
        sample_list = np.random.choice(range(nrow), int(nrow * sample_size), replace=False)

        feature_score = np.zeros((1, ncol))

        prob = label_set.mean(axis=0)
        n_sample = dict()
        pair_score = dict()

        # find positive and negative samples for each label
        for label in range(nlabel):
            n_sample[label, 0] = []
            n_sample[label, 1] = []
            for row in sample_list:
                if label_set[row, label] == 0:
                    n_sample[label, 0].append(row)
                else:
                    n_sample[label, 1].append(row)

        for label1 in range(nlabel - 1):
            for label2 in range(label1 + 1, nlabel):
                pair_score[label1, label2] = np.zeros((1, ncol))
            if n_sample[label1, 0].__len__() >= nb and n_sample[label1, 1].__len__() >= nb:
                # find near miss for label1
                n_neighbor_finder = nn.NearestNeighbors(n_neighbors=nb, p=distance_p)
                n_neighbor_finder.fit(np.asarray(feature_set[n_sample[label1, 0], :]))
                near_miss = n_neighbor_finder.kneighbors(np.asarray(feature_set[n_sample[label1, 0], :]), return_distance=False)

                # find near hit for label1
                p_neighbor_finder = nn.NearestNeighbors(n_neighbors=nb, p=distance_p)
                p_neighbor_finder.fit(np.asarray(feature_set[n_sample[label1, 1], :]))
                near_hit = p_neighbor_finder.kneighbors(np.asarray(feature_set[n_sample[label1, 1], :]), return_distance=False)

                for label2 in range(label1 + 1, nlabel):
                    for r in range(near_miss.__len__()):
                        for c in range(nb):
                            if label_set[near_miss[r, c], label2] == 1:
                                pair_score[label1, label2] += 1.0 * (prob[label2] / (1 - prob[label1])) *\
                                                              np.abs(feature_set[n_sample[label1, 0][r], :] -
                                                                     feature_set[near_miss[r, c], :]) / nb
                    for r in range(n_sample[label1, 1].__len__()):
                        for c in range(nb):
                            if label_set[near_hit[r, c], label2] == 1:
                                pair_score[label1, label2] -= (prob[label2] /(1 - prob[label1])) *\
                                                              np.abs(feature_set[n_sample[label1, 1][r], :] -
                                                                     feature_set[near_hit[r, c], :]) / nb

        for label1 in range(nlabel - 1):
            for label2 in range(label1 + 1, nlabel):
                feature_score += pair_score[label1, label2]

        feature_score = feature_score[0]
        sorted_index = feature_score.argsort().tolist()
        sorted_index.reverse()
        sorted_index = np.array(sorted_index)
        feature_score = feature_score[sorted_index]

        if numf is None:
            numf = len(sorted_index)

        # Make sure we select at maximum all features
        numf = min(numf, len(sorted_index))
        sorted_index = sorted_index[0:numf]

        return sorted_index, feature_score

    def single_class_relief(self, feature_set, label_set, nb=3, sample_size=1,
                           distance_p=2, numf=None, random_state=None):

        nrow, ncol = feature_set.shape
        np.random.seed(random_state)
        sample_list = np.random.choice(range(nrow), int(nrow * sample_size), replace=False)

        feature_score = np.zeros((1, ncol))

        n_sample = dict()

        # find positive and negative samples for each label
        n_sample[0] = []
        n_sample[1] = []
        for row in sample_list:
            if label_set[row] == 0:
                n_sample[0].append(row)
            else:
                n_sample[1].append(row)

        if n_sample[0].__len__() >= nb and n_sample[1].__len__() >= nb:
            # find near miss for label1
            n_neighbor_finder = nn.NearestNeighbors(n_neighbors=nb, p=distance_p)
            n_neighbor_finder.fit(np.asarray(feature_set[n_sample[0], :]))
            near_miss = n_neighbor_finder.kneighbors(np.asarray(feature_set[n_sample[0], :]), return_distance=False)

            # find near hit for label1
            p_neighbor_finder = nn.NearestNeighbors(n_neighbors=nb, p=distance_p)
            p_neighbor_finder.fit(np.asarray(feature_set[n_sample[1], :]))
            near_hit = p_neighbor_finder.kneighbors(np.asarray(feature_set[n_sample[1], :]), return_distance=False)

            for r in range(near_miss.__len__()):
                for c in range(nb):
                    if label_set[near_miss[r, c]] == 1:
                        feature_score += 1.0 * np.abs(feature_set[n_sample[0][r], :] -
                                                             feature_set[near_miss[r, c], :]) / nb
            for r in range(n_sample[1].__len__()):
                for c in range(nb):
                    if label_set[near_hit[r, c]] == 1:
                        feature_score -= 1.0 * np.abs(feature_set[n_sample[1][r], :] -
                                                             feature_set[near_hit[r, c], :]) / nb

        feature_score = feature_score[0]
        sorted_index = feature_score.argsort().tolist()
        sorted_index.reverse()
        sorted_index = np.array(sorted_index)
        feature_score = feature_score[sorted_index]

        if numf is None:
            numf = len(sorted_index)

        # Make sure we select at maximum all features
        numf = min(numf, len(sorted_index))
        sorted_index = sorted_index[0:numf]

        return sorted_index, feature_score


    # def single_class_relief_skrebate(self, feature_set, label_set, nb=3, sample_size=1,
    #                        distance_p=2, numf=None):
    #     self.ReliefF = ReliefF(n_features_to_select=numf, n_neighbors=nb, n_jobs=1)
    #     self.ReliefF.fit(feature_set, label_set)
    #     return None, None
