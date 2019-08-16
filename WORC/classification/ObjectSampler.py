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

from imblearn import over_sampling, under_sampling, combine
import numpy as np
from sklearn.utils import check_random_state
import WORC.addexceptions as ae


class ObjectSampler(object):
    """
    Samples objects for learning based on various under-, over- and combined sampling methods.

    The choice of included methods is largely based on:

    He, Haibo, and Edwardo A. Garcia. "Learning from imbalanced data."
    IEEE Transactions on Knowledge & Data Engineering 9 (2008): 1263-1284.


    """
    def __init__(self, method,
                 sampling_strategy='auto',
                 SMOTE_ratio=1,
                 SMOTE_neighbors=5,
                 n_jobs=1,
                 n_neighbors=3,
                 ):

        # Initialize a random state
        self.random_seed = np.random.randint(5000)
        self.random_state = check_random_state(random_seed)

        # Initialize all objects as Nones: overriden when required by functions
        self.sampling_strategy = None
        self.object = None
        self.n_neighbors = None
        self.n_jobs = None

        if method == 'RandomUnderSampling':
            self.init_RandomUnderSampling(sampling_strategy)
        elif method == 'NearMiss':
            self.init_NearMiss(sampling_strategy, n_neighbors, n_jobs)
        elif method == 'NeigbourhoodCleaningRule':
            self.init_NeigbourhoodCleaningRule()
        elif method == 'RandomOverSampling':
            self.init_RandomOverSampling(sampling_strategy)
        elif method == 'ADASYN':
            self.init_ADASYN()
        elif method == 'BorderlineSMOTE':
            self.init_BorderlineSMOTE()
        elif method == 'SMOTEENN':
            self.init_SMOTEENN()
        elif method == 'SMOTETomek':
            self.init_SMOTETomek()
        else:
            raise ae.WORCKeyError(f'{method} is not a valid sampling method!')

    def init_RandomUnderSampling(self, sampling_strategy):
        self.object = under_sampling.RandomUnderSampler(sampling_strategy=sampling_strategy,
                                                        random_state=self.random_state)
        self.sampling_strategy = sampling_strategy

    def init_NearMiss(self, sampling_strategy, n_neighbors, n_jobs):
        self.object = under_sampling.NearMiss(sampling_strategy=sampling_strategy,
                                              random_state=self.random_state,
                                              n_neighbors=n_neighbors,
                                              n_jobs=n_jobs)

        self.sampling_strategy = sampling_strategy
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def init_RandomOverSampling(self, sampling_strategy):
        self.object = over_sampling.RandomOverSampler(sampling_strategy=sampling_strategy,
                                                      random_state=self.random_state)
        self.sampling_strategy = sampling_strategy

    def init_SMOTE(self):
        sm = SMOTE(random_state=None,
                   ratio=para_estimator['SampleProcessing_SMOTE_ratio'],
                   m_neighbors=para_estimator['SampleProcessing_SMOTE_neighbors'],
                   kind='borderline1',
                   n_jobs=para_estimator['SampleProcessing_SMOTE_n_cores'])

        self.object = sm

    def fit(self, **kwargs):
        self.object.fit(**kwargs)

    def fit(self, **kwargs):
        self.object.fit(**kwargs)