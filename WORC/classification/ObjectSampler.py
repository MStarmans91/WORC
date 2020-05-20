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

from imblearn import over_sampling, under_sampling, combine, SMOTE
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
                 ratio=1,
                 k_neighbors=5,
                 kind='borderline-1',
                 n_jobs=1,
                 n_neighbors=3,
                 threshold_cleaning=0.5
                 ):
        """Initialize object."""
        # Initialize a random state
        self.random_seed = np.random.randint(5000)
        self.random_state = check_random_state(self.random_seed)

        # Initialize all objects as Nones: overriden when required by functions
        self.sampling_strategy = None
        self.object = None
        self.k_neighbors = None
        self.n_jobs = None
        self.ratio = None

        if method == 'RandomUnderSampling':
            self.init_RandomUnderSampling(sampling_strategy)
        elif method == 'NearMiss':
            self.init_NearMiss(sampling_strategy, n_neighbors, n_jobs)
        elif method == 'NeigbourhoodCleaningRule':
            self.init_NeigbourhoodCleaningRule(sampling_strategy, n_neighbors,
                                               n_jobs, threshold_cleaning)
        elif method == 'RandomOverSampling':
            self.init_RandomOverSampling(sampling_strategy)
        elif method == 'ADASYN':
            self.init_ADASYN(sampling_strategy, ratio, n_neighbors, n_jobs)
        elif method == 'BorderlineSMOTE':
            self.init_BorderlineSMOTE(ratio, k_neighbors, kind, n_jobs)
        elif method == 'SMOTE':
            self.init_SMOTE(ratio, k_neighbors, kind, n_jobs)
        elif method == 'SMOTEENN':
            self.init_SMOTEENN(sampling_strategy, ratio, k_neighbors, kind,
                               n_jobs)
        elif method == 'SMOTETomek':
            self.init_SMOTETomek(sampling_strategy, ratio, n_jobs)
        else:
            raise ae.WORCKeyError(f'{method} is not a valid sampling method!')

    def init_RandomUnderSampling(self, sampling_strategy):
        """Creata a random under sampler object."""
        self.object = under_sampling.RandomUnderSampler(sampling_strategy=sampling_strategy,
                                                        random_state=self.random_state)
        self.sampling_strategy = sampling_strategy

    def init_NearMiss(self, sampling_strategy, n_jobs):
        """Creata a near miss sampler object."""
        self.object = under_sampling.NearMiss(sampling_strategy=sampling_strategy,
                                              random_state=self.random_state,
                                              n_jobs=n_jobs)

        self.sampling_strategy = sampling_strategy
        self.n_jobs = n_jobs

    def init_NeigbourhoodCleaningRule(self, sampling_strategy, n_neighbors,
                                      n_jobs, threshold_cleaning):
        """Creata a NeigbourhoodCleaningRule sampler object."""
        self.object =\
            under_sampling.NeigbourhoodCleaningRule(sampling_strategy=sampling_strategy,
                                                    random_state=self.random_state,
                                                    threshold_cleaning=threshold_cleaning,
                                                    n_jobs=n_jobs)

        self.sampling_strategy = sampling_strategy
        self.threshold_cleaning = threshold_cleaning
        self.n_jobs = n_jobs

    def init_RandomOverSampling(self, sampling_strategy):
        """Creata a random over sampler object."""
        self.object = over_sampling.RandomOverSampler(sampling_strategy=sampling_strategy,
                                                      random_state=self.random_state)
        self.sampling_strategy = sampling_strategy

    def init_ADASYN(self, sampling_strategy, ratio, n_neighbors, n_jobs):
        """Creata a ADASYN sampler object."""
        self.object = over_sampling.ADASYN(sampling_strategy=sampling_strategy,
                                           random_state=self.random_state,
                                           ratio=ratio,
                                           n_neighbors=n_neighbors,
                                           n_jobs=n_jobs)

        self.sampling_strategy = sampling_strategy
        self.ratio = ratio
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def init_BorderlineSMOTE(self, ratio, k_neighbors, kind, n_jobs):
        """Creata a BorderlineSMOTE sampler object."""
        self.object =\
            over_sampling.BorderlineSMOTE(random_state=self.random_state,
                                          ratio=ratio,
                                          k_neighbors=k_neighbors,
                                          kind=kind,
                                          n_jobs=n_jobs)

        self.ratio = ratio
        self.k_neighbors = k_neighbors
        self.kind = kind
        self.n_jobs = n_jobs

    def init_SMOTE(self, ratio, k_neighbors, kind, n_jobs):
        """Creata a SMOTE sampler object."""
        sm = SMOTE(random_state=self.random_state,
                   ratio=ratio,
                   k_neighbors=k_neighbors,
                   kind=kind,
                   n_jobs=n_jobs)

        self.object = sm

        self.ratio = ratio
        self.k_neighbors = k_neighbors
        self.kind = kind
        self.n_jobs = n_jobs

    def init_SMOTEEN(self, sampling_strategy, ratio, n_jobs):
        """Creata a SMOTEEN sampler object."""
        self.object =\
            combine.SMOTEENN(random_state=self.random_state,
                             sampling_strategy=sampling_strategy,
                             ratio=ratio,
                             n_jobs=n_jobs)

        self.ratio = ratio
        self.sampling_strategy = sampling_strategy
        self.n_jobs = n_jobs

    def init_SMOTETomek(self, sampling_strategy, ratio, n_jobs):
        """Creata a SMOTE Tomek sampler object."""
        self.object =\
            combine.SMOTETomek(random_state=self.random_state,
                               sampling_strategy=sampling_strategy,
                               ratio=ratio,
                               n_jobs=n_jobs)

        self.ratio = ratio
        self.sampling_strategy = sampling_strategy
        self.n_jobs = n_jobs

    def fit(self, **kwargs):
        """Fit a sampler object."""
        self.object.fit(**kwargs)

    def transform(self, **kwargs):
        """Transform objects with a fitted sampler."""
        return self.object.transform(**kwargs)
