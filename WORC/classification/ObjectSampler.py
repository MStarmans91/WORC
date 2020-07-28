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
                 n_jobs=1,
                 n_neighbors=3,
                 k_neighbors=5,
                 threshold_cleaning=0.5,
                 verbose=True):
        """Initialize object."""
        # Initialize a random state
        self.random_seed = np.random.randint(5000)
        self.random_state = check_random_state(self.random_seed)

        # Initialize all objects as Nones: overriden when required by functions
        self.object = None
        self.sampling_strategy = None
        self.n_jobs = None
        self.n_neighbors = None
        self.k_neighbors = None
        self.threshold_cleaning = None
        self.verbose = verbose

        if method == 'RandomUnderSampling':
            self.init_RandomUnderSampling(sampling_strategy)
        elif method == 'NearMiss':
            self.init_NearMiss(sampling_strategy, n_jobs)
        elif method == 'NeighbourhoodCleaningRule':
            self.init_NeighbourhoodCleaningRule(sampling_strategy, n_neighbors,
                                                n_jobs, threshold_cleaning)
        elif method == 'RandomOverSampling':
            self.init_RandomOverSampling(sampling_strategy)
        elif method == 'ADASYN':
            self.init_ADASYN(sampling_strategy, n_neighbors, n_jobs)
        elif method == 'BorderlineSMOTE':
            self.init_BorderlineSMOTE(k_neighbors, n_jobs)
        elif method == 'SMOTE':
            self.init_SMOTE(k_neighbors, n_jobs)
        elif method == 'SMOTEENN':
            self.init_SMOTEENN(sampling_strategy)
        elif method == 'SMOTETomek':
            self.init_SMOTETomek(sampling_strategy)
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
                                              n_jobs=n_jobs)

        self.sampling_strategy = sampling_strategy
        self.n_jobs = n_jobs

    def init_NeighbourhoodCleaningRule(self, sampling_strategy, n_neighbors,
                                       n_jobs, threshold_cleaning):
        """Creata a NeighbourhoodCleaningRule sampler object."""
        self.object =\
            under_sampling.NeighbourhoodCleaningRule(sampling_strategy=sampling_strategy,
                                                     threshold_cleaning=threshold_cleaning,
                                                     n_jobs=n_jobs)

        self.sampling_strategy = sampling_strategy
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.threshold_cleaning = threshold_cleaning

    def init_RandomOverSampling(self, sampling_strategy):
        """Creata a random over sampler object."""
        self.object = over_sampling.RandomOverSampler(sampling_strategy=sampling_strategy,
                                                      random_state=self.random_state)
        self.sampling_strategy = sampling_strategy

    def init_ADASYN(self, sampling_strategy, n_neighbors, n_jobs):
        """Creata a ADASYN sampler object."""
        self.object = over_sampling.ADASYN(sampling_strategy=sampling_strategy,
                                           random_state=self.random_state,
                                           n_neighbors=n_neighbors,
                                           n_jobs=n_jobs)

        self.sampling_strategy = sampling_strategy
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def init_BorderlineSMOTE(self, k_neighbors, n_jobs):
        """Creata a BorderlineSMOTE sampler object."""
        self.object =\
            over_sampling.BorderlineSMOTE(random_state=self.random_state,
                                          k_neighbors=k_neighbors,
                                          n_jobs=n_jobs)

        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

    def init_SMOTE(self, k_neighbors, n_jobs):
        """Creata a SMOTE sampler object."""
        self.object =\
            over_sampling.SMOTE(random_state=self.random_state,
                                k_neighbors=k_neighbors,
                                n_jobs=n_jobs)

        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

    def init_SMOTEENN(self, sampling_strategy):
        """Creata a SMOTEEN sampler object."""
        self.object =\
            combine.SMOTEENN(random_state=self.random_state,
                             sampling_strategy=sampling_strategy)

        self.sampling_strategy = sampling_strategy

    def init_SMOTETomek(self, sampling_strategy):
        """Creata a SMOTE Tomek sampler object."""
        self.object =\
            combine.SMOTETomek(random_state=self.random_state,
                               sampling_strategy=sampling_strategy)

        self.sampling_strategy = sampling_strategy

    def fit(self, *args, **kwargs):
        """Fit a sampler object."""
        if hasattr(self.object, 'fit_resample'):
            if self.verbose:
                print('[WORC WARNING] Sampler does have fit_resample construction: not fitting now.')
        else:
            # Object has a fit-transform construction
            self.object.fit(*args, **kwargs)

    def transform(self, *args, **kwargs):
        """Transform objects with a fitted sampler."""
        if hasattr(self.object, 'fit_resample'):
            if self.verbose:
                print('[WORC WARNING] Sampler does have fit_resample construction: fit and resampling.')
            try:
                return self.object.fit_resample(*args, **kwargs)
            except ValueError as message:
                message = str(message)
                message = 'The ObjectSampler could not ' +\
                          'resample the objects with ' +\
                          'the given parameters. ' +\
                          'Probably your number of samples ' +\
                          'is too small for the parameters ' +\
                          'used. Original error: ' + message
                raise ae.WORCValueError(message)

        else:
            return self.object.transform(*args, **kwargs)
