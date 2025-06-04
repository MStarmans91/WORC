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

from sklearn.utils import check_random_state
import numpy as np
import six
import scipy
from scipy.stats import uniform
import math


class log_uniform():
    def __init__(self, loc=-1, scale=0, base=10):
        self.loc = loc
        self.scale = scale
        self.base = base
        self.uniform_dist = uniform(loc=self.loc, scale=self.scale)

    def rvs(self, size=None, random_state=None):
        if size is None:
            return np.power(self.base, self.uniform_dist.rvs(random_state=random_state))
        else:
            return np.power(self.base, self.uniform_dist.rvs(size=size, random_state=random_state))


class discrete_uniform():
    def __init__(self, loc=-1, scale=0):
        self.loc = loc
        self.scale = scale
        self.uniform_dist = uniform(loc=self.loc, scale=self.scale)

    def rvs(self, size=None, random_state=None):
        if size is None:
            return int(self.uniform_dist.rvs(random_state=random_state))
        else:
            return int(self.uniform_dist.rvs(size=size, random_state=random_state))


class boolean_uniform():
    '''
    Uniform distribution thresholded at a certain value to output booleans.

    Note: as Booleans cannot be saved in JSOn, which WORC later does, this
    object returns strings.

    '''
    def __init__(self, loc=0, scale=1, threshold=0.5):
        self.loc = loc
        self.scale = scale
        self.threshold = threshold
        self.uniform_dist = uniform(loc=self.loc, scale=self.scale)

    def rvs(self, size=None, random_state=None):
        if size is None:
            return str(self.uniform_dist.rvs(random_state=random_state) < self.threshold)
        else:
            return str([k < self.threshold for k in self.uniform_dist.rvs(size=size, random_state=random_state)])


class exp_uniform():
    def __init__(self, loc=-1, scale=0, base=math.e):
        self.loc = loc
        self.scale = scale
        self.base = base

    def rvs(self, size=None, random_state=None):
        uniform_dist = uniform(loc=self.loc, scale=self.scale)
        if size is None:
            return np.power(self.base, uniform_dist .rvs(random_state=random_state))
        else:
            return np.power(self.base, uniform_dist .rvs(size=size, random_state=random_state))


    def __len__(self):
        """Number of points that will be sampled."""
        return self.n_iter
