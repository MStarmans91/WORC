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
from ghalton import Halton
# from sobol_seq import i4_sobol_generate as Sobol
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


class AdvancedSampler(object):
    """Generator on parameters sampled from given distributions using
    numerical sequences. Based on the sklearn ParameterSampler.

    Non-deterministic iterable over random candidate combinations for hyper-
    parameter search. If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Note that before SciPy 0.16, the ``scipy.stats.distributions`` do not
    accept a custom RNG instance and always use the singleton RNG from
    ``numpy.random``. Hence setting ``random_state`` will not guarantee a
    deterministic iteration whenever ``scipy.stats`` distributions are used to
    define the parameter search space. Deterministic behavior is however
    guaranteed from SciPy 0.16 onwards.

    Read more in the :ref:`User Guide <search>`.

    Parameters
    ----------
    param_distributions : dict
        Dictionary where the keys are parameters and values
        are distributions from which a parameter is to be sampled.
        Distributions either have to provide a ``rvs`` function
        to sample from them, or can be given as a list of values,
        where a uniform distribution is assumed.

    n_iter : integer
        Number of parameter settings that are produced.

    random_state : int or RandomState
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.

    Returns
    -------
    params : dict of string to any
        **Yields** dictionaries mapping each estimator parameter to
        as sampled value.

    Examples
    --------
    >>> from WORC.classification.AdvancedSampler import HaltonSampler
    >>> from scipy.stats.distributions import expon
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> param_grid = {'a':[1, 2], 'b': expon()}
    >>> param_list = list(HaltonSampler(param_grid, n_iter=4))
    >>> rounded_list = [dict((k, round(v, 6)) for (k, v) in d.items())
    ...                 for d in param_list]
    >>> rounded_list == [{'b': 0.89856, 'a': 1},
    ...                  {'b': 0.923223, 'a': 1},
    ...                  {'b': 1.878964, 'a': 2},
    ...                  {'b': 1.038159, 'a': 2}]
    True
    """
    def __init__(self, param_distributions, n_iter, random_state=None,
                 method='Halton'):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        self.method = method

        if method == 'Halton':
            self.Halton = Halton(len(self.param_distributions.keys()))

    def __iter__(self):
        # Create a random state to be used
        rnd = check_random_state(self.random_state)

        # Generate the sequence generator
        if self.method == 'Halton':
            sequence = self.Halton.get(self.n_iter)
        elif self.method == 'Sobol':
            sequence = Sobol(len(self.param_distributions.keys()), self.n_iter)

        # Always sort the keys of a dictionary, for reproducibility
        items = sorted(self.param_distributions.items())
        for i in six.moves.range(self.n_iter):
            sample = sequence[i]
            params = dict()
            for ind, (k, v) in enumerate(items):
                point = sample[ind]
                # Check if the parameter space is a distribution or a list
                if hasattr(v, "rvs"):
                    print(point)
                    # Parameter space is a distribution, hence sample
                    params[k] = v.ppf(point)
                else:
                    # Parameter space is a list, so select an index
                    point = int(round(point*float(len(v) - 1)))
                    print(point)
                    params[k] = v[point]
            yield params

        # For reproducibility, reset sampler if needed
        if self.method == 'Halton':
            self.Halton.reset()

    def __len__(self):
        """Number of points that will be sampled."""
        return self.n_iter


if __name__ == '__main__':
    random_seed = np.random.randint(1, 5000)
    random_state = check_random_state(random_seed)

    param_distributions = {'kernel': ['poly', 'RGB'],
                           'C': scipy.stats.uniform(loc=0, scale=1E6),
                           'degree': scipy.stats.uniform(loc=1, scale=6),
                           'coef0': scipy.stats.uniform(loc=0, scale=1),
                           'gamma': scipy.stats.uniform(loc=1E-5, scale=1),
                           'histogram_features': ['True', 'False']}

    n_iter = 6

    method = 'Sobol'
    sampled_params = AdvancedSampler(param_distributions,
                                     n_iter,
                                     random_state)


    for s in sampled_params:
        print(s)
