#!/usr/bin/env python

# Copyright 2025-2025 Biomedical Imaging Group Rotterdam, Departments of
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

import numbers
from collections.abc import Sequence
import numpy as np
from collections.abc import Sized

# Mostly remakes of sklearn private functions

def _is_arraylike(x):
    return isinstance(x, Sized) or hasattr(x, "shape")

def _num_samples(x):
    """Estimate the number of samples in x."""
    if hasattr(x, "shape"):
        if len(x.shape) == 0:
            raise TypeError("Singleton array cannot be considered a valid collection.")
        return x.shape[0]
    elif hasattr(x, "__len__"):
        return len(x)
    else:
        raise TypeError(f"Expected sequence or array-like, got {type(x)}")

def _safe_indexing(X, indices):
    """Safely index arrays/lists with given indices."""
    if isinstance(X, (list, tuple)):
        return [X[i] for i in indices]
    elif isinstance(X, dict):
        return {key: _safe_indexing(val, indices) for key, val in X.items()}
    elif hasattr(X, "__getitem__"):
        return X[indices]
    else:
        raise TypeError(f"Cannot index object of type {type(X)}")

def _make_indexable(x):
    """Ensure the input supports indexing (i.e. slicing by integers)."""
    if hasattr(x, "__getitem__"):
        return x
    return np.array(x)

def validate_fit_params(X, fit_params, indices=None):
    """Validate fit parameters against X, optionally applying indexing."""
    fit_params_validated = {}

    n_samples = _num_samples(X)

    for param_key, param_value in fit_params.items():
        try:
            if (
                not _is_arraylike(param_value)
                or _num_samples(param_value) != n_samples
            ):
                fit_params_validated[param_key] = param_value
            else:
                indexed = _make_indexable(param_value)
                if indices is not None:
                    indexed = _safe_indexing(indexed, indices)
                fit_params_validated[param_key] = indexed
        except Exception:
            # If something goes wrong (e.g. param_value doesn't support len()), just keep as-is
            fit_params_validated[param_key] = param_value

    return fit_params_validated

def estimator_has(attr, *, delegates=("best_estimator_", "estimator_", "estimator")):
    """
    Check if we can delegate a method to the underlying estimator.

    Parameters
    ----------
    attr : str
        Name of the attribute the delegate might or might not have.

    delegates : tuple of str, default=("estimator_", "estimator")
        A tuple of sub-estimator(s) to check if we can delegate the `attr` method.

    Returns
    -------
    check : function
        Function to check if the delegate has the attribute.

    Raises
    ------
    ValueError
        Raised when none of the delegates are present in the object.
    """

    def check(self):
        for delegate in delegates:
            if hasattr(self, delegate):
                delegator = getattr(self, delegate)
                if isinstance(delegator, Sequence):
                    return getattr(delegator[0], attr)
                else:
                    return getattr(delegator, attr)

        raise ValueError(f"None of the delegates {delegates} are present in the class.")

    return check

def normalize_score_results(scores, scaler_score_key="score"):
    """Creates a scoring dictionary based on the type of `scores`"""
    if isinstance(scores[0], dict):
        # multimetric scoring
        return aggregate_score_dicts(scores)
    # scaler
    return {scaler_score_key: scores}

def aggregate_score_dicts(scores):
    """Aggregate the list of dict to dict of np ndarray

    The aggregated output of _aggregate_score_dicts will be a list of dict
    of form [{'prec': 0.1, 'acc':1.0}, {'prec': 0.1, 'acc':1.0}, ...]
    Convert it to a dict of array {'prec': np.array([0.1 ...]), ...}

    Parameters
    ----------

    scores : list of dict
        List of dicts of the scores for all scorers. This is a flat list,
        assumed originally to be of row major order.

    Example
    -------

    >>> scores = [{'a': 1, 'b':10}, {'a': 2, 'b':2}, {'a': 3, 'b':3},
    ...           {'a': 10, 'b': 10}]                         # doctest: +SKIP
    >>> _aggregate_score_dicts(scores)                        # doctest: +SKIP
    {'a': array([1, 2, 3, 10]),
     'b': array([10, 2, 3, 10])}
    """
    return {
        key: (
            np.asarray([score[key] for score in scores])
            if isinstance(scores[0][key], numbers.Number)
            else [score[key] for score in scores]
        )
        for key in scores[0]
    }
