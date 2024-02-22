#!/usr/bin/env python

# Copyright 2024 Biomedical Imaging Group Rotterdam, Department of
# Radiology and Nuclear Medicine, Erasmus MC, Rotterdam, The Netherlands
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

import shap
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import tikzplotlib
import pandas as pd
import argparse
from WORC.plotting.compute_CI import compute_confidence as CI
import numpy as np
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_curve
import csv
from WORC.plotting.plot_estimator_performance import plot_estimator_performance



def plot_shap_values(prediction, pinfo, ensemble_method='top_N',
                     ensemble_size=1, label_type=None):
    # Convert the inputs to the correct format
    if type(prediction) is list:
        prediction = ''.join(prediction)

    if type(pinfo) is list:
        pinfo = ''.join(pinfo)

    if type(ensemble_method) is list:
        ensemble_method = ''.join(ensemble_method)

    if type(ensemble_size) is list:
        ensemble_size = int(ensemble_size[0])

    if type(label_type) is list:
        label_type = ''.join(label_type)

    # Read the inputs
    prediction = pd.read_hdf(prediction)
    if label_type is None:
        # Assume we want to have the first key
        label_type = prediction.keys()[0]
    elif len(label_type.split(',')) != 1:
        # Multiclass, just take the prediction label
        label_type = prediction.keys()[0]

    N_1 = len(prediction[label_type].Y_train[0])
    N_2 = len(prediction[label_type].Y_test[0])

    # Determine the predicted score per patient
    print('Determining shap values per patient.')
    shap_values, X_test, feature_labels =\
        plot_estimator_performance(prediction, pinfo, [label_type],
                                   alpha=0.95, ensemble_method=ensemble_method,
                                   ensemble_size=ensemble_size,
                                   output='shap')
    print(shap_values)



# X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)

# # rather than use the whole training set to estimate expected values, we could summarize with
# # a set of weighted kmeans, each weighted by the number of points they represent. But this dataset
# # is so small we don't worry about it
# #X_train_summary = shap.kmeans(X_train, 50)

# def print_accuracy(f):
#     print("Accuracy = {0}%".format(100*np.sum(f(X_test) == Y_test)/len(Y_test)))
#     time.sleep(0.5) # to let the print get out before any progress bars

# # shap.initjs()

# knn = sklearn.neighbors.KNeighborsClassifier()
# knn.fit(X_train, Y_train)

# print_accuracy(knn.predict)

# explainer = shap.KernelExplainer(knn.predict_proba, X_train)
# shap_values = explainer.shap_values(X_test)

if __name__ == '__main__':
    # Test
    prediction = r"C:\\Users\\795023\WORC\\output\\WORC_Example_STWStrategyHN_py311\\estimator_all_0.hdf5"
    pinfo = r"C:\\Users\\795023\\Documents\\GitHub\\WORCTutorial\\Data\\Examplefiles\\pinfo_HN.csv"
    plot_shap_values(prediction, pinfo)



# Plotting: to fix
# shap_values = explainer.shap_values(X_test.iloc[0,:])
# shap.force_plot(explainer.expected_value[0], shap_values[0], X_test.iloc[0,:])

# print(explainer.expected_value[0], shap_values[0], X_test.iloc[0,:])

# 
# shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)
