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

from WORC.tests import test_helpers as th
import os
import glob
from WORC.plotting.plot_errors import plot_errors


def test_plot_errors():
    """Testing of plot error function."""
    # Check if example data directory exists
    example_data_dir = th.find_exampledatadir()
    outputdir = r'C:\Users\Martijn Starmans\WORC\output\WORC_Example_STWStrategyHN_WORC335_201020'

    # Features
    featurefiles = glob.glob(os.path.join(outputdir, 'Features', 'features_predict*.hdf5'))
    featurefiles = ['CT_0=' + ','.join(featurefiles)]

    patientinfo = r'C:\Users\Martijn Starmans\Documents\GitHub\WORCTutorial\Data\Examplefiles\pinfo_HN.csv'
    label_type = ['imaginary_label_1']
    featurenames = ['PREDICT_original_semf_Sex_CT_0', 'PREDICT_original_semf_Age_CT_0']

    posteriors_csv = r'C:\Users\Martijn Starmans\WORC\output\WORC_Example_STWStrategyHN_WORC335_201020\Evaluation\RankedPosteriors_all_0.csv'

    plot_errors(featurefiles, patientinfo, label_type, featurenames,
                posteriors_csv=posteriors_csv)


if __name__ == "__main__":
    test_plot_errors()
