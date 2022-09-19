#!/usr/bin/env python

# Copyright 2016-2022 Biomedical Imaging Group Rotterdam, Departments of
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

import numpy as np
from sklearn.model_selection import train_test_split
import WORC.addexceptions as ae
from WORC.processing.label_processing import load_labels
import pandas as pd
import WORC.processing.label_processing as lp


def createfixedsplits(label_file=None, label_type=None, patient_IDs=None,
                      stratify=True, test_size=0.2, N_iterations=1,
                      modus='singlelabel', output=None):
    '''
    Create fixed splits for a random-split cross-validation.
    

        Parameters
        ----------
            label_file : filepath
                CSV file containing the labels of the patients.
            label_type: list of strings
                labels to extracted from the label file, e.g. ['label1']
            patient_IDs: list of strings
                names of patients to take into account. If None, take all
            stratify: Boolean
                If True, splits are stratified. In this case, you need to provide
                label data.
            test_size: float
                Percentage of patients in test set per iteration.
            N_iterations: integer
                Number of cross-validation iterations
            modus: str
                singlelabel or regression. Multilabel not implemented yet.
            output: filepath
                csv filename to save output to.
        
        Returns
        -------
            df: pandas Dataframe
                Fixed splits created.
            
    '''
    # Check whether input is valid
    if patient_IDs is None:
        if label_file is not None and label_type is not None:
            # Read the label file
            label_data = load_labels(label_file, label_type)
            patient_IDs = label_data['patient_IDs']

        else:
            raise ae.WORCValueError('Either a label file and label type or patient_IDs need to be provided!')
    else:
        if stratify is True:
            if label_file is not None and label_type is not None:
                # Extract data for specific patients only
                label_data, _ = lp.findlabeldata(label_file,
                                                 label_type,
                                                 pids=patient_IDs)
            else:
                raise ae.WORCValueError('A label file and label type needs to be provided for stratified splitting!')
    
    # Create the stratification object
    if stratify:
        if modus == 'singlelabel':
            stratify = label_data['label'][0].tolist()
        elif modus == 'multilabel':
            # Create a stratification object from the labels
            # Label = 0 means no label equals one
            # Other label numbers refer to the label name that is 1
            stratify = list()
            labels = label_data['label']
            for pnum in range(0, len(labels[0])):
                plabel = 0
                for lnum, slabel in enumerate(labels):
                    if slabel[pnum] == 1:
                        plabel = lnum + 1
                stratify.append(plabel)
        else:
            raise ae.WORCKeyError('{} is not a valid modus!').format(modus)
                           
    pd_dict = dict()
    for i in range(N_iterations):
        print(f'Splitting iteration {i + 1} / {N_iterations}')
        # Create a random seed for the splitting
        random_seed = np.random.randint(5000)

        # Define stratification
        unique_patient_IDs, unique_indices =\
            np.unique(np.asarray(patient_IDs), return_index=True)
            
        if modus == 'regression' or not stratify:
            unique_stratify = None
        else:
            unique_stratify = [stratify[i] for i in unique_indices]

        # Split, throw error when dataset is too small for split ratio's
        try:
            unique_PID_train, indices_PID_test\
                = train_test_split(unique_patient_IDs,
                                   test_size=test_size,
                                   random_state=random_seed,
                                   stratify=unique_stratify)
        except ValueError as e:
            e = str(e) + ' Increase the size of your test set.'
            raise ae.WORCValueError(e)

        # Check for all IDs if they are in test or training
        indices_train = list()
        indices_test = list()
        patient_ID_train = list()
        patient_ID_test = list()
        for num, pid in enumerate(patient_IDs):
            if pid in unique_PID_train:
                indices_train.append(num)

                # Make sure we get a unique ID
                if pid in patient_ID_train:
                    n = 1
                    while str(pid + '_' + str(n)) in patient_ID_train:
                        n += 1
                    pid = str(pid + '_' + str(n))
                patient_ID_train.append(pid)
            else:
                indices_test.append(num)

                # Make sure we get a unique ID
                if pid in patient_ID_test:
                    n = 1
                    while str(pid + '_' + str(n)) in patient_ID_test:
                        n += 1
                    pid = str(pid + '_' + str(n))
                patient_ID_test.append(pid)

        # Add to train object
        pd_dict[str(i) + '_train'] = patient_ID_train

        # Test object has to be same length as training object
        extras = [""]*(len(patient_ID_train) - len(patient_ID_test))
        patient_ID_test.extend(extras)
        pd_dict[str(i) + '_test'] = patient_ID_test

    # Convert into pandas dataframe for easy use and conversion
    df = pd.DataFrame(pd_dict)

    # Write output if required
    if output is not None:
        print("Writing Output.")
        df.to_csv(output)

    return df


def test():
    patient_IDs = ['HN1004', 'HN1077', 'HN1088', 'HN1146', 'HN1159', 'HN1192', 'HN1259', 'HN1260',
                   'HN1323', 'HN1331', 'HN1339', 'HN1342', 'HN1372', 'HN1491', 'HN1501', 'HN1519',
                   'HN1524', 'HN1554', 'HN1560', 'HN1748']
    createfixedsplits(label_file=r'C:\Users\Martijn Starmans\Documents\GitHub\WORCTutorial\Data\Examplefiles\pinfo_HN.csv',
                      patient_IDs=patient_IDs, stratify=True,
                      label_type=['imaginary_label_1'], N_iterations=3, output='fixedsplits.csv')
    
    
if __name__ == "__main__":
    test()



