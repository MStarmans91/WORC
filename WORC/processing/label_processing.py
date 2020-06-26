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


import xnat
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import numpy as np
import os
import configparser
import WORC.addexceptions as ae
import pandas as pd


def load_labels(label_file, label_type):
    """Loads the label data from a label file

    Args:
        label_file (string): The path to the label file
        label_type (list): List of the names of the labels to load

    Returns:
        dict: A dict containing 'patient_IDs', 'label' and
         'label_type'
    """
    if not os.path.exists(label_file):
        raise ae.WORCKeyError(f'File {label_file} does not exist!')

    _, extension = os.path.splitext(label_file)
    if extension == '.txt':
        label_names, patient_IDs, label_status = load_label_txt(
            label_file)
    elif extension == '.csv':
        label_names, patient_IDs, label_status = load_label_csv(
            label_file)
    elif extension == '.ini':
        label_names, patient_IDs, label_status = load_label_XNAT(
            label_file)
    else:
        raise ae.WORCIOError(extension + ' is not valid label file extension.')

    print("Label names to extract: " + str(label_type))
    labels = list()
    for i_label in label_type:
        label_index = np.where(label_names == i_label)[0]
        if label_index.size == 0:
            raise ae.WORCValueError('Could not find label: ' + str(i_label))
        else:
            labels.append(label_status[:, label_index])

    label_data = dict()
    label_data['patient_IDs'] = patient_IDs
    label_data['label'] = labels
    label_data['label_name'] = label_type

    return label_data


def load_label_txt(input_file):
    """
    Load the patient IDs and label data from the label file

    Args:
        input_file (string): Path of the label file

    Returns:
        label_names (numpy array): Names of the different labels
        patient_ID (numpy array): IDs of patients for which label data is
         loaded
        label_status (numpy array): The status of the different labels
         for each patient
    """
    data = np.loadtxt(input_file, np.str)

    # Load and check the header
    header = data[0, :]
    if header[0] != 'Patient':
        raise ae.WORCAssertionError('First column should be patient ID!')
    else:
        # cut out the first header, only keep label header
        label_names = header[1::]

    # Patient IDs are stored in the first column
    patient_ID = data[1:, 0]

    # label status is stored in all remaining columns
    label_status = data[1:, 1:]
    label_status = label_status.astype(np.float)

    return label_names, patient_ID, label_status


def load_label_csv(input_file):
    """
    Load the patient IDs and label data from the label file

    Args:
        input_file (string): Path of the label file

    Returns:
        label_names (numpy array): Names of the different labels
        patient_ID (numpy array): IDs of patients for which label data is
         loaded
        label_status (numpy array): The status of the different labels
         for each patient
    """
    data = pd.read_csv(input_file, header=0)

    # Load and check the header
    header = data.keys()
    if header[0] != 'Patient':
        raise ae.WORCAssertionError('First column should be patient ID!')
    else:
        # cut out the first header, only keep label header
        label_names = header[1::]

    # Patient IDs are stored in the first column
    patient_ID = data['Patient'].values

    # label status is stored in all remaining columns
    label_status = data.values[:, 1:]
    label_status = label_status.astype(np.float)

    return label_names, patient_ID, label_status


def load_label_XNAT(label_info):
    """
    Load the patient IDs and label data from XNAT, Only works if you have a
    file /resources/GENETICS/files/genetics.json for each patient containing
    a single dictionary of all labels.

    Args:
        url (string): XNAT URL
        project: XNAT project ID

    Returns:
        label_names (numpy array): Names of the different labels
        patient_ID (numpy array): IDs of patients for which label data is
         loaded
        label_status (numpy array): The status of the different labels
         for each patient
    """
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

    config = load_config_XNAT(label_info)
    url = config['XNAT']['url']
    projectID = config['XNAT']['projectID']

    # Example
    # url = "http://bigr-rad-xnat.erasmusmc.nl"
    # projectID = 'LGG-Radiogenom'
    # url = label_info['url']
    # projectID = label_info['projectID']

    session = xnat.connect(url, verify=False)

    subkeys = session.projects[projectID].subjects.keys()
    subkeys.sort()
    session.disconnect()

    baseurl = url + '/data/archive/projects/' + projectID + '/subjects/'
    patient_ID = list()
    label_names = None
    label_status = list()
    for i_patient in subkeys:
        # Extra check as there are bound to be some fake patients
        if projectID in i_patient:
                patient_ID.append(i_patient)

                data = requests.get(baseurl + i_patient +
                                    '/resources/GENETICS/files/genetics.json')
                datadict = data.json()

                # Load and check the header
                if label_names is None:
                    label_names = list()
                    label_names_temp = datadict.keys()
                    for i_name in label_names_temp:
                        # convert u_str to str
                        label_names.append(str(i_name))

                label_status_temp = datadict.values()
                label_status.append(label_status_temp)

    label_names = np.asarray(label_names)
    label_status = np.asarray(label_status)

    return label_names, patient_ID, label_status


def findlabeldata(patientinfo, label_type, filenames=None,
                  objects=None, pids=None):
    """
    Load the label data and match to the unage features.

    Args:
        patientinfo (string): file with patient label data
        label_type (string): name of the label read out from patientinfo
        filenames (list): names of the patient feature files, used for matching
        objects (np.array or list): array of objects you want to order as well

    Returns:
        label_data (dict): contains patient ids, their labels and the label name
    """
    # Get the labels and patient IDs
    label_data_temp = load_labels(patientinfo, label_type)
    label_data = dict()
    patient_IDs = list()
    label_value = list()
    for i_len in range(len(label_data_temp['label_name'])):
        label_value.append(list())

    # Check per feature file / pid if there is a match in the label data
    if filenames:
        iterator = filenames
    elif pids:
        iterator = pids
    else:
        raise ae.WORCValueError('Either input pids or filenames for label matching!')

    objects_out = list()
    for i_feat, feat in enumerate(iterator):
        ifound = 0
        matches = list()
        for i_num, i_patient in enumerate(label_data_temp['patient_IDs']):
            if i_patient.lower() in str(feat).lower():

                # Match: add the patient ID to the ID's and to the matches
                patient_IDs.append(i_patient)
                matches.append(i_patient)

                # If there are feature files given, add it to the list
                if objects is not None:
                    objects_out.append(objects[i_feat])

                # For each label that we have, add the value to the label list
                for i_len in range(len(label_data_temp['label_name'])):
                    label_value[i_len].append(label_data_temp['label'][i_len][i_num])

                # Calculate how many matches we found for this (feature) file: should be one
                ifound += 1

        if ifound > 1:
            message = ('Multiple matches ({}) found in labeling for feature file {}.').format(str(matches), str(feat))
            raise ae.WORCValueError(message)

        elif ifound == 0:
            message = ('No entry found in labeling for feature file {}.').format(str(feat))
            raise ae.WORCKeyError(message)

    # Convert to arrays
    for i_len in range(len(label_value)):
        label_value[i_len] = np.asarray(label_value[i_len])

    label_data['patient_IDs'] = np.asarray(patient_IDs)
    label_data['label'] = np.asarray(label_value)
    label_data['label_name'] = label_data_temp['label_name']

    return label_data, objects_out


def load_config_XNAT(config_file_path):
    '''
    Configparser for retreiving patient data from XNAT.
    '''
    settings = configparser.ConfigParser()
    settings.read(config_file_path)

    settings_dict = {'XNAT': dict()}

    settings_dict['XNAT']['url'] =\
        str(settings['Genetics']['url'])

    settings_dict['XNAT']['projectID'] =\
        str(settings['Genetics']['projectID'])

    return settings_dict
