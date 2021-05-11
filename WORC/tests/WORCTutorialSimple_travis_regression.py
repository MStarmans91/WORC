# Welcome to the tutorial of WORC: a Workflow for Optimal Radiomics
# Classification! It will provide you with basis knowledge and practical
# skills on how to run the WORC. For advanced topics and WORCflows, please see
# the other notebooks provided with this tutorial. For installation details,
# see the ReadMe.md provided with this tutorial.

# This tutorial interacts with WORC through SimpleWORC and is especially
# suitable for first time usage.

# import neccesary packages
from WORC import BasicWORC
import os

# These packages are only used in analysing the results
import pandas as pd
import json
import fastr
import glob

# If you don't want to use your own data, we use the following example set,
# see also the next code block in this example.
from WORC.exampledata.datadownloader import download_HeadAndNeck

# Define the folder this script is in, so we can easily find the example data
script_path = os.path.dirname(os.path.abspath(__file__))

# Determine whether you would like to use WORC for classification or regression
modus = 'regression'


def main():
    """Execute WORC Tutorial experiment."""
    # ---------------------------------------------------------------------------
    # Input
    # ---------------------------------------------------------------------------
    # The minimal inputs to WORC are:
    #   - Images
    #   - Segmentations
    #   - Labels
    #
    # In SimpleWORC, we assume you have a folder "datadir", in which there is a
    # folder for each patient, where in each folder there is a image.nii.gz and a mask.nii.gz:
    #           Datadir
    #               Patient_001
    #                   image.nii.gz
    #                   mask.nii.gz
    #               Patient_002
    #                   image.nii.gz
    #                   mask.nii.gz
    #               ...
    #
    #
    # You can skip this part if you use your own data.
    # In the example, We will use open source data from the online XNAT platform
    # at https://xnat.bmia.nl/data/archive/projects/stwstrategyhn1. This dataset
    # consists of CT scans of patients with Head and Neck tumors. We will download
    # a subset of 20 patients in this folder. You can change this settings if you
    # like

    nsubjects = 20  # use "all" to download all patients
    # NOTE: already downloaded for classification, so simply set the data path
    data_path = '/home/travis/build/MStarmans91/WORC/WORCTutorial/Data'
    # data_path = r'C:\Users\Martijn Starmans\Documents\GitHub\WORCTutorial\Data'
    # download_HeadAndNeck(datafolder=data_path, nsubjects=nsubjects)

    # Identify our data structure: change the fields below accordingly
    # if you use your own data.
    imagedatadir = os.path.join(data_path, 'stwstrategyhn1')
    image_file_name = 'image.nii.gz'
    segmentation_file_name = 'mask.nii.gz'

    # File in which the labels (i.e. outcome you want to predict) is stated
    # Again, change this accordingly if you use your own data.
    label_file = os.path.join(data_path, 'Examplefiles', 'pinfo_HN.csv')

    # Name of the label you want to predict
    if modus == 'classification':
        # Classification: predict a binary (0 or 1) label
        label_name = 'imaginary_label_1'

    elif modus == 'regression':
        # Regression: predict a continuous label
        label_name = 'Age'

    # Determine whether we want to do a coarse quick experiment, or a full lengthy
    # one. Again, change this accordingly if you use your own data.
    coarse = True

    # Give your experiment a name
    experiment_name = 'Example_STWStrategyHN_Regression'

    # Instead of the default tempdir, let's but the temporary output in a subfolder
    # in the same folder as this script
    tmpdir = os.path.join(script_path, 'WORC_' + experiment_name)

    # ---------------------------------------------------------------------------
    # The actual experiment
    # ---------------------------------------------------------------------------

    # Create a Basic WORC object to be able to set features ourselves
    experiment = BasicWORC(experiment_name)

    # Set the input data according to the variables we defined earlier
    experiment.images_from_this_directory(imagedatadir,
                                          image_file_name=image_file_name)
    experiment.segmentations_from_this_directory(imagedatadir,
                                                 segmentation_file_name=segmentation_file_name)

    # Use features from classification to be quicker
    # Locate output folder
    outputfolder = fastr.config.mounts['output']
    classification_experiment_folder =\
        os.path.join(outputfolder, 'WORC_Example_STWStrategyHN')

    feature_files = glob.glob(os.path.join(classification_experiment_folder,
                                           'Features',
                                           'features_*.hdf5'))
    features_train = {os.path.splitext(os.path.basename(f))[0].split('CT_0_')[1]: f for f in feature_files}
    experiment.features_train.append(features_train)

    # Labels
    experiment.labels_from_this_file(label_file)
    experiment.predict_labels([label_name])

    # Use the standard workflow for binary classification
    if modus == 'classification':
        experiment.binary_classification(coarse=coarse)
    elif modus == 'regression':
        experiment.regression(coarse=coarse)

    # Set the temporary directory
    experiment.set_tmpdir(tmpdir)

    # Extra's
    semantics_file = os.path.join(data_path, 'Examplefiles', 'semantics_HN.csv')
    experiment.semantics_from_this_file(semantics_file)
    experiment.add_evaluation()
    # experiment.add_config_overrides({'General':
    #                                   {'FeatureCalculators': '[predict/CalcFeatures:1.0, pyradiomics/Pyradiomics:1.0]'}
    #                                  })

    experiment.set_multicore_execution()

    # Run the experiment!
    experiment.execute()

    # NOTE:  Precomputed features can be used instead of images and masks
    # by instead using ``experiment.features_from_this_directory(featuresdatadir)`` in a similar fashion.

    # ---------------------------------------------------------------------------
    # Analysis of results
    # ---------------------------------------------------------------------------

    # There are two main outputs: the features for each patient/object, and the overall
    # performance. These are stored as .hdf5 and .json files, respectively. By
    # default, they are saved in the so-called "fastr output mount", in a subfolder
    # named after your experiment name.

    # Read the overall peformance
    experiment_folder = os.path.join(outputfolder, 'WORC_' + experiment_name)
    performance_file = os.path.join(experiment_folder, 'performance_all_0.json')
    if not os.path.exists(performance_file):
        raise ValueError('No performance file found: your network has failed.')

    with open(performance_file, 'r') as fp:
        performance = json.load(fp)

    # Print the output performance
    print("\n Performance:")
    stats = performance['Statistics']
    for k, v in stats.items():
        print(f"\t {k} {v}.")

    # NOTE: the performance is probably horrible, which is expected as we ran
    # the experiment on coarse settings. These settings are recommended to only
    # use for testing: see also below.

    # ---------------------------------------------------------------------------
    # Tips and Tricks
    # ---------------------------------------------------------------------------

    # For tips and tricks on running a full experiment instead of this simple
    # example, adding more evaluation options, debuggin a crashed network etcetera,
    # please go to https://worc.readthedocs.io/en/latest/static/user_manual.html

    # Some things we would advice to always do:
    #   - Run actual experiments on the full settings (coarse=False):
    #       coarse = False
    #       experiment.binary_classification(coarse=coarse)
    #   Note: this will result in more computation time. We therefore recommmend
    #   to run this script on either a cluster or high performance PC. If so,
    #   you may change the execution to use multiple cores to speed up computation
    #   just before before experiment.execute():
    #       experiment.set_multicore_execution()
    #
    #   - Add extensive evaluation: experiment.add_evaluation() before experiment.execute():
    #       experiment.add_evaluation()


if __name__ == '__main__':
    main()
