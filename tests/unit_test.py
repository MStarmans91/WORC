# impor neccesary packages
from WORC import SimpleWORC
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


nsubjects = 20  # use "all" to download all patients
data_path = os.path.join(script_path, 'Data')
download_HeadAndNeck(datafolder=data_path, nsubjects=nsubjects)

# Identify our data structure: change the fields below accordingly
# if you use your own data.
imagedatadir = os.path.join(data_path, 'stwstrategyhn1')
image_file_name = 'image.nii.gz'
segmentation_file_name = 'mask.nii.gz'

# File in which the labels (i.e. outcome you want to predict) is stated
# Again, change this accordingly if you use your own data.
label_file = os.path.join(data_path, 'Examplefiles', 'pinfo_HN.csv')

# Name of the label you want to predict
label_name = 'imaginary_label_1'

# Determine whether we want to do a coarse quick experiment, or a full lengthy
# one. Again, change this accordingly if you use your own data.
coarse = True

# Give your experiment a name
experiment_name = 'Example_STWStrategyHN4'


# ---------------------------------------------------------------------------
# The actual experiment
# ---------------------------------------------------------------------------

# Create a Simple WORC object
network = SimpleWORC(experiment_name)

# Set the input data according to the variables we defined earlier
network.images_from_this_directory(imagedatadir,
                             image_file_name=image_file_name)
network.segmentations_from_this_directory(imagedatadir,
                                    segmentation_file_name=segmentation_file_name)
network.labels_from_this_file(label_file)
network.predict_labels([label_name])

# Use the standard workflow for binary classification
network.binary_classification(coarse=coarse)

# Run the experiment!
network.execute()

# NOTE:  Precomputed features can be used instead of images and masks
# by instead using ``network.features_from_this_directory()`` in a similar fashion.


# ---------------------------------------------------------------------------
# Analysis of results
# ---------------------------------------------------------------------------

# There are two main outputs: the features for each patient/object, and the overall
# performance. These are stored as .hdf5 and .json files, respectively. By
# default, they are saved in the so-called "fastr output mount", in a subfolder
# named after your experiment name.

# Locate output folder
outputfolder = fastr.config.mounts['output']
experiment_folder = os.path.join(outputfolder, experiment_name)

print(f"Your output is stored in {experiment_folder}.")

# Read the features for the first patient
# NOTE: we use the glob package for scanning a folder to find specific files
feature_files = glob.glob(os.path.join(experiment_folder,
                                       'Features',
                                       'features_*.hdf5'))
featurefile_p1 = feature_files[0]
features_p1 = pd.read_hdf(featurefile_p1)

# Read the overall peformance
performance_file = os.path.join(experiment_folder, 'performance_all_0.json')
with open(performance_file, 'r') as fp:
    performance = json.load(fp)

# Print the feature values and names
print("Feature values:")
for v, l in zip(features_p1.feature_values, features_p1.feature_labels):
    print(f"\t {l} : {v}.")

# Print the output performance
print("\n Performance:")
stats = performance['Statistics']
del stats['Percentages']  # Omitted for brevity
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
#       network.binary_classification(coarse=coarse)

#   - Add extensive evaluation: network.add_evaluation() before network.execute():
#       network.add_evaluation()
