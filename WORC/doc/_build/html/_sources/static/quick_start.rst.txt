Quick start guide
=================

This manual will show users how to install WORC, configure WORC and construct and run a simple experiment.

.. _installation-chapter:

Installation
------------

You can install WORC either using pip, or from the source code. We strongly advice you to install ``WORC`` in a `virtualenv <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.



**Installing via pip**


You can simply install WORC using ``pip``:

.. code-block:: bash

    pip install WORC

**Installing from source code**

To install from source code, use git via the command-line:

.. code-block:: bash

    git clone https://github.com/MStarmans91/WORC.git  # for http
    git clone ssh://git@github.com:MStarmans91/WORC.git # for ssh

**Windows installation**

On Windows, we strongly recommend to install python through the
`Anaconda distribution <https://www.anaconda.com/distribution/#windows>`_.

Regardless of your installation, you will need `Microsoft Visual Studio <https://visualstudio.microsoft.com/vs/features/python>`_: the Community
edition can be downloaded and installed for free.

If you still get an error similar to error: ``Microsoft Visual C++ 14.0 is required. Get it with``
`Microsoft Visual C++ Build Tools   <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019>`_
, please follow the respective link and install the requirements.


Tutorials
---------
To start out using WORC, we recommend you to follow the tutorials located in the
`WORCTutorial Github <https://github.com/MStarmans91/WORCTutorial/>`_. This repository
contains tutorials for an introduction to WORC, as well as more advanced workflows.

If you run into any issue, you can first debug your network using
`the fastr trace tool <https://fastr.readthedocs.io/en/stable/static/user_manual.html#debugging-a-network-run-with-errors/>`_.
If you're stuck, feel free to post an issue on the `WORC Github <https://github.com/MStarmans91/WORC/>`_.


Running an experiment
---------------------

Below is the same script as found in the SimpleWORC tutorial found in the `WORCTutorial Github <https://github.com/MStarmans91/WORCTutorial/>`_.
In this tutorial, we will make use of the ``SimpleWORC`` facade, which simplifies interacting with ``WORC``.
Additional information on ``SimpleWORC`` can be found in the
:ref:`User Manual <usermanual-chapter>`. This chapter also includes the documentation on using the ``BasicWORC`` facade,
which is slightly more advanced, and the ``WORC`` object directly, which provides the most advanced options.

Import packages
```````````````

First, import WORC and some additional python packages.

.. code-block:: python

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

Input
`````
The minimal inputs to WORC are:

    1. Images
    2. Segmentations
    3. Labels

In SimpleWORC, we assume you have a folder "datadir", in which there is a
folder for each patient, where in each folder there is a image.nii.gz and a mask.nii.gz:

    * Datadir

      * Patient_001
          * image.nii.gz
          * mask.nii.gz
      * Patient_002
          * image.nii.gz
          * mask.nii.gz
      * ...

In the example, we will use open source data from the online
`BMIA XNAT platform <https://xnat.bmia.nl/data/archive/projects/stwstrategyhn1/>`_
This dataset consists of CT scans of patients with Head and Neck tumors. We will download
a subset of 20 patients in this folder. You can change this settings if you like.

.. code-block:: python

    nsubjects = 20  # use "all" to download all patients
    data_path = os.path.join(script_path, 'Data')
    download_HeadAndNeck(datafolder=data_path, nsubjects=nsubjects)

.. note:: You can skip this code block if you use your own data.

Identify our data structure: change the fields below accordingly if you use your own dataset.

.. code-block:: python

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
    experiment_name = 'Example_STWStrategyHN'

    # Instead of the default tempdir, let's but the temporary output in a subfolder
    # in the same folder as this script
    tmpdir = os.path.join(script_path, 'WORC_' + experiment_name)

The actual experiment
`````````````````````

After defining the inputs, the following code can be used to run your first experiment.

.. code-block:: python

    # Create a Simple WORC object
    experiment = SimpleWORC(experiment_name)

    # Set the input data according to the variables we defined earlier
    experiment.images_from_this_directory(imagedatadir,
                                 image_file_name=image_file_name)
    experiment.segmentations_from_this_directory(imagedatadir,
                                        segmentation_file_name=segmentation_file_name)
    experiment.labels_from_this_file(label_file)
    experiment.predict_labels([label_name])

    # Use the standard workflow for binary classification
    experiment.binary_classification(coarse=coarse)

    # Set the temporary directory
    experiment.set_tmpdir(tmpdir)

    # Run the experiment!
    experiment.execute()

.. note::  Precomputed features can be used instead of images and masks by instead using ``experiment.features_from_this_directory(featuresdatadir)`` in a similar fashion.

Analysis of the results
```````````````````````
There are two main outputs: the features for each patient/object, and the overall
performance. These are stored as .hdf5 and .json files, respectively. By
default, they are saved in the so-called "fastr output mount", in a subfolder
named after your experiment name.

.. code-block:: python

    # Locate output folder
    outputfolder = fastr.config.mounts['output']
    experiment_folder = os.path.join(outputfolder, 'WORC_' + experiment_name)

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

.. note:: the performance is probably horrible, which is expected as we ran the experiment on coarse settings. These settings are recommended to only use for testing: see also below.


Tips and Tricks
```````````````

For tips and tricks on running a full experiment instead of this simple
example, adding more evaluation options, debugging a crashed network etcetera,
please go to :ref:`User Manual <usermanual-chapter>` chapter.

Some things we would advice to always do:

* Run actual experiments on the full settings (coarse=False):

.. code-block:: python

      coarse = False
      experiment.binary_classification(coarse=coarse)

.. note:: This will result in more computation time. We therefore recommmend
  to run this script on either a cluster or high performance PC. If so,
  you may change the execution to use multiple cores to speed up computation
  just before before ``experiment.execute()``:

    .. code-block:: python

          experiment.set_multicore_execution()

  This is not required when running WORC on the BIGR or SURFSara Cartesius cluster,
  as automatic detectors for these clusters have been built into SimpleWORC and BasicWORC.

* Add extensive evaluation: ``experiment.add_evaluation()`` before ``experiment.execute()``:

.. code-block:: python

      experiment.add_evaluation()

For a complete overview of all functions, please look at the
:ref:`Config chapter <config-chapter>`.
