Quick start guide
=================

This manual will show users how to install WORC, configure WORC and construct and run simple networks.

.. _installation-chapter:

Installation
------------

You can install WORC either using pip, or from the source code.

Installing via pip
``````````````````

You can simply install WORC using ``pip``:

.. code-block:: bash

    pip install WORC

.. note:: You might want to consider installing ``WORC`` in a `virtualenv <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_


Installing from source code
```````````````````````````

To install from source code, use git via the command-line:

.. code-block:: bash

    git clone https://github.com/MStarmans91/WORC.git  # for http
    git clone ssh://git@github.com:MStarmans91/WORC.git # for ssh

.. _subsec-installing:

To install to your current Python environment, run:

.. code-block:: bash

    cd WORC/
    pip install .

This installs the scripts and packages in the default system folders. For
Windows this is the python ``site-packages`` directory for the WORC python
library. For Ubuntu this is in the ``/usr/local/lib/python3.x/dist-packages/`` folder.

.. note:: If you want to develop WORC, you might want to use ``pip install -e .`` to get an editable install

.. note:: You might want to consider installing ``WORC`` in a `virtualenv <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_

Windows installation
````````````````````

Visual studio


Hello World
------------

If you don't have a dataset of your own you want to use, you can
download an example:

.. code-block:: python

    import WORC

    WORC.exampledata.download('STWHeadAndNeck')
    datadir = WORC.getfolder()

The data will be located in the same folder as where you run the code,
which is stored in the WORC object, which we retreived as above.

The minimal inputs for a WORC experiment are:
1. Images
2. Segmentations
3. Labels

Alternatively, you can use precomputed features instead of images and
segmentations, see below for an example on how to use these.

In the data directory, there is a folder for each patient/object. Inside,
there is an image ('image.nii.gz') and a 'mask.nii.gz'. The quick
way to use WORC is to keep this structure (i.e. one main directory,
subdirectories for each patient/object), although you can vary
the image and mask filename and format. Additionally, a label file has
been created, in which the outcome 'imaginary_label_1' is stored,
which is what we are going to try and predict.

.. code-block:: python

    # Define names of image and segmentation in each patient folder
    image_file_name = 'image.nii.gz'
    segmentation_file_name = 'mask.nii.gz'

    # File in which the labels (i.e. outcome you want to predict) is stated
    import os
    label_file = os.path.join(datadir, 'pinfo.csv')

    # Name of the label you want to predict
    label_name = 'imaginary_label_1'

You can give the experiment a name, which will be used for storing the output.
For this examples, we will do a quick, coarse experiment: we advice to always
start with a coarse experiment for testing the setup, than run a full experiment
to get the optimal performance.

.. code-block:: python

    # Determine whether we want to do a coarse quick experiment, or a full lengthy one
    coarse = True

    # Give your experiment a name
    experiment_name = 'Example_STWStrategyMMD'

After defining the inputs and settings, we are ready to create and run the actual experiment

.. code-block:: python
    # Create a WORC object
    I = IntermediateFacade(name)

    # Set the input data according to the variables we defined earlier
    I.images_from_this_directory(datadir,
                                 image_file_name=image_file_name)
    I.segmentations_from_this_directory(datadir,
                                        segmentation_file_name=segmentation_file_name)
    I.labels_from_this_file(label_file)
    I.predict_labels([label_name])

    # Use the standard workflow for binary classification
    I.binary_classification(coarse=coarse)

    # Run the experiment!
    I.execute()

.. note:: Precomputed features can be used instead of images and masks by instead using ``I.features_from_this_directory()`` in a similar fashion.

There are two outputs: the features for each patient/object, and the overall
performance. These are stored as .hdf5 and .json files, respectively. By
default, they are saved in the so-called "fastr output mount", in a subfolder
named after your experiment name.

.. code-block:: python
    import pandas as pd
    import json
    import fastr

    # Locate output folder
    outputfolder = fastr.config.mounts['output']
    experiment_folder = os.path.join(outputfolder, experiment)

    print(f"Your output is stored in {experiment_folder}.")

    # Read the features for the first patient
    featurefile_p1 = os.path.join(experiment_folder, 'Features', 'features_CT_0_patient1.hdf5')
    features_p1 = pd.read_hdf(featurefile_p1)

    # Read the overall peformance
    performance_file = os.path.join(experiment_folder, 'performance_0.json')
    with open(performance_file, 'r') as fp:
      performance = json.load(fp)


Print the outputs to the command line:

.. code-block:: python
    # Print the feature values and names
    for v, l in zip(features_p1.values, features_p1.lbels):
      print(f"Features {l} has value {v}.)

    # Print the output performance
    stats = performance['Statistics']
    for k, v in stats.iteritems():
      print(f"The {k} was {v}.")

Extensive evaluation, including feature importance, patient ranking from
typical to atypical, and ROC analysis, can be simply added, see section ...





Advanced Tutorials
-------------------

To start out using WORC, we recommend you to follow the tutorial located in the
[WORCTutorial Github](https://github.com/MStarmans91/WORCTutorial). Besides some more advanced tutorials,
the main tutorial can be found in the WORCTutorial.ipynb Jupyter notebook. Instructions on how
to use the notebook can be found in the Github.

If you run into any issue, you can first debug your network using
`the fastr trace tool <https://fastr.readthedocs.io/en/stable/static/user_manual.html#debugging-a-network-run-with-errors/>`_.
If you're stuck, feel free to post an issue on the `WORC Github <https://github.com/MStarmans91/WORC/>`_.
