.. _config-chapter:

Configuration
=============

Introduction
------------

WORC has defaults for all settings so it can be run out of the box to test the examples.
However, you may want to alter the fastr configuration to your system settings, e.g.
to locate your input and output folders and how much you want to parallelize the execution.

Fastr will search for a config file named ``config.py`` in the ``$FASTRHOME`` directory
(which defaults to ``~/.fastr/`` if it is not set). So if ``$FASTRHOME`` is set the ``~/.fastr/``
will be ignored. Additionally, .py files from the ``$FASTRHOME/config.d`` folder will be parsed
as well. You will see that upon installation, WORC has already put a ``WORC_config.py`` file in the
``config.d`` folder.

% Note: Above was originally from quick start
As ``WORC`` and the default tools used are mostly Python based, we've chosen
to put our configuration in a ``configparser`` object. This has several
advantages:

1. The object can be treated as a python dictionary and thus is easily adjusted.
2. Second, each tool can be set to parse only specific parts of the configuration,
   enabling us to supply one file to all tools instead of needing many parameter files.


Creation and interaction
-------------------------

The default configuration is generated through the
:py:meth:`WORC.defaultconfig() <WORC.defaultconfig()>`
function. You can then change things as you would in a dictionary and
then append it to the configs source:


.. code-block:: python

    >>> network = WORC.WORC('somename')
    >>> config = network.defaultconfig()
    >>> config['Classification']['classifier'] = 'RF'
    >>> network.configs.append(config)

When executing the :py:meth:`WORC.set() <WORC.set()>` command, the config objects are saved as
.ini files in the ``WORC.fastr_tempdir`` folder and added to the
:py:meth:`WORC.fastrconfigs() <WORC.fastrconfigs()>` source.

Below are some details on several of the fields in the configuration.
Note that for many of the fields, we currently only provide one default
value. However, when adding your own tools, these fields can be adjusted
to your specific settings.

WORC performs Combined Algorithm Selection and Hyperparameter (CASH)
optimization. The configuration determines how the optimization is
performed and which hyperparameters and models will be included.
Repeating specific models/parameters in the config will make them more
likely to be used, e.g.

.. code-block:: python

    >>> config['Classification']['classifiers'] = 'SVM, SVM, LR'

means that the SVM is 2x more likely to be tested in the model selection than LR.

.. note::

    All fields in the config must either be supplied as strings. A
    list can be created by using commas for separation, e.g.
    :py:meth:`Network.create_source <'value1, value2, ... ')>`.

Contents
--------
The config object can be indexed as ``config[key][subkey] = value``. The various keys, subkeys, and the values
(description, defaults and options) can be found below.

.. include:: ../autogen/WORC.config.rst

Details on each section of the config can be found below.


.. _config-General:

General
~~~~~~~
These fields contain general settings for when using WORC.
For more info on the Joblib settings, which are used in the Joblib
Parallel function, see `here <https://pythonhosted.org/joblib/parallel.html>`__. When you run
WORC on a cluster with nodes supporting only a single core to be used
per node, e.g. the BIGR cluster, use only 1 core and threading as a
backend.

**Description:**

.. include:: ../autogen/config/WORC.config_General_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_General_defopts.rst


.. _config-Labels:

Labels
~~~~~~~~
Set the label used for classification.

This part is quite important, as it should match your label file.
Suppose your patientclass.txt file you supplied as source for labels
looks like this:


+----------+--------+--------+
| Patient  | Label1 | Label2 |
+==========+========+========+
| patient1 | 1      | 0      |
+----------+--------+--------+
| patient2 | 2      | 1      |
+----------+--------+--------+
| patient3 | 1      | 5      |
+----------+--------+--------+

You can supply a single label or multiple labels split by commas, for
each of which an estimator will be fit. For example, suppose you simply
want to use Label1 for classification, then set:

.. code-block:: python

   config['Labels']['label_names'] = 'Label1'


If you want to first train a classifier on Label1 and then Label2,
set: ``config[Labels][label_names] = Label1, Label2``


**Description:**

.. include:: ../autogen/config/WORC.config_Labels_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_Labels_defopts.rst


.. _config-Preprocessing:

Preprocessing
~~~~~~~~~~~~~
The preprocessing node acts before the feature extraction on the image.
Additionally, scans with imagetype CT (see later in the tutorial) provided
as DICOM are scaled to Hounsfield Units. For more details on the preprocessing
options, please see
:ref:`the additional functionality chapter <additonalfunctionality-chapter>`.

**Description:**

.. include:: ../autogen/config/WORC.config_Preprocessing_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_Preprocessing_defopts.rst


.. _config-Segmentix:

Segmentix
~~~~~~~~~
These fields are only important if you specified using the segmentix
tool in the general configuration.

**Description:**

.. include:: ../autogen/config/WORC.config_Segmentix_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_Segmentix_defopts.rst


.. _config-ImageFeatures:

ImageFeatures
~~~~~~~~~~~~~
If using the PREDICT toolbox for feature extraction, you can specify some
settings for the
feature computation here. Also, you can select if the certain features
are computed or not.

**Description:**

.. include:: ../autogen/config/WORC.config_ImageFeatures_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_ImageFeatures_defopts.rst


.. _config-PyRadiomics:

PyRadiomics
~~~~~~~~~~~~~
If using the PyRadiomics toolbox, you can specify some settings for the
feature computation here. For more information, see
https://pyradiomics.readthedocs.io/en/latest/customization.htm.

**Description:**

.. include:: ../autogen/config/WORC.config_PyRadiomics_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_PyRadiomics_defopts.rst


.. _config-ComBat:

ComBat
~~~~~~~~~~~~~
If using the ComBat toolbox, you can specify some settings for the
feature harmonization here. For more information, see
https://github.com/Jfortin1/ComBatHarmonization.

**Description:**

.. include:: ../autogen/config/WORC.config_ComBat_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_ComBat_defopts.rst


.. _config-FeatPreProcess:

FeatPreProcess
~~~~~~~~~~~~~~
Before the features are given to the classification function, and thus the
hyperoptimization, these can be preprocessed as following.

**Description:**

.. include:: ../autogen/config/WORC.config_FeatPreProcess_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_FeatPreProcess_defopts.rst

.. _config-OneHotEncoding:

OneHotEncoding
~~~~~~~~~~~~~~~~
Optionally, you can use OneHotEncoding on specific features. For more
information on why and how this is done, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html.
By default, this is not done, as WORC does not know for which
specific features you would like to do this.

**Description:**

.. include:: ../autogen/config/WORC.config_OneHotEncoding_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_OneHotEncoding_defopts.rst

.. _config-Imputation:

Imputation
~~~~~~~~~~~~~~~~
These settings are used for feature imputation. Note that these settings
are actually used
in the hyperparameter optimization. Hence you can provide multiple
values per field, of which random samples will be drawn of which finally
the best setting in combination with the other hyperparameters is
selected.

**Description:**

.. include:: ../autogen/config/WORC.config_Imputation_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_Imputation_defopts.rst


.. _config-FeatureScaling:

FeatureScaling
~~~~~~~~~~~~~~
Determines which method is applied to scale each feature.


**Description:**

.. include:: ../autogen/config/WORC.config_FeatureScaling_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_FeatureScaling_defopts.rst

.. _config-Featsel:

Featsel
~~~~~~~
Define feature selection methods. Note that these settings are
actually used in the hyperparameter optimization. Hence you can provide
multiple values per field, of which random samples will be drawn of
which finally the best setting in combination with the other
hyperparameters is selected. Again, these should be formatted as string
containing the actual values, e.g. value1, value2.

**Description:**

.. include:: ../autogen/config/WORC.config_Featsel_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_Featsel_defopts.rst

.. _config-SelectFeatGroup:

SelectFeatGroup
~~~~~~~~~~~~~~~
If the PREDICT and/or PyRadiomics feature computation tools are used,
then you can do a gridsearch among the various feature groups for the
optimal combination. Here, you determine which groups can be selected.

**Description:**

.. include:: ../autogen/config/WORC.config_SelectFeatGroup_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_SelectFeatGroup_defopts.rst

.. _config-Resampling:

Resampling
~~~~~~~~~~~~~~~~
Before performing the hyperoptimization, you can use various resampling
techniques to resample (under-sampling, over-sampling, or both) the data.
All methods are adopted from `imbalanced learn <https://imbalanced-learn.readthedocs.io/>`_.


**Description:**

.. include:: ../autogen/config/WORC.config_Resampling_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_Resampling_defopts.rst


.. _config-Classification:

Classification
~~~~~~~~~~~~~~
Determine settings for the classification in the hyperoptimization. Most of the
classifiers are implemented using sklearn; hence descriptions of the
hyperparameters can also be found there.

Defaults for XGB are based on
https://towardsdatascience.com/doing-xgboost-hyper-parameter-tuning-the-smart-way-part-1-of-2-f6d255a45dde
and https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

Note, as XGB and AdaBoost take significantly longer to fit (3x), they
are picked less often by default.

**Description:**

.. include:: ../autogen/config/WORC.config_Classification_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_Classification_defopts.rst


.. _config-CrossValidation:

CrossValidation
~~~~~~~~~~~~~~~
When using cross validation, specify the following settings.

**Description:**

.. include:: ../autogen/config/WORC.config_CrossValidation_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_CrossValidation_defopts.rst


.. _config-HyperOptimization:

HyperOptimization
~~~~~~~~~~~~~~~~~
Specify the hyperparameter optimization procedure here.

**Description:**

.. include:: ../autogen/config/WORC.config_HyperOptimization_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_HyperOptimization_defopts.rst


.. _config-Ensemble:

Ensemble
~~~~~~~~
WORC supports ensembling of workflows. This is not a default approach in
radiomics, hence the default is to not use it and select only the best
performing workflow.

**Description:**

.. include:: ../autogen/config/WORC.config_Ensemble_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_Ensemble_defopts.rst


.. _config-Evaluation:

Evaluation
~~~~~~~~~~
In the evaluation of the performance, several adjustments can be made.

**Description:**

.. include:: ../autogen/config/WORC.config_Evaluation_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_Evaluation_defopts.rst


.. _config-Bootstrap:

Bootstrap
~~~~~~~~~
Besides cross validation, WORC supports bootstrapping on the test set for performance evaluation.

**Description:**

.. include:: ../autogen/config/WORC.config_Bootstrap_description.rst

**Defaults and Options:**

.. include:: ../autogen/config/WORC.config_Bootstrap_defopts.rst
