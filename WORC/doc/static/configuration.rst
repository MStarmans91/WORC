Configuration
=============

As WORC and the default tools used are mostly Python based, we've chosen
to put our configuration in a configparser object. This has several
advantages:
1. The object can be treated as a python dictionary and thus is easily adjusted.
2. Second, each tool can be set to parse only specific parts of the configuration, enabling us to supply one file to all tools instead of needing many parameter files.

The default configuration is generated through the :py:meth:`Network.create_source <WORC.defaultconfig()>`
function. You can then change things as you would in a dictionary and
then append it to the configs source:

.. code-block:: python

    >>> network = WORC.WORC('somename')
    >>> config = network.defaultconfig()
    >>> config['Classification']['classifier'] = 'RF'
    >>> network.configs.append(config)

When executing the WORC.set() command, the config objects are saved as
.ini files in the WORC.fastr_tempdir folder and added to the
WORC.fastrconfigs() source.

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

means that the SVM is 2x more likely to be tested in the model selection than LR.

.. note::

    All fields in the config must either be supplied as strings. A list can be created by using commas for separation, e.g.
     :py:meth:`Network.create_source <'value1, value2, ... ')>`.

General
-------

These fields contain general settings for running WORC.

+-------------+------------+---------------------+--------------------+
| Field       | Possible   | Used by Tool        | Explanation        |
|             | values     |                     |                    |
+=============+============+=====================+====================+
| cross_valid | True       | PREDICT.TrainClassi | Use cross          |
| ation       | (Default), | fier                | validation         |
|             | False      |                     |                    |
+-------------+------------+---------------------+--------------------+
| Segmentix   | True,      | WORC.build()        | Use Segmentix tool |
|             | False      |                     | for segmentation   |
|             | (default)  |                     | preprocessing      |
+-------------+------------+---------------------+--------------------+
| PCE         | False      | WIP                 | WIP                |
+-------------+------------+---------------------+--------------------+
| FeatureCalc | CalcFeatur | WORC                | Specifies which    |
| ulator      | es         |                     | feature            |
|             | (Default), |                     | calculation tool   |
|             | CF_pyradio |                     | should be used     |
|             | mics       |                     |                    |
+-------------+------------+---------------------+--------------------+
| Preprocessi | PreProcess | WORC                | Specifies which    |
| ng          |            |                     | tool will be used  |
|             |            |                     | for image          |
|             |            |                     | preprocessing      |
+-------------+------------+---------------------+--------------------+
| Registratio | Elastix    | WORC                | Specifies which    |
| nNode       |            |                     | tool will be used  |
|             |            |                     | for image          |
|             |            |                     | registration       |
+-------------+------------+---------------------+--------------------+
| Transformat | Transformi | WORC                | Specifies which    |
| ionNode     | x          |                     | tool will be used  |
|             |            |                     | for applying image |
|             |            |                     | transformations    |
+-------------+------------+---------------------+--------------------+

PREDICTGeneral
--------------

These fields contain general settings for when using PREDICT.

+-------------------+------------------+------------------------------+
| Field             | Possible values  | Explanation                  |
+===================+==================+==============================+
| Joblib_ncores     | integer, default | Number of cores to be used   |
|                   | 4                | by joblib for multicore      |
|                   |                  | processing.                  |
+-------------------+------------------+------------------------------+
| Joblib_backend    | threading or     | Type of backend to be used   |
|                   | multiprocessing  | by joblib for multicore      |
|                   | (default)        | processing.                  |
+-------------------+------------------+------------------------------+
| tempsave          | boolean, default | Determines whether after     |
|                   | False          | every cross validation       |
|                   |                  | iteration the result will be |
|                   |                  | saved, in addition to the    |
|                   |                  | result after all iterations. |
|                   |                  | Especially useful for        |
|                   |                  | debugging.                   |
+-------------------+------------------+------------------------------+

For more info on the Joblib settings, which are used in the Joblib
Parallel function, see
`here <https://pythonhosted.org/joblib/parallel.html>`__. When you run
WORC on a cluster with nodes supporting only a single core to be used
per node, e.g. the BIGR cluster, use only 1 core and threading as a
backend.

Segmentix
---------

These fields are only important if you specified using the segmentix
tool in the general configuration.

+-------------------+------------------+-------------------------------+
| Field             | Possible values  | Explanation                   |
+===================+==================+===============================+
| mask              | subtract,        | If a mask is supplied, should |
|                   | multiply         | the mask be subtracted from   |
|                   |                  | the contour or multiplied     |
+-------------------+------------------+-------------------------------+
| segtype           | None, Ring       | If Ring, then a ring around   |
|                   |                  | the segmentation will be used |
|                   |                  | as contour.                   |
+-------------------+------------------+-------------------------------+
| segradius         | Integer          | Define the radius of the ring |
|                   |                  | used if segtype is Ring       |
+-------------------+------------------+-------------------------------+
| N_blobs           | Integer          | How many of the largest blobs |
|                   |                  | are extracted from the        |
|                   |                  | segmentation. If None, no     |
|                   |                  | blob extraction is used       |
+-------------------+------------------+-------------------------------+
| fillholes         | Boolean, default | Determines whether hole       |
|                   | False            | filling will be used.         |
+-------------------+------------------+-------------------------------+

Preprocessing
-------------

The preprocessing node acts before the feature extraction on the image.
Currently, only normalization is included: hence the dictionary name is
*Normalize*. Additionally, scans with image type CT (see later in the
tutorial) provided as DICOM are scaled to Hounsfield Units.

+-------------------+------------------+-------------------------------+
| Field             | Possible values  | Explanation                   |
+===================+==================+===============================+
| ROI               | True, False,     | If a mask is supplied and     |
|                   | Full (Default)   | this is set to True,          |
|                   |                  | normalize image based on      |
|                   |                  | supplied ROI. Otherwise, the  |
|                   |                  | full image is used for        |
|                   |                  | normalization using the       |
|                   |                  | SimpleITK Normalize function. |
|                   |                  | Lastly, setting this to False |
|                   |                  | will result in no             |
|                   |                  | normalization being applied.  |
+-------------------+------------------+-------------------------------+
| Method            | z_score          | Method used for normalization |
|                   | (Default),       | if ROI is supplied.           |
|                   | minmed           | Currently, z-scoring or using |
|                   |                  | the minimum and median of the |
|                   |                  | ROI can be used.              |
+-------------------+------------------+-------------------------------+

Imagefeatures
-------------

If using the PREDICT toolbox, you can specify some settings for the
feature computation here. Also, you can select if the certain features
are computed or not.

+----------------------------+--------------------+--------------------+
| Field                      | Possible values    | Explanation        |
+============================+====================+====================+
| orientation                | True (Default),    | Determine whether  |
|                            | False              | orientation        |
|                            |                    | features are       |
|                            |                    | computed or not.   |
+----------------------------+--------------------+--------------------+
| texture                    | all (Default),     | Determine whether  |
|                            | None, Gabor, LBP,  | all, none or only  |
|                            | GLCM, GLRLM,       | a select group of  |
|                            | GLSZM, NGTDM       | the texture        |
|                            |                    | features are       |
|                            |                    | computed.          |
+----------------------------+--------------------+--------------------+
| coliage                    | True, False        | Determine whether  |
|                            | (Default)          | coliage features   |
|                            |                    | are computed or    |
|                            |                    | not.               |
+----------------------------+--------------------+--------------------+
| vessel                     | True, False        | Determine whether  |
|                            | (Default)          | vessel features    |
|                            |                    | are computed or    |
|                            |                    | not.               |
+----------------------------+--------------------+--------------------+
| log                        | True, False        | Determine whether  |
|                            | (Default)          | LoG features are   |
|                            |                    | computed or not.   |
+----------------------------+--------------------+--------------------+
| phase                      | True, False        | Determine whether  |
|                            | (Default)          | local phase        |
|                            |                    | features are       |
|                            |                    | computed or not.   |
+----------------------------+--------------------+--------------------+
| image_type                 | CT (Default), MR,  | Modality of images |
|                            | DTI, DTI_post      | supplied.          |
|                            |                    | Determines how the |
|                            |                    | image is loaded.   |
+----------------------------+--------------------+--------------------+
| gabor_frequencies          | float(s), default  | Frequencies of     |
|                            | '0.05, 0.2, 0.5'   | Gabor filters      |
|                            |                    | used: can be a     |
|                            |                    | single float or a  |
|                            |                    | list.              |
+----------------------------+--------------------+--------------------+
| gabor_angles               | integer(s),        | Angles of Gabor    |
|                            | default 0, 45,    | filters in         |
|                            | 90, 135           | degrees: can be a  |
|                            |                    | single integer or  |
|                            |                    | a list.            |
+----------------------------+--------------------+--------------------+
| GLCM_angles                | floats(s), default | Angles used in     |
|                            | 0, 0.79, 1.57,    | GLCM computation   |
|                            | 2.36              | in radians: can be |
|                            |                    | a single float or  |
|                            |                    | a list.            |
+----------------------------+--------------------+--------------------+
| GLCM_levels                | integer, default   | Number of          |
|                            | 16                 | grayscale levels   |
|                            |                    | used in            |
|                            |                    | discretization     |
|                            |                    | before GLCM        |
|                            |                    | computation.       |
+----------------------------+--------------------+--------------------+
| GLCM_distances             | integer(s),        | Distance(s) used   |
|                            | default 1, 3     | in GLCM            |
|                            |                    | computation in     |
|                            |                    | pixels: can be a   |
|                            |                    | single integer or  |
|                            |                    | a list.            |
+----------------------------+--------------------+--------------------+
| LBP_radius                 | integer(s),        | Radii used for LBP |
|                            | default 3, 8, 15 | computation: can   |
|                            |                    | be a single        |
|                            |                    | integer or a list. |
+----------------------------+--------------------+--------------------+
| LBP_npoints                | integer(s),        | Number(s) of       |
|                            | default 12, 24,   | points used in LBP |
|                            | 36                | computation: can   |
|                            |                    | be a single        |
|                            |                    | integer or a list. |
+----------------------------+--------------------+--------------------+
| phase_minwavelength        | integer, default   | Minimal wavelength |
|                            | 3                | in pixels used for |
|                            |                    | phase features.    |
+----------------------------+--------------------+--------------------+
| phase_nscale               | integer, default   | Number of scales   |
|                            | 5                | used in phase      |
|                            |                    | feature            |
|                            |                    | computation.       |
+----------------------------+--------------------+--------------------+
| log_sigma                  | integer(s),        | Standard           |
|                            | default 1, 5, 10 | deviation(s) in    |
|                            |                    | pixels used in log |
|                            |                    | feature            |
|                            |                    | computation: can   |
|                            |                    | be a single        |
|                            |                    | integer or a list. |
+----------------------------+--------------------+--------------------+
| vessel_scale_range         | two integers,      | Scale in pixels    |
|                            | default 1, 10    | used for Frangi    |
|                            |                    | vessel filter.     |
|                            |                    | Given as a minimum |
|                            |                    | and a maximum.     |
+----------------------------+--------------------+--------------------+
| vessel_scale_step          | integer, default   | Step size used to  |
|                            | 2                | go from minimum to |
|                            |                    | maximum scale on   |
|                            |                    | Frangi vessel      |
|                            |                    | filter.            |
+----------------------------+--------------------+--------------------+
| vessel_radius              | integer, default   | Radius to          |
|                            | 5                | determine boundary |
|                            |                    | of between inner   |
|                            |                    | part and edge in   |
|                            |                    | Frangi vessel      |
|                            |                    | filter.            |
+----------------------------+--------------------+--------------------+

Featsel
-------

When using the PREDICT toolbox for classification, these settings can be
used for feature selection methods. Note that these settings are
actually used in the hyperparameter optimization. Hence you can provide
multiple values per field, of which random samples will be drawn of
which finally the best setting in combination with the other
hyperparameters is selected. Again, these should be formatted as string
containing the actual values, e.g. value1, value2.

+----------------------------+--------------------+--------------------+
| Field                      | Possible values    | Explanation        |
+============================+====================+====================+
| Variance                   | True (Default),    | Exclude features   |
|                            | False              | which have a       |
|                            |                    | variance < 0.01    |
+----------------------------+--------------------+--------------------+
| GroupwiseSearch            | True (Default),    | Randomly select    |
|                            | False              | which feature      |
|                            |                    | groups to use.     |
|                            |                    | Parameters         |
|                            |                    | determined by the  |
|                            |                    | SelectFeatGroup    |
|                            |                    | config part, see   |
|                            |                    | below              |
+----------------------------+--------------------+--------------------+
| SelectFromModel            | True, False        | Select features by |
|                            | (Default)          | first training a   |
|                            |                    | LASSO model. The   |
|                            |                    | alpha for the      |
|                            |                    | LASSO model is     |
|                            |                    | randomly           |
|                            |                    | generated.         |
+----------------------------+--------------------+--------------------+
| UsePCA                     | True, False        | Use Principle      |
|                            | (Default)          | Component Analysis |
|                            |                    | (PCA) to select    |
|                            |                    | features.          |
+----------------------------+--------------------+--------------------+
| PCAType                    | 95variance         | Method to select   |
|                            | (Default),         | number of          |
|                            | integer(s)         | components using   |
|                            |                    | PCA: Either the    |
|                            |                    | number of          |
|                            |                    | components that    |
|                            |                    | explains 95% of    |
|                            |                    | the variance, or   |
|                            |                    | use a fixed number |
|                            |                    | of components.     |
+----------------------------+--------------------+--------------------+
| StatisticalTestUse         | True, False        | Use statistical    |
|                            | (Default)          | test to select     |
|                            |                    | features.          |
+----------------------------+--------------------+--------------------+
| StatisticalTestMetric      | ttest, Welch,      | Define the type of |
|                            | Wilcoxon,          | statistical test   |
|                            | MannWhitneyU,      | to be used.        |
|                            | default all        |                    |
+----------------------------+--------------------+--------------------+
| StatisticalTestThreshold   | two floats,        | Specify a          |
|                            | default 0.02,     | threshold for the  |
|                            | 0.2               | p-value threshold  |
|                            |                    | used in the        |
|                            |                    | statistical test   |
|                            |                    | to select          |
|                            |                    | features. The      |
|                            |                    | first element      |
|                            |                    | defines the lower  |
|                            |                    | boundary, the      |
|                            |                    | other the upper    |
|                            |                    | boundary. Random   |
|                            |                    | sampling will      |
|                            |                    | occur between the  |
|                            |                    | boundaries.        |
+----------------------------+--------------------+--------------------+
| ReliefUse                  | Boolean(s),        | Use Relief to      |
|                            | default False      | select features.   |
+----------------------------+--------------------+--------------------+
| ReliefNN                   | Integer(s),        | Min and max of     |
|                            | default 2, 4       | number of nearest  |
|                            |                    | neighbors search   |
|                            |                    | range in Relief.   |
+----------------------------+--------------------+--------------------+
| ReliefSampleSize           | Integer(s),        | Min and max of     |
|                            | default 1, 1       | sample size search |
|                            |                    | range in Relief.   |
+----------------------------+--------------------+--------------------+
| ReliefDistanceP            | Integer(s),        | Min and max of     |
|                            | default 1, 3       | positive distance  |
|                            |                    | search range in    |
|                            |                    | Relief.            |
+----------------------------+--------------------+--------------------+
| ReliefNumFeatures          | Integer(s),        | Min and max of     |
|                            | default 25, 200    | number of features |
|                            |                    | that is selected   |
|                            |                    | search range in    |
|                            |                    | Relief.            |
+----------------------------+--------------------+--------------------+

SelectFeatGroup
---------------

If the PREDICT feature computation and classification tools are used,
then you can do a gridsearch among the various feature groups for the
optimal combination. If you do not want this, set all fields to a single
value.

+-------------------------+----------------------------+
| Field                   | Possible values            |
+=========================+============================+
| shape_features          | True, False, default both  |
+-------------------------+----------------------------+
| histogram_features      | True, False , default both |
+-------------------------+----------------------------+
| orientation_features    | True, False , default both |
+-------------------------+----------------------------+
| texture_Gabor_features  | True, False, default both  |
+-------------------------+----------------------------+
| texture_GLCM_features   | True, False, default both  |
+-------------------------+----------------------------+
| texture_GLCMMS_features | True, False, default both  |
+-------------------------+----------------------------+
| texture_GLRLM_features  | True, False, default both  |
+-------------------------+----------------------------+
| texture_GLSZM_features  | True, False, default both  |
+-------------------------+----------------------------+
| texture_NGTDM_features  | True, False, default both  |
+-------------------------+----------------------------+
| texture_LBP_features    | True, False, default both  |
+-------------------------+----------------------------+
| patient_features        | True, False (Default)      |
+-------------------------+----------------------------+
| semantic_features       | True, False (Default)      |
+-------------------------+----------------------------+
| coliage_features        | True, False (Default)      |
+-------------------------+----------------------------+
| log_features            | True, False (Default)      |
+-------------------------+----------------------------+
| vessel_features         | True, False (Default)      |
+-------------------------+----------------------------+
| phase_features          | True, False (Default)      |
+-------------------------+----------------------------+

Previously, there was a single parameter for the texture features,
selecting all, none or a single group. This is still supported, but not
recommended, and looks as follows:

+-----------------------------------------+----------------------------+
| Field                                   | Possible values            |
+=========================================+============================+
| texture_features                        | True, False (Default),     |
|                                         | Gabor, LBP, GLCM, GLRLM,   |
|                                         | GLSZM, NGTDM               |
+-----------------------------------------+----------------------------+

Imputation (WIP)
----------------

When using the PREDICT toolbox for classification, these settings are
used for feature imputation.Note that these settings are actually used
in the hyperparameter optimization. Hence you can provide multiple
values per field, of which random samples will be drawn of which finally
the best setting in combination with the other hyperparameters is
selected.

+----------------------------+--------------------+--------------------+
| Field                      | Possible values    | Explanation        |
+============================+====================+====================+
| use                        | True, False        | Whether to use     |
|                            | (Default)          | imputation or not. |
|                            |                    | If not, all NaN    |
|                            |                    | features will be   |
|                            |                    | set to zero.       |
+----------------------------+--------------------+--------------------+
| strategy                   | mean (Default),    | Method used for    |
|                            | median,            | feature            |
|                            | most_frequent, knn | imputation.        |
+----------------------------+--------------------+--------------------+
| n_neighbors                | integer(s),        | When using         |
|                            | default 5        | k-Nearest          |
|                            |                    | Neighbors (kNN)    |
|                            |                    | for feature        |
|                            |                    | imputation,        |
|                            |                    | determines the     |
|                            |                    | number of          |
|                            |                    | neighbors used for |
|                            |                    | imputation. Can be |
|                            |                    | a single integer   |
|                            |                    | or a list.         |
+----------------------------+--------------------+--------------------+

Classification
--------------

When using the PREDICT toolbox for classification, you can specify the
following settings. Almost all of these are used in CASH. Most of the
classifiers are implemented using sklearn; hence descriptions of the
hyperparameters can also be found there.

+----------------------------+--------------------+--------------------+
| Field                      | Possible values    | Explanation        |
+============================+====================+====================+
| fastr                      | True, False        | Use fastr for the  |
|                            | (Default)          | optimization       |
|                            |                    | gridsearch         |
|                            |                    | (recommended on    |
|                            |                    | clusters) or if    |
|                            |                    | set to False ,     |
|                            |                    | joblib             |
|                            |                    | (recommended for   |
|                            |                    | PCs, default).     |
+----------------------------+--------------------+--------------------+
| fastr_plugin               | Name of `fastr     | Name of execution  |
|                            | execution          | plugin to be used. |
|                            | plugin <http://fas | Default use the    |
|                            | tr.readthedocs.io/ | same as the        |
|                            | en/stable/fastr.re | self.fastr_plugin  |
|                            | ference.html#execu | for the WORC       |
|                            | tionplugin-referen | object.            |
|                            | ce>`__             |                    |
+----------------------------+--------------------+--------------------+
| classifiers                | SVM (Default),     | Select the         |
|                            | SVR, SGD, SGDR,    | estimator(s) to    |
|                            | RF, LDA, QDA,      | use. Most are all  |
|                            | ComplementND,      | implemented from   |
|                            | GaussianNB, LR,    | sklearn, so see    |
|                            | RFR, Lasso,        | sklearn for more   |
|                            | ElasticNet         | details. Included  |
|                            |                    | in CASH.           |
+----------------------------+--------------------+--------------------+
| max_iter                   | integer, default   | Number of          |
|                            | 100.000          | iterations to use  |
|                            |                    | in training an     |
|                            |                    | estimator. Only    |
|                            |                    | for specific       |
|                            |                    | estimators, see    |
|                            |                    | sklearn.           |
+----------------------------+--------------------+--------------------+
| SVMKernel                  | polynomial         | When using a SVM,  |
|                            | (Default) , rbf,   | specify the kernel |
|                            | linear             | type. Included in  |
|                            |                    | CASH.              |
+----------------------------+--------------------+--------------------+
| SVMC                       | two integers,      | Range of the SVM   |
|                            | default 0, 6     | slack parameter.   |
|                            |                    | We sample on a     |
|                            |                    | uniform log scale: |
|                            |                    | the parameters     |
|                            |                    | specify the range  |
|                            |                    | of the exponent    |
|                            |                    | (a, a + b).        |
|                            |                    | Included in CASH.  |
+----------------------------+--------------------+--------------------+
| SVMdegree                  | two integers,      | Range of the SVM   |
|                            | default 1, 6     | polynomial degree  |
|                            |                    | when using a       |
|                            |                    | polynomial kernel. |
|                            |                    | We sample on a     |
|                            |                    | uniform scale: the |
|                            |                    | parameters specify |
|                            |                    | the range (a, a +  |
|                            |                    | b). Included in    |
|                            |                    | CASH.              |
+----------------------------+--------------------+--------------------+
| SVMcoef0                   | two integers,      | Range of SVM       |
|                            | default 0, 1     | homogeneity        |
|                            |                    | parameter. We      |
|                            |                    | sample on a        |
|                            |                    | uniform scale: the |
|                            |                    | parameters specify |
|                            |                    | the range (a, a +  |
|                            |                    | b). Included in    |
|                            |                    | CASH.              |
+----------------------------+--------------------+--------------------+
| SVMgamma                   | two integers,      | Range of the SVM   |
|                            | default -5, 5    | gamma parameter.   |
|                            |                    | We sample on a     |
|                            |                    | uniform log scale: |
|                            |                    | the parameters     |
|                            |                    | specify the range  |
|                            |                    | of the exponent    |
|                            |                    | (a, a + b).        |
|                            |                    | Included in CASH.  |
+----------------------------+--------------------+--------------------+
| RFn_estimators             | two integers,      | Range of number of |
|                            | default 10, 190  | trees in a RF. We  |
|                            |                    | sample on a        |
|                            |                    | uniform scale: the |
|                            |                    | parameters specify |
|                            |                    | the range (a, a +  |
|                            |                    | b). Included in    |
|                            |                    | CASH.              |
+----------------------------+--------------------+--------------------+
| RFmin_samples_split        | two integers,      | Range of minimum   |
|                            | default 2, 3     | number of samples  |
|                            |                    | required to split  |
|                            |                    | a branch in a RF.  |
|                            |                    | We sample on a     |
|                            |                    | uniform scale: the |
|                            |                    | parameters specify |
|                            |                    | the range (a, a +  |
|                            |                    | b). Included in    |
|                            |                    | CASH.              |
+----------------------------+--------------------+--------------------+
| RFmax_depth                | two integers,      | Range of maximum   |
|                            | default 5, 5     | depth of a RF. We  |
|                            |                    | sample on a        |
|                            |                    | uniform scale: the |
|                            |                    | parameters specify |
|                            |                    | the range (a, a +  |
|                            |                    | b). Included in    |
|                            |                    | CASH.              |
+----------------------------+--------------------+--------------------+
| LRpenalty                  | l2, l1             | Penalty term used  |
|                            |                    | in LR. Included in |
|                            |                    | CASH.              |
+----------------------------+--------------------+--------------------+
| LRC                        | two floats,        | Range of           |
|                            | default 0.01,     | regularization     |
|                            | 0.99              | strength in LR. We |
|                            |                    | sample on a        |
|                            |                    | uniform scale: the |
|                            |                    | parameters specify |
|                            |                    | the range (a, a +  |
|                            |                    | b). Included in    |
|                            |                    | CASH.              |
+----------------------------+--------------------+--------------------+
| LDA_solver                 | svd, lsqr, eigen   | Solver used in     |
|                            |                    | LDA. Included in   |
|                            |                    | CASH.              |
+----------------------------+--------------------+--------------------+
| LDA_shrinkage              | two integers,      | Range of the LDA   |
|                            | default -5, 5    | shrinkage          |
|                            |                    | parameter. We      |
|                            |                    | sample on a        |
|                            |                    | uniform log scale: |
|                            |                    | the parameters     |
|                            |                    | specify the range  |
|                            |                    | of the exponent    |
|                            |                    | (a, a + b).        |
|                            |                    | Included in CASH.  |
+----------------------------+--------------------+--------------------+
| QDA_reg_param              | two integers,      | Range of the QDA   |
|                            | default -5, 5    | regularization     |
|                            |                    | parameter. We      |
|                            |                    | sample on a        |
|                            |                    | uniform log scale: |
|                            |                    | the parameters     |
|                            |                    | specify the range  |
|                            |                    | of the exponent    |
|                            |                    | (a, a + b).        |
|                            |                    | Included in CASH.  |
+----------------------------+--------------------+--------------------+
| ElasticNet_alpha           | two integers,      | Range of the       |
|                            | default -5, 5    | ElasticNet penalty |
|                            |                    | parameter. We      |
|                            |                    | sample on a        |
|                            |                    | uniform log scale: |
|                            |                    | the parameters     |
|                            |                    | specify the range  |
|                            |                    | of the exponent    |
|                            |                    | (a, a + b).        |
|                            |                    | Included in CASH.  |
+----------------------------+--------------------+--------------------+
| ElasticNet_l1_ratio        | two floats,        | Range of l1 ratio  |
|                            | default 0.00,     | in LR. We sample   |
|                            | 1.00              | on a uniform       |
|                            |                    | scale: the         |
|                            |                    | parameters specify |
|                            |                    | the range (a, a +  |
|                            |                    | b). Included in    |
|                            |                    | CASH.              |
+----------------------------+--------------------+--------------------+
| SGD_alpha                  | two integers,      | Range of the SGD   |
|                            | default -5, 5    | penalty parameter. |
|                            |                    | We sample on a     |
|                            |                    | uniform log scale: |
|                            |                    | the parameters     |
|                            |                    | specify the range  |
|                            |                    | of the exponent    |
|                            |                    | (a, a + b).        |
|                            |                    | Included in CASH.  |
+----------------------------+--------------------+--------------------+
| SGD_l1_ratio               | two floats,        | Range of l1 ratio  |
|                            | default 0.00,     | in SGD. We sample  |
|                            | 1.00              | on a uniform       |
|                            |                    | scale: the         |
|                            |                    | parameters specify |
|                            |                    | the range (a, a +  |
|                            |                    | b). Included in    |
|                            |                    | CASH.              |
+----------------------------+--------------------+--------------------+
| SGD_loss                   | hinge,             | Loss function of   |
|                            | squared_hinge,     | SGD. Included in   |
|                            | modified_huber     | CASH.              |
+----------------------------+--------------------+--------------------+
| SGD_penalty                | none, l2, l1       | Penalty term in    |
|                            |                    | SGD. Included in   |
|                            |                    | CASH.              |
+----------------------------+--------------------+--------------------+
| CNB_alpha                  | two integers,      | Regularization     |
|                            | default 0, 1     | strenght in        |
|                            |                    | ComplementNB. We   |
|                            |                    | sample on a        |
|                            |                    | uniform scale: the |
|                            |                    | parameters specify |
|                            |                    | the range (a, a +  |
|                            |                    | b). Included in    |
|                            |                    | CASH.              |
+----------------------------+--------------------+--------------------+

CrossValidation
---------------

When using the PREDICT toolbox for classification and you specified
using cross validation, specify the following settings.

+----------------------------+--------------------+--------------------+
| Field                      | Possible values    | Explanation        |
+============================+====================+====================+
| N_iterations               | integer, default   | Number of times    |
|                            | 50               | the data is split  |
|                            |                    | in training and    |
|                            |                    | test.              |
+----------------------------+--------------------+--------------------+
| test_size                  | float between 0 -  | The percentage of  |
|                            | 1, default 0.2   | data to be used    |
|                            |                    | for testing.       |
+----------------------------+--------------------+--------------------+

Genetics
--------

When using the PREDICT toolbox for classification, you have to set the
label used for classification.

+----------------------------+--------------------+--------------------+
| Field                      | Possible values    | Explanation        |
+============================+====================+====================+
| label_names                | string             | The labels used    |
|                            |                    | from your          |
|                            |                    | segmentation file  |
|                            |                    | for classification |
+----------------------------+--------------------+--------------------+
| modus                      | singlelabel        | Determine whether  |
|                            | (default) or       | multilabel or      |
|                            | multilabel         | singlelabel        |
|                            |                    | classification     |
|                            |                    | will be performed. |
+----------------------------+--------------------+--------------------+

This part is really important, as it should match your label file.
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

::

   config['Genetics']['label_names'] = 'Label1'

If you want to first train a classifier on Label1 and then Label2,
set: config[Genetics][label_names] = Label1, Label2

**Note: this config part also contains the url and projectID fields,
which are currently WIP and should be left untouched.**

Hyperoptimization
-----------------

When using the PREDICT toolbox for classification, you have to supply
your hyperparameter optimization procedure here.

+----------------------------+--------------------+--------------------+
| Field                      | Possible values    | Explanation        |
+============================+====================+====================+
| scoring_method             | See                | Specify the        |
|                            | http://scikit-lear | optimization       |
|                            | n.org/stable/modul | metric for your    |
|                            | es/model_evaluatio | hyperparameter     |
|                            | n.html,            | search.            |
|                            | default            |                    |
|                            | f1_weighted      |                    |
+----------------------------+--------------------+--------------------+
| test_size                  | float, default     | Size of test set   |
|                            | 0.15             | in the             |
|                            |                    | hyperoptimization  |
|                            |                    | cross validation,  |
|                            |                    | given as a         |
|                            |                    | percentage of the  |
|                            |                    | whole dataset.     |
+----------------------------+--------------------+--------------------+
| N_iterations               | integer, default   | Number of          |
|                            | 10000            | iterations used in |
|                            |                    | the hyperparameter |
|                            |                    | optimization. This |
|                            |                    | corresponds to the |
|                            |                    | number of samples  |
|                            |                    | drawn from the     |
|                            |                    | parameter grid.    |
+----------------------------+--------------------+--------------------+
| n_jobspercore              | integer, default   | Number of jobs     |
|                            | 2000             | assigned to a      |
|                            |                    | single core. Only  |
|                            |                    | used if fastr is   |
|                            |                    | set to true in the |
|                            |                    | classfication.     |
+----------------------------+--------------------+--------------------+

FeatureScaling
--------------

Determines which method is applied to scale each feature.

+----------------------------+--------------------+--------------------+
| Field                      | Possible values    | Explanation        |
+============================+====================+====================+
| scale_features             | boolean, default   | Determine if       |
|                            | True             | feature scaling is |
|                            |                    | used.              |
+----------------------------+--------------------+--------------------+
| scaling_method             | z_score (Default)  | Determine the      |
|                            | , minmax           | scaling method.    |
+----------------------------+--------------------+--------------------+

SampleProcessing
----------------

Before performing the hyperoptimization, you can use SMOTE: Synthetic
Minority Over-sampling Technique to oversample your data.

+----------------------------+--------------------+--------------------+
| Field                      | Possible values    | Explanation        |
+============================+====================+====================+
| SMOTE                      | boolean, default   | Determine if to    |
|                            | True             | use SMOTE          |
+----------------------------+--------------------+--------------------+
| SMOTE_ratio                | two integers,      | Determine the      |
|                            | default 1, 0     | ratio of           |
|                            |                    | oversampling. If   |
|                            |                    | 1, the minority    |
|                            |                    | class will be      |
|                            |                    | oversampled to the |
|                            |                    | same size as the   |
|                            |                    | majority class. We |
|                            |                    | sample on a        |
|                            |                    | uniform scale: the |
|                            |                    | parameters specify |
|                            |                    | the range (a, a +  |
|                            |                    | b). Included in    |
|                            |                    | CASH.              |
+----------------------------+--------------------+--------------------+
| SMOTE_neighbors            | two integers,      | Number of          |
|                            | default 5, 15    | neighbors used in  |
|                            |                    | SMOTE. This should |
|                            |                    | be much smaller    |
|                            |                    | than the number of |
|                            |                    | objects/patients   |
|                            |                    | you supply. We     |
|                            |                    | sample on a        |
|                            |                    | uniform scale: the |
|                            |                    | parameters specify |
|                            |                    | the range (a, a +  |
|                            |                    | b). Included in    |
|                            |                    | CASH.              |
+----------------------------+--------------------+--------------------+
| Oversampling               | Boolean, default   | Determine whether  |
|                            | False              | full random        |
|                            |                    | oversampling will  |
|                            |                    | be used or not.    |
+----------------------------+--------------------+--------------------+

Ensemble
--------

WORC supports ensembling of workflows. This is not a default approach in
radiomics, hence the default is to not use it and select only the best
performing workflow.

+-------+---------------------+
| Field | Possible values     |
+=======+=====================+
| Use   | False or an integer |
+-------+---------------------+

FASTR_bugs
----------

Currently, when using XNAT as a source, FASTR can only retrieve DICOM
directories. We made a workaround for this for the images and
segmentations, but this only works if all your files have the same name
and extension. These are provided in this configuration part.

+---------------+--------------------------------------+
| Field         | Possible values                      |
+===============+======================================+
| images        | FILENAME.EXT, default image.nii.gz |
+---------------+--------------------------------------+
| segmentations | FILENAME.EXT, default mask.nii.gz  |
+---------------+--------------------------------------+
