..  additonalfunctionality-chapter:

Additional functionality
========================

When using ``SimpleWORC``, or WORC with similar simple configuration settings, you can
already benefit from the main functionality of WORC, i.e. the automatic algorithm
optimization. However, several additional functionalities are provided, which are discussed in
this chapter.

For a description of the radiomics features, please see
:ref:`the radiomics features chapter <features-chapter>`. For a description of
the data mining components, see
:ref:`the data mining chapter <datamining-chapter>`. All other components
are discussed here.

For a comprehensive overview of all functions and parameters, please look at
:ref:`the config chapter <config-chapter>`.


Image Preprocessing
--------------------
Preprocessing of the image, and accordingly the mask, is done in respectively
the :py:mod:`WORC.processing.preprocessing` and the
:py:mod:`WORC.processing.segmentix` scripts. Options for preprocessing
the image include, in the following order:

1. N4 Bias field correction, see also https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html.
2. Checking and optionally correcting the spacing if it's 1x1x1 and the DICOM metadata says otherwise.
3. Clipping of the image intensities above and below a certain value.
4. Normalization, see :py:mod:`WORC.processing.preprocessing.normalize_image` for all options.
5. Transposing the image to another ''main'' orientation, e.g. axial.
6. Resampling the image to a different spacing.

Options for preprocessing the segmentation include:

1. Hole filling. Many feature computations cannot deal with holes.
2. Removing small objects. Many feature computations cannot deal with multiple
  objects in a single segmentation.
3. Extracing the largest blob. Many feature computations cannot deal with
  multiple objects in a single segmentation.
4. Instead of using the full segmentation, extracting a ring around the border
  of the image to compute the features on. Ring captures both the inner and
  outer border.
5. Dilating the contour.
6. Masking the contour with another contour.
7. When assuming the same image and metadata, copy the metadata of the image
  to the segmentation.
8. Checking and optionally correcting the spacing if it's 1x1x1 and the
  DICOM metadata says otherwise. Same as image preprocessing step 2.
9. Transposing the segmentation to another ''main'' orientation, e.g. axial.
  Same as image preprocessing step 5.
10. Resampling the segmentation **and the segmentation** to a different spacing.
  Same as image preprocessing step 10.

Image Registration
-------------------

Documentation WIP.

ComBat
--------

ComBat feature harmonization is embedded in WORC. A wrapper, compatible with
the other tools provided by WORC, is included in the installation. We have included
wrapping around the Matlab and Python code (neurocombat) from the
original `ComBat code <https://github.com/Jfortin1/ComBatHarmonization/>`_. We recommend
to use the Python code by default.

When using ComBat, the following configurations should be done:

1. Set ``config['General']['ComBat']`` to ``'True'``.
2. To change the ComBat parameters (i.e. which batch and moderation variable to use),
   change the relevant config fields, see the :ref:`Config chapter <config-chapter>`.
3. WORC extracts the batch and moderation variables from the label file which you also
   use to give WORC the actual label you want to predict. The same format therefore applies, see
   the :ref:`User manual <usermanual-chapter>` for more details..

.. note:: In line with current literature, ComBat is applied once on the full dataset
    straight after the feature extraction, thus before the actual hyperoptimization.
    Hence, to avoid serious overfitting, we advice to **NEVER** use the variable
    you are trying to predict as the moderation variable.

ICC
----
Documentation WIP.

Additional classifiers
-----------------------
When using the XGDBoost classifiers or regressors, install ``xgdboost``,
which can be done using ``pip``, see https://xgboost.readthedocs.io/en/latest/python/python_intro.html.
``WORC`` makes use of the scikit-learn API.


SMAC
-----
.. note:: In order to use SMAC, first run ``git install [my smac fork]``.
The SMAC algorithm, using Bayesian optimization, can be used for the hyperparameter optimization by
setting the ``config['SMAC']['use']`` parameter to ``'True'``. For details on which SMAC parameters
can be modified, see :ref:`Config chapter <config-chapter>`.

The core functionality of SMAC within WORC is implemented in
:py:mod:`WORC.resources.fastr_tools.worc.bin.smac_tool`. The configuration space of SMAC is specified
in :py:mod:`WORC.classification.smac`, which is also where new methods can be added to the search space.

There is additional output when using SMAC. The final output file ``smac_results_all_0.json``
is added along with the regular performance files. It contains information on the optimization procedure for
each cross-validation split, with statistics on the performance and all intermediate best found configurations.
The end of the file contains a summary of the average statistics over all train-test cross-validations.


