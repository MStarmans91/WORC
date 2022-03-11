.. _additonalfunctionality-chapter:

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

Feature scaling
--------------------
The default method for feature scaling in ``WORC`` is a robust version
of z-scoring. Additional options include:

1. regular z-scoring
2. MinMax scaling, i.e., scaling to a range between 0 and 1
3. Scaling by centering using the median and IQR
4. A combination of z-scoring with a logarithmic transform and a correction
   term to better cope with outliers and non-normally distributed features [CIT1]_.

Image Registration
-------------------

When using multiple modalities or sequences, and there is only a segmentation
on a single image, image registration is applied to spatially align all
sequences and warp the segmentation to the other images through
``elastix`` [CIT2]_. Usage of ``elastix`` is automatically included in ``WORC``
when only a single segmentation and multiple modalities are supplied.
The image on which the segmentation is provided is used as the moving image,
the others as the fixed image, as the segmentations will be moved from the
segmented image to the others.

Registration is by default performed using a
rigid transformation model, based on a mutual information using the adaptive
stochastic gradient descent optimizer. Manual
overrides of these defaults are included in the ``WORC`` configuration.


When using Elastix, parameter files have to be provided in the
``network.Elastix_Para`` object, e.g.

.. code-block:: python

   network.Elastix_Para = [['Parameters_Rigid.txt', 'Parameters_BSpline.txt']]

The outer list defines the parameter files used per modality. If only one
element is provided, the same will be applied for all modalities. Each element
of the list should be a list of its own, including the filenames
of ``elastix``. In the example, we provided two files, resulting
in first a rigid registration being performed, followed by a bspline registration.
Examples of ``elastix`` parameter files can be found at https://github.com/SuperElastix/ElastixModelZoo/tree/master/models/default

ComBat
--------

Commonly, radiomics studies include multicenter data,
resulting in heterogeneity in the acquisition protocols. As radiomics features
are generally sensitive to these variations, this limits the repeatability and
reproducibility. To compensate for the differences in acquisition, feature
harmonization techniques may be used, one of the most frequently used
is ComBat. In ComBat, feature distributions are harmonized for variations in
the imaging acquisition, e.g. due to differences in hospitals, manufacturers,
or acquisition parameters. The dataset is divided in groups based on these
differences, and a correction of the error caused by these differences
is estimated using empirical Bayes.

ComBat is included in ``WORC`` and can be turned on in the configuration,
including options to use empirical Bayes or not, a parametric or
non-parametric approach, and a moderation variable.

ComBat feature harmonization is embedded in WORC. A wrapper around the
original `ComBat code <https://github.com/Jfortin1/ComBatHarmonization/>`_,
compatible with the other tools provided by ``WORC``, is included in the
``WORC`` installation.

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

Bayesian optimization with SMAC instead of random search
--------------------------------------------------------
.. note:: The SMAC algorithm only works on Linux, because of its random forest surrogate model
    implementation. Make sure to use ``swig3.0``. To circumvent ``pyrfr`` issues
    with SMAC, we use a custom fork of the original SMAC package that needs to be installed separately.

Steps to take in order to use SMAC within WORC:

1. ``sudo apt-get remove swig``
2. ``sudo apt-get install swig3.0``
3. ``sudo ln -s /usr/bin/swig3.0 /usr/bin/swig``
4. ``pip install pyrfr==0.8.0``
5. ``pip install git+https://github.com/mitchelldeen/SMAC3.git``

The SMAC algorithm, using Bayesian optimization, can be used for the hyperparameter optimization by
setting the ``config['SMAC']['use']`` parameter to ``'True'``. For details on which SMAC parameters
can be modified, see :ref:`Config chapter <config-chapter>`.

The core functionality of SMAC within WORC is implemented in
:py:mod:`WORC.resources.fastr_tools.worc.bin.smac_tool`. The configuration space of SMAC is specified
in :py:mod:`WORC.classification.smac`, which is also where new methods can be added to the search space.

There is additional output when using SMAC. The final output file ``smac_results_all_0.json``
is added along with the regular performance file in the output folder. It contains information on the
optimization procedure for each cross-validation split, with statistics on the performance and all
intermediate best found configurations.The end of the file contains a summary of the average statistics
over all train-test cross-validations.

Multilabel classification and regression
----------------------------------------
While ``WORC`` was primarily designed for binary classification, as also
demonstrated in the main manuscript, various other types of machine
learning workflows have been included as well.

In multilabel classification, several mutually exclusive classes are
predicted at the same time. This is a special form of multiclass classification,
in which the classes do not have to be mutually exclusive. When using
multilabel classification in ``WORC``, the only differences with binary
classification in the workflows is in the machine learning component.
For the other components, e.g. feature selection and resampling, when not
supporting multiclass classification, the methods are performed per
class in a one-vs-rest approach. Some of the binary classifiers naturally
support multilabel classification (i.e., random forest,  AdaBoost,
and extreme gradient boosting) and are thus normally used. Others only
support binary classification (i.e., LDA, QDA, Naive Bayes, SVM, logistic
regression), and are therefore also performed per class in a one-vs-rest
approach and combined in a single multilabel model. In the evaluation,
the same metrics as in the binary classification are evaluated per class.
Additionally, the multiclass AUC [CIT3]_. and multiclass BCR are computed.

In regression, a continuous label is predicted. As there are no classes,
all class-based feature and sample preprocessing methods
(RELIEF, univariate testing, and all resampling methods) cannot be used.
In the machine learning component, ``WORC`` includes the following regressors:

1. linear regression;
2. support vector machines;
3. random forest;
4. elastic net;
5. LASSO;
6. ridge regression;
7. AdaBoost;
8. extreme gradient boosting (XGBoost).

The optimization is by default based on the R2-score. Performance metrics
computed are the rw-score, mean squared error, inter-class correlation
coefficient, Pearson coefficient and p-value, and Spearman coefficient
and p-value.

References
------------
.. [CIT1] Chen, Jianan, et al. *AMINN: Autoencoder-based Multiple Instance
  Neural Network for Outcome Prediction of Multifocal Liver Metastases.*
  arXiv preprint arXiv:2012.06875 (2020).

.. [CIT2] Klein, Stefan, et al. *Elastix: a toolbox for intensity-based medical
   image registration.* IEEE transactions on medical imaging 29.1 (2009): 196-205.

.. [CIT3] Hand, David J., and Robert J. Till. *A simple generalisation
  of the area under the ROC curve for multiple class classification problems.*
  Machine learning 45.2 (2001): 171-186.
