..  usermanual-chapter:

User Manual
===========

In this chapter we will discuss the parts of WORC in more detail, mostly focussed on the inputs,
outputs, and the various workflow possible. We will give a more complete overview of the system
and describe the more advanced features.

.. _tools:

Interacting with WORC
---------------------
The WORC toolbox is build around of one main object, the WORC object. This object provides all functionality
of the toolbox. However, to make certain functionalities easier to use and limit the complexity,
we have constructed two facades. The ``SimpleWORC`` facade is the simplest to interact with and provides
all functionality required for conducting basic experiments. The ``BasicWORC`` object is based on the ``SimpleWORC``
object and provides several more advances functions. The specific functionalities of these two facades and the
``WORC`` object itself can be found in this section.

For documentation on ``SimpleWORC`` and ``BasicWORC``, please look at the documentation
within those modules: :py:mod:`WORC.facade.simpleworc` and :py:mod:`WORC.facade.basicworc`. Many of the functions are actually wrappers to interact with the WORC
object, and therefore use the functionality described below. For basic usage, only using
``SimpleWORC``, it's respective documentation and the
`WORCTutorial Github <https://github.com/MStarmans91/WORCTutorial/>`_ should be sufficient.

Additionally, we advice you to take a look at the :ref:`configuration chapter <config-chapter>`
for all the settings that can be adjusted in ``WORC``.

The WORC Object
~~~~~~~~~~~~~~~~
.. code-block:: python

   import WORC
   network = WORC.WORC('somename')

It's attributes are split in a couple of categories. We will not discuss
the WORC.defaultconfig() function here, which generates the default
configuration, as it is listed in a separate page, see the :ref:`Config chapter <config-chapter>`.
More detailed documentation of the various functions can be found in the docstrings of :py:mod:`WORC.WORC`:
we will mostly focus on the attributes, inputs, outputs and workflows here.


Input file definitions
----------------------

Attributes: Sources
~~~~~~~~~~~~~~~~~~~

There are numerous WORC attributes which serve as source nodes for the
FASTR network. These are:


-  images_train and images_test
-  segmentations_train and segmentations_test
-  semantics_train and semantics_test
-  labels_train and labels_test
-  masks_train and masks_test
-  features_train and features_test
-  metadata_train and metadata_test
-  Elastix_Para
-  fastrconfigs


When using a single dataset for both training and evaluation, you should
supply all sources in train objects. By default, performance on a single
dataset will be evaluated using cross-validation. Optionally, you can supply
a separate training and test set.

Each source should be given as a dictionary of strings corresponding to
the source filenames. Each element should correspond to a single object for the classification,
e.g. a patient. The keys are used to match the features to the
label and semantics sources, so make sure these correspond to the label
file.

You can off course have multiple images or ROIs per object, e.g. a liver
ROI and a tumor ROI. This can be easily done by appending to the
sources. For example:

.. code-block:: python

   images1 = {'patient1': '/data/Patient1/image_MR.nii.gz', 'patient2': '/data/Patient2/image_MR.nii.gz'}
   segmentations1 = {'patient1': '/data/Patient1/seg_tumor_MR.nii.gz', 'patient2': '/data/Patient2/seg_tumor_MR.nii.gz'}
   segmentations2 = {'patient1': '/data/Patient1/seg_liver_MR.nii.gz', 'patient2': '/data/Patient2/seg_liver_MR.nii.gz'}

   network.images_train.append(images1)
   network.images_train.append(images1)

   network.segmentations_train.append(segmentations1)
   network.segmentations_train.append(segmentations2)

When using multiple sequences per patients (e.g. T1 and T2), the same
appending procedure can be used.

If you want to use multiple ROIs independently per patient, e.g. multiple tumors, you can do so
by simply adding them to the dictionary. To make sure the data is still split per patient in the
cross-validation, please add a sample number after an underscore to the key, e.g.

.. code-block:: python

   images1 = {'patient1_0': '/data/Patient1/image_MR.nii.gz', 'patient1_1': '/data/Patient1/image_MR.nii.gz'}
   segmentations1 = {'patient1_0': '/data/Patient1/seg_tumor1_MR.nii.gz', 'patient1_1': '/data/Patient1/seg_tumor2_MR.nii.gz'}

If your label file (see below) contains the label ''patient1'', both samples will get this label
in the classification.

.. note:: You have to make sure the images and segmentation sources match in size.

.. note:: You have to supply a configuration file for each image or feature source you append.
          Thus, in the first example above, you need to append two configurations!

.. note:: When you use
          multiple image sequences, you can supply a ROI for each sequence by
          appending to to segmentations object. Alternatively, when you do not
          supply a segmentation for a specific sequence, WORC will use Elastix to
          align this sequence to another through image registration. It will then
          warp the segmentation from this sequence to the sequence for which you
          did not supply a segmentation. **WORC will always align these sequences with no segmentations to the first sequence, i.e. the first object in the images_train list.**
          Hence make sure you supply the sequence for which you have a ROI as the first object.

Images and segmentations
^^^^^^^^^^^^^^^^^^^^^^^^

The minimal input for a Radiomics pipeline consists of either images
plus segmentations, or features, plus a label file (and a configuration,
but you can just use the default one).

If you supply images and segmentations, features will be computed within the segmentations
on the images. They are read out using SimpleITK, which supports various
image formats such as DICOM, NIFTI, TIFF, NRRD and MHD.

.. _um-labels:

Labels
^^^^^^
The labels are predicted in the classification: should be a .txt or .csv file.
The first column should head ``Patient`` and contain the patient ID. The next columns
can contain labels you want to predict, e.g. tumor type, risk, genetics. For example:

+----------+--------+--------+
| Patient  | Label1 | Label2 |
+==========+========+========+
| patient1 | 1      | 0      |
+----------+--------+--------+
| patient2 | 2      | 1      |
+----------+--------+--------+
| patient3 | 1      | 5      |
+----------+--------+--------+


These labels are matched to the correct image/features by the sample names of the image/features. So in this
case, your sources should look as following:


.. code-block:: python

   images_train = {'patient1': ..., 'patient2': ..., ...}
   segmentations_train = {'patient1': ..., 'patient2': ..., ...}

Semantics
^^^^^^^^^
Semantic features are non-computational features and are extracted using PREDICT. Examples include
using the age and sex of the patients in the classification. You can
supply these as a .csv listing your features per patient, similar to the :ref:`label file <um-labels>`


Masks
^^^^^
WORC contains a segmentation preprocessing tool, called segmentix.
The idea is that you can manipulate
your segmentation, e.g. using dilation, then use a mask to make sure it
is still valid. See the :ref:`config chapter <config-chapter>` for all segmentix options.



Features
^^^^^^^^
If you already computed your features, e.g. from a previous run, you can
directly supply the features instead of the images and segmentations and
skip the feature computation step. These should be stored in .hdf5 files
matching the WORC format.


Metadata
^^^^^^^^
This source can be used if you want to use tags from the DICOM header as
features, e.g. patient age and sex. In this case, this source should
contain a single DICOM per patient from which the tags that are read.
Check the PREDICT.imagefeatures.patient_feature module for the currently
implemented tags.



Elastix_Para
^^^^^^^^^^^^
If you have multiple images for each patient, e.g. T1 and T2, but only a
single segmentation, you can use image registration to align and
transform the segmentation to the other modality. This is done in WORC
using Elastix http://elastix.isi.uu.nl/. In this source, you can supply
a parameter file for Elastix to be used in the registration in .txt.
format. Alternatively, you can use SimpleElastix to generate a parameter
map and pass this object to ``WORC``.

.. note:: ``WORC`` assumes your segmentation is made on the first
    ``WORC.images_train`` (or test) source you supply. The segmentation
    will be alligned to all other image sources.



Construction and execution commands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After supplying your sources as described above, you need to build the FASTR network. This
can be done through the ``WORC.build()`` command. Depending on your sources,
several nodes will be added and linked. This creates the ``WORC.network``
object, which is a ``fastr.network`` object. You can edit this network
freely, e.g. add another source or node. You can print the network with
the ``WORC.network.draw_network`` command.


Next, we have to tell the network which sources should be used in the
source nodes. This can be done through the ``WORC.set()`` function. This will
put your supplied sources into the source nodes and also creates the
needed sink nodes. You can check these by looking at the created
``WORC.source_data`` and ``WORC.sink_data`` objects.

Finally, after completing above steps, you can execute the network
through the ``WORC.execute()`` command.

Thus a typical experiment in ``WORC`` would follow the following structure,
assuming you have created the relevant objects as listed above:

.. code-block:: python

    import WORC

    # Create object
    experiment = WORC.WORC('name')

    # Append sources
    experiment.images_train.append(images_train)
    experiment.segmentations_train.append(segmentations_train)
    experiment.labels_train.append(labels_train)

    # Create a configuration
    config = experiment.defaultconfig()
    experiment.configs.append(config)

    # Build, set, and execute
    network.build()
    network.set()
    network.execute()


Evaluation of your network
--------------------------

In WORC, there are two options for testing your fitted models:

1. Single dataset: cross-validation (currently only random-split)
2. Separate train and test dataset: bootstrapping on test dataset

Within these evaluation settings, the following performance evaluation methods are used:

1. Confidence intervals on several metrics:

    For classification:

    a. Area under the curve (AUC) of the receiver operating characteristic (ROC) curve. In a multiclass setting, weuse the multiclass AUC from the `TADPOLE Challenge <https://tadpole.grand-challenge.org/>`_.
    b. Accuracy.
    c. Balanced Classification Accuracy (BCA) as defined by the `TADPOLE Challenge <https://tadpole.grand-challenge.org/>`_.
    d. F1-score
    e. Sensitivity, aka recall or true positive rate
    f. Specificity, aka true negative rate
    g. Negative predictive value (NPV)
    h. Precision, aka Positive predictive value (PPV)

    For regression:

    a. R2-score
    b. Mean Squared Error (MSE)
    c. Intraclass Correlation Coefficient (ICC)
    d. Pearson correlation coefficient and p-value
    e. Spearman correlation coefficient and p-value

    For survival, in addition to the regression scores:
    a. Concordance index
    b. Cox regression coefficient and p-value

    In cross-validation, by default, 95% confidence intervals for the mean performance measures are constructed using
    the corrected resampled t-test base on all cross-validation iterations, thereby taking into account that the samples
    in the cross-validation splits are not statistically independent. See als
    `Nadeau C, Bengio Y. Inference for the generalization error. In Advances in Neural Information Processing Systems, 2000; 307–313.`

    In bootstrapping, 95% confidence intervals are created using the ''standard'' method according to a normal distribution: see Table 6, method 1 in  `Efron B., Tibshirani R. Bootstrap Methods for Standard Errors,
    Confidence Intervals, and Other Measures of Statistical Accuracy, Statistical Science Vol.1, No,1, 54-77, 1986`.

2. ROC curve with 95% confidence intervals using the fixed-width bands method, see `Macskassy S. A., Provost F., Rosset S. ROC Confidence Bands: An Empirical Evaluation. In: Proceedings of the 22nd international conference on Machine learning. 2005.`

3. Univariate statistical testing of the features using:

    a. A student t-test
    b. A Welch test
    c. A Wilcoxon test
    d. A Mann-Whitney U test

    The uncorrected p-values for all these tests are reported in a single excel sheet. Pick the right test and significance
    level based on your assumptions. Normally, we make use of the Mann-Whitney U test, as our features do not have to be normally
    distributed, it's nonparametric, and assumes independent samples.

4. Ranking patients from typical to atypical as determined by the model, based on either:

    a. The percentage of times a patient was classified correctly when occuring in the test set. Patients always correctly classified
    can be seen as typical examples; patients always classified incorrectly as atypical.
    b. The mean posterior of the patient when occuring in the test set.

    These measures can only be used in classification. Besides an Excel with the rankings, snapshots of the middle slice
    of the image + segmentation are saved with the ground truth label and the percentage/posterior in the filename. In
    this way, one can scroll through the patients from typical to atypical to distinguish a pattern.

5. A barchart of how often certain features groups were selected in the optimal methods. Only useful when using
   groupwise feature selection.

    By default, only the first evaluation method, e.g. metric computation, is used. The other methods can simply be added
    to WORC by using the ``add_evaluation()`` function, either directly in WORC or through the facade:

6. Decomposition of your feature space.

    The following decompositions are performed:

    a. Principle Component Analysis (PCA)
    b. Sparse PCA
    c. Kernel PCA: linear kernel
    d. Kernel PCA: polynomial kernel
    e. Kernel PCA: radial basis function kernel
    f. t-SNE

    A decomposition can help getting insight into how your dataset can be separated. for example, if the
    regular PCA shows good separation of your classes, your classes can be split using linear combinations
    of your features.

To add the evaluation workflow, simply use the ``add_evaluation`` function:

.. code-block:: python

   import WORC
   experiment = WORC.WORC('somename')
   label_type = 'name_of_label_predicted_for_evaluation'
   ...
   experiment.add_evaluation(label_type)

.. code-block:: python

    import WORC
    from WORC import SimpleWORC
    experiment = SimpleWORC('somename')
    ...
    experiment.add_evaluation()

Debugging
---------

As WORC is based on fastr, debugging is similar to debugging a fastr pipeline: see therefore also
`the fastr debugging guidelines <https://fastr.readthedocs.io/en/stable/static/user_manual.html#debugging/>`_.

If you run into any issue, please create an issue on the `WORC Github <https://github.com/MStarmans91/WORC/issues/>`_.


Example data
------------

For many files used in typical WORC experiments, we provide example data. Some
of these can be found in the exampledata folder within the WORC package:
https://github.com/MStarmans91/WORC/tree/master/WORC/exampledata. To
save memory, for several types the example data is not included, but a script
is provided to create the example data. This script (``create_example_data``) can
be found in the exampledata folder as well.
