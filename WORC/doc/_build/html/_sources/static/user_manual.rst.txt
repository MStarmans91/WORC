..  usermanual-chapter:

User Manual
===========

In this chapter we will discuss the parts of WORC in more detail, mostly focussed on the inputs,
outputs, and the various workflow possible. We will give a more complete overview of the system
and describe the more advanced features.

.. _tools:

WORC object and facades
------------------------

The WORC toolbox is build around of one main object, the ``WORC`` object. This object provides all functionality
of the toolbox. However, to make certain functionalities easier to use and limit the complexity,
we have constructed two facades: ``SimpleWORC`` and ``BasicWORC``. We advice new users to start with ``SimpleWORC``,
more advanced users ``BasicWORC``, and only use ``WORC`` for development purposes. Additionally, we advice you to take a look at the :ref:`configuration chapter <config-chapter>`
for all the settings that can be adjusted in WORC.

The specific functionalities of these two facades and the ``WORC`` object itself can be found in this section.

SimpleWORC
~~~~~~~~~~~~~~~~
The ``SimpleWORC`` facade is the simplest to interact with and provides
all functionality required for conducting basic experiments. 
Much of the documentation of ``SimpleWORC`` can be found in its tutorial (https://github.com/MStarmans91/WORCtutorial and
:ref:`the quick start <quickstart-chapter>`) and the docstrings of the functions in the object (:py:mod:`WORC.facade.simpleworc`).
Many of the functions are  wrappers to interact with the ``WORC`` object, and therefore in the background use the functionality described below.

BasicWORC
~~~~~~~~~~~~~~~~
The ``BasicWORC`` object is based on the ``SimpleWORC`` object, and thus provides exactly the same functionality,
plus several more advances functions. Much of the documentation of ``BasicWORC`` can be found in its tutorial (https://github.com/MStarmans91/WORCtutorial) 
and the docstrings of the functions in the object (:py:mod:`WORC.facade.basicworc`).

One of the functionalities that ``BasicWORC`` provides over ``SimpleWORC`` is that you can also directly provide
your data to ``WORC`` (e.g. ``images_train``) instead of using one of the wrapping functions of
``SimpleWORC`` (e.g. ``images_from_this_directory)

.. _WORC:

WORC
~~~~~~~~~~~~~~~
The ``WORC`` object can directly be assessed in the following way:
.. code-block:: python

   import WORC
   network = WORC.WORC('somename')

It's attributes are split in a couple of categories. We will not discuss
the ``WORC.defaultconfig()`` function here, which generates the default
configuration, as it is listed in a separate page, see the :ref:`Config chapter <config-chapter>`.
More detailed documentation of the various functions can be found in the docstrings of :py:mod:`WORC.WORC`:
we will mostly focus on the attributes, inputs, outputs and workflows here.

There are numerous ``WORC`` attributes which serve as source nodes (i.e. inputs) for the
FASTR network. These are:

-  ``images_train`` and ``images_test``
-  ``segmentations_train`` and ``segmentations_test``
-  ``semantics_train`` and ``semantics_test``
-  ``labels_train`` and ``labels_test``
-  ``masks_train`` and ``masks_test``
-  ``features_train`` and ``features_test``
-  ``metadata_train`` and ``metadata_test``
-  ``Elastix_Para``
-  ``fastrconfigs``

These directly correspond to the :ref:`input file definitions discussed below <inputs>`
How to provide your data to ``WORC`` is also described in this section.

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


.. _inputs:

Input file definitions and how to provide them to WORC
-------------------------------------------------------

Providing your inputs to WORC and data flows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let's first start on how to provide any of the below mentioned types of input data to  ``WORC``.
``WORC`` facilitates different data flows (or networks or pipelines), which are automatically 
constructed based on the inputs and configuration you provide. We here 
discuss how the data can be set in ``BasicWORC`` and ``WORC``: 
``SimpleWORC`` provides several wrappers to more easily provide data, which interact with 
thee objects.

As an example, we here show how to provide images and segmentations to ``BasicWORC`` and ``WORC``. 

.. code-block:: python

   images1 = {'patient1': '/data/Patient1/image_MR.nii.gz', 'patient2': '/data/Patient2/image_MR.nii.gz'}
   segmentations1 = {'patient1': '/data/Patient1/seg_tumor_MR.nii.gz', 'patient2': '/data/Patient2/seg_tumor_MR.nii.gz'}

   network.images_train.append(images1)
   network.segmentations_train.append(segmentations1)

Here ``network`` can be a ``BasicWORC`` or ``WORC`` object. Each source is a list, to which you can provide
dictionaries containing the actual sources. In these dictionaries, each element should correspond to a single
object for classification, e.g., a patient or a lesions. The keys indicate
the ID of the element, e.g. the patient name, while the values should be strings corresponding to
the source filenames. The keys are used to match the images and segmentations to the
label and semantics sources, so make sure these correspond to the label file.

.. note:: You have to make sure the images and segmentation (and other) sources match in size,
           i.e., that the same keys are present.

.. note:: You have to supply a configuration file for each image or feature source you append.
          Thus, in the first example above, you need to append two configurations!

Using multiple sources per patient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you want to provide multiple sources, e.g. images, per patient, simply append another dictionary
to the source list, e.g.:

.. code-block:: python

   images1 = {'patient1': '/data/Patient1/image_MR.nii.gz', 'patient2': '/data/Patient2/image_MR.nii.gz'}
   images2 = {'patient1': '/data/Patient1/image_CT.nii.gz', 'patient2': '/data/Patient2/image_CT.nii.gz'}
   segmentations1 = {'patient1': '/data/Patient1/seg_tumor_MR.nii.gz', 'patient2': '/data/Patient2/seg_tumor_MR.nii.gz'}
   segmentations2 = {'patient1': '/data/Patient1/seg_tumor_CT.nii.gz', 'patient2': '/data/Patient2/seg_tumor_CT.nii.gz'}

   network.images_train.append(images1)
   network.images_train.append(images2)

   network.segmentations_train.append(segmentations1)
   network.segmentations_train.append(segmentations2)


``WORC`` will use the keys of the dictionaries to match the features from the same object or patient and combine
them for the machine learning part.

Mutiple ROIs or segmentations per object/patient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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

``WORC`` will use the keys of the dictionaries to match the features from the same object or patient and combine
them for the machine learning part.

If you want to use multiple ROIs independently per patient, e.g. multiple tumors, you can do so
by simply adding them to the dictionary. To make sure the data is still split per patient in the
cross-validation, please add a sample number after an underscore to the key, e.g.

.. code-block:: python

   images1 = {'patient1_0': '/data/Patient1/image_MR.nii.gz', 'patient1_1': '/data/Patient1/image_MR.nii.gz'}
   segmentations1 = {'patient1_0': '/data/Patient1/seg_tumor1_MR.nii.gz', 'patient1_1': '/data/Patient1/seg_tumor2_MR.nii.gz'}

If your label file (see below) contains the label ''patient1'', both samples will get this label
in the classification.

.. note:: ``WORC`` will automatically group all samples from a patient either all in the training
          or all in the test set.

Training and test sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When using a single dataset for both training and evaluation, you should
only supply "training" datasets. By default, performance on a single
dataset will be evaluated using cross-validation (default random split, but leave-one-out can also be configured). 
Alternatively, you can supply a separate training and test set, by which you tell 
``WORC`` to use this single train-test split. To distinguish between these, for every source, we have a 
train and test object which you can set:

.. code-block:: python

   images_train = {'patient1': '/data/Patient1/image_MR.nii.gz', 'patient2': '/data/Patient2/image_MR.nii.gz'}
   segmentations_train = {'patient1': '/data/Patient1/seg_tumor_MR.nii.gz', 'patient2': '/data/Patient2/seg_tumor_MR.nii.gz'}

   network.images_train.append(images_train)
   network.segmentations_train.append(segmentations_train)

   images_test = {'patient3': '/data/Patient3/image_MR.nii.gz', 'patient4': '/data/Patient4/image_MR.nii.gz'}
   segmentations_test = {'patient3': '/data/Patient3/seg_tumor_MR.nii.gz', 'patient4': '/data/Patient4/seg_tumor_MR.nii.gz'}

   network.images_test.append(images_test)
   network.segmentations_test.append(segmentations_test)

Another alternative is to only provide training objects, but also a .csv defining fixed training and test splits to be used for the 
evaluation, e.g. ``network.fixed_splits = '/data/fixedsplits.csv``. See the https://github.com/MStarmans91/WORCtutorial repository for an example. ``SimpleWORC`` has the ``set_fixed_splits`` to set this object.

Missing data and dummy's
^^^^^^^^^^^^^^^^^^^^^^^^^^
Suppose you are missing a specific image for a specific patient. ``WORC`` can impute the features of this patient. 
The underlying package we use for workflow execution (fastr) can however handle missing data. Therefore, to tell ``WORC`` to 
do so, you still have to provide a source but can add ''Dummy'' to the key:

.. code-block:: python

   images1 = {'patient1': '/data/Patientc/image_MR.nii.gz', 'patient2_Dummy': '/data/Patient1/image_MR.nii.gz'}
   segmentations1 = {'patient1': '/data/Patient1/seg_tumor_MR.nii.gz', 'patient2_Dummy': '/data/Patient1/seg_tumor_MR.nii.gz'}

   network.images_train.append(images1)
   network.segmentations_train.append(segmentations1)

``WORC``  will process the sources normally up till the imputation part, so you have to provide valid data. As you see in the example above,
we simply provided data from another patient.

Segmentation on the first image, but not on the others
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When you use multiple image sequences, you can supply a ROI for each sequence by
appending to to segmentations object as above. Alternatively, when you do not
supply a segmentation for a specific sequence, ``WORC`` will use Elastix to
align this sequence to another through image registration. It will then
warp the segmentation from this sequence to the sequence for which you
did not supply a segmentation. **WORC will always align these sequences with no segmentations to the first sequence, i.e. the first object in the images_train list.**
Hence make sure you supply the sequence for which you have a ROI as the first object:

.. code-block:: python

   images1 = {'patient1': '/data/Patient1/image_MR.nii.gz', 'patient2': '/data/Patient2/image_MR.nii.gz'}
   images2 = {'patient1': '/data/Patient1/image_CT.nii.gz', 'patient2': '/data/Patient2/image_CT.nii.gz'}
   segmentations1 = {'patient1': '/data/Patient1/seg_tumor_MR.nii.gz', 'patient2': '/data/Patient2/seg_tumor_MR.nii.gz'}

   network.images_train.append(images1)
   network.images_train.append(images2)

   network.segmentations_train.append(segmentations1)

When providing only a segmentation for the first image in this way, ``WORC`` will automatically
recognize that it needs to use registration.

Images and segmentations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The minimal input for a radiomics pipeline consists of either images
plus segmentations, or features, plus a label file (and a configuration,
but you can just use the default one).

If you supply images and segmentations, features will be computed within the segmentations
on the images. They are read out using SimpleITK, which supports various
image formats such as DICOM, NIFTI, TIFF, NRRD and MHD.

.. _um-labels:

Labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

.. note:: ``WORC`` will automatically group all samples from a patient either all in the training
            or all in the test set.

Semantics or non-radiomics features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Semantic features are non-computational features, thus features that you supply instead of extract. Examples include
using the age and sex of the patients in the classification. You can
supply these as a .csv listing your features per patient, similar to the :ref:`label file <um-labels>`

Masks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
WORC contains a segmentation preprocessing tool, called segmentix.
The idea is that you can manipulate
your segmentation, e.g. using dilation, then use a mask to make sure it
is still valid. See the :ref:`config chapter <config-chapter>` for all segmentix options.


Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you already computed your features, e.g. from a previous run, you can
directly supply the features instead of the images and segmentations and
skip the feature computation step. These should be stored in .hdf5 files
matching the WORC format.


Metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This source can be used if you want to use tags from the DICOM header as
features, e.g. patient age and sex. In this case, this source should
contain a single DICOM per patient from which the tags that are read.
Check the PREDICT.imagefeatures.patient_feature module for the currently
implemented tags.


Elastix_Para
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

.. _um-evaluation:

Outputs and evaluation of your network
---------------------------------------
General remark: when we talk about a sample, we mean one sample that has a set of features associated with it and is thus used as such in the model training or evaluation.
A sample can correspond with a single patient, but if you have multiple tumors per patient for which features are separately extracted per tumor, these can be treated as separate sample.

The following outputs and evaluation methods are always generated:

.. note:: For every output file, fastr generates a provenance file (``...prov.json``) stating how a file was generated, see https://fastr.readthedocs.io/en/stable/static/user_manual.html#provenance.

1. Performance of your models (main output).

    Stored in file ``performance_all_{num}.json``. If you created multiple models to predict multiple labels, or did multilabel classification, the ``{num}`` corresponds
    to the label. The file consists of three parts.
    
    **Mean and 95% confidence intervals of several performance metrics.**
 
    For classification:

    a. Area under the curve (AUC) of the receiver operating characteristic (ROC) curve. In a multiclass setting, weuse the multiclass AUC from the `TADPOLE Challenge <https://tadpole.grand-challenge.org/>`_.
    b. Accuracy.
    c. Balanced Classification Accuracy (BCA), based on Balanced Classification Rate by `Tharwat, A., 2021. Classification assessment methods. Applied Computing and Informatics 17, 168–192.`.
    d. F1-score
    e. Sensitivity or recall or true positive rate
    f. Specificity or true negative rate
    g. Negative predictive value (NPV)
    h. Precision or Positive predictive value (PPV)

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

    **Rankings of your samples**
    In thid dictionary, the "Percentages" part shows how often a sample was classified correctly
    when that sample was in the test set. The number of times the sample was in in the test set is also listed.
    Those samples that were always classified correctly or always classified incorrecty are also named, including their ground truth label. 

    **The metric values for each train-test cross-validation iteration**
    These are where the confidence intervals are based upon.

2. The configuration used by WORC.

    Stored in files ``config_{type}_{num}.ini``. These are the result of the fingerprinting of your dataset. The ``config_all_{num}.ini`` config is used in classification, the other types
    are used for feature extraction and are named after the image types you provided. For example, if you provided two image types, ``['MRI', 'CT']``, you will get
    ``config_MRI_0.ini`` and ``config_CT_0.ini``. If you provide multiple of the same types, the numbers will change. The fields correspond with those from :ref:`configuration chapter <config-chapter>`.

3. The fitted models.

    Stored in file ``estimator_all_{num}.hdf5``. Contains a pandas dataframe, with inside a pandas series per label for which WORC fitted a model, commonly just one.
    The series contains the following attributes:

    - classifiers: a list with per train-test cross-validation, the fitted model on the training set. These are thus the actually fitted models.
    - X_train: a list with per train-test cross-validation, a list with for each sample in the training set all feature values. These can be used in re-fitting.
    - Y_train: a list with per train-test cross-validation, a list with for each sample in the training set the ground truth labels. These can be used in re-fitting.
    - patient_ID_train: a list with per train-test cross-validation, a list with the labels of all samples included in the training set.
    - X_test: a list with per train-test cross-validation, a list with for each sample in the test set all feature values. These can be used in re-fitting.
    - X_test: a list with per train-test cross-validation, a list with for each sample in the test set the ground truth labels. These can be used in re-fitting.
    - patient_ID_test: a list with per train-test cross-validation, a list with the labels of all samples included in the test set.
    - config: the WORC config used. Corresponds to the ``config_all_{num}.ini`` file mentioned above.
    - random-seed: a list with per train-test cross-validation, the random seed used in splitting the train and test dataset. 
    - feature_labels: the names of the features. As these are the same for all samples, only one set is provided.

4. The extracted features.

    Stored in the ``Features`` folder, in the files ``features_{featuretoolboxname}_{image_type}_{num}_{sample_id}.hdf5``. Contains a panas series wih the following attributes:

    - feature_labels: the labels or names of the features.
    - feature_values: the value of the features. Each element corresponds with the same element from the feature_labels attribute.
    - parameters: the parameters used in the feature extraction. Originate from the WORC config.
    - image_type: the type of the image that was used, which you as user provided. Used in the feature labels to distinguish between features extracted from different images.

The following outputs and evaluation methods are only created when ``WORC.add_evaluation()`` is used (similar for ``SimpleWORC`` and ``BasicWORC``),
and are stored in the ``Evaluation`` in the output folder of your experiment.

1. Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves.
   
   Stored in files ``ROC_all_{num}.{ext}`` and ``PRC_all_{num}.{ext}``. For each curve, a ``.png`` is generated for previewing, a ``.tex`` with tikzplotlib
   which can be used to plot the figure in LateX in high quality, and a ``.csv`` with the confidence intervals so you can easily check these.

    95% confidence bands are constructured using the fixed-width bands method from `Macskassy S. A., Provost F., Rosset S. ROC Confidence Bands: An Empirical Evaluation. In: Proceedings of the 22nd international conference on Machine learning. 2005.`

2. Univariate statistical testing of the features.

    Stored in files ``StatisticalTestFeatures_all_{num}.{ext}``. A ``.png`` is generated for previewing, a ``.tex`` with tikzplotlib
    which can be used to plot the figure in LateX in high quality, and a ``.csv`` with the p-values. 

    The following statistical tests are used:

    a. A student t-test
    b. A Welch test
    c. A Wilcoxon test
    d. A Mann-Whitney U test

    The uncorrected p-values for all these tests are reported in a the .csv. Pick the right test and significance
    level based on your assumptions. 
    
    Normally, we make use of the Mann-Whitney U test, as our features do not have to be normally
    distributed, it's nonparametric, and assumes independent samples. Additionally, generally correction should be done
    for multiple testing, which we always do with Bonferonni correction. Hence, .png and .tex files contain the 
    p-values of the Mann-Whitney U; the p-value of the magenta statistical significance has been corrected with 
    Bonferonni correction.

3. Overview of hyperparameters used in the top ranked models.
   
    Stored in file ``Hyperparameters_all_{num}.csv``. 

    Each row corresponds with the hyperparameters of one workflow. The following information is displayed in the respective columns:

    A. The cross-validation iteration.
    B. The rank of that workflow in that cross-validation.
    C. The metric on which the ranking in column B was based.
    D. The mean score on the validation datasets in the nested cross-validation of the metric in column C.
    E. The mean score on the training datasets in the nested cross-validation of the metric in column C.
    F. The mean time it took to fit that workflow in the validation datasets.
    G. and further: the actual hyperparameters.

    For how many of the top ranked workflows the hyperparameters are included in this file depends on the ``config["Ensemble"]["Size"]``, see :ref:`configuration chapter <config-chapter>`.

4. Boxplots of the features.

    Stored in ``BoxplotsFeatures_all_{num}.zip``. The .zip files contains multiple .png files, each with maximum 25 boxplots of features.

    For the full **training** dataset (i.e., if a separate test-set is provided, this is not included in these plots.), per features, one boxplot
    is generated depicting the distribution of features for all samples (blue), and for binary classification, also only for the samples
    with label 0 (green) and for the samples with label 1 (red). Hence, this gives an impression whether some features show major differences
    in the distribution among the different classes, and thus could be useful in the classification to separate them.     

5. Ranking patients from typical to atypical as determined by the model.

    Stored in files ``RankedPosteriors_all_{num}.{ext}`` and ``RankedPercentages_all_{num}.{ext}``. 

    Two types of rankings are done:

    a. The percentage of times a patient was classified correctly when occuring in the test set. Patients always correctly classified
    can be seen as typical examples; patients always classified incorrectly as atypical.
    b. The mean posterior of the patient when occuring in the test set.

    These measures can only be used in classification. Besides a .csv with the rankings, snapshots of the middle slice
    of the image + segmentation are saved with the ground truth label and the percentage/posterior in the filename in 
    a .zip file. In this way, one can scroll through the patients from typical to atypical to distinguish a pattern.

6. A barchart of how often certain features groups or feature selection groups were selected in the optimal methods.

    Stored in files ``Barchart_all_{num}.{ext}``. A ``.png`` is generated for previewing, a ``.tex`` with tikzplotlib
    which can be used to plot the figure in LateX in high quality.

    Gives an idea of which features are most relevant for the predictions of the model, and which feature methods are often succesful.
    The overview of the hyperparameters, see above, is more quantitative and useful however.

7. Decomposition of your feature space.

    Stored in file ``Decomposition_all_{num}.png``.

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

Or in the ``SimpleWORC`` or ``BasicWORC`` facades:

.. code-block:: python

    from WORC import SimpleWORC
    experiment = SimpleWORC('somename')
    ...
    experiment.add_evaluation()

The following outputs are only generated if certain configuration settings are used:

1. Adjusted segmentations.

    Stored in the ``Segmentations`` folder, in the files ``seg__{image_type}_{num}_{howsegmentationwasgenerated}_{sample_id}.hdf5``.
    Only generated when the original segmentations were modified, e.g. using WORC's internal program segmentix 
    (see relevant section of the :ref:`configuration chapter <config-chapter>`) or when registration was 
    performed to warp the segmentations from one sequence to another.


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
