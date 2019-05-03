User Manual
===========

In this chapter we will discuss the parts of Fastr in more detail. We will give a more complete overview of the system
and describe the more advanced features.

.. _tools:

The WORC object
---------------

The WORC toolbox consists of one main object, the WORC object:



.. code-block:: python

   import WORC
   network = WORC.WORC('somename')



It's attributes are split in a couple of categories. We will not discuss
the WORC.defaultconfig() function here, which generates the default
configuration, as it is listed in a separate page, see the  :ref:`config file section <config-chapter>`.



Attributes: Sources
-------------------



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
supply all sources in train objects. You can use several kinds of
validation methods (e.g.cross validation) to compute the performance on
this dataset. Optionally, you can supply a separate training and test
set.


Each source should be given as a dictionary of strings corresponding to
the source files. Each element should correspond to a single object,
e.g. tumor, or patient. The keys are used to match the features to the
label and semantics sources, so make sure these correspond to the label
file. The values should refer to the actual source files corresponding
to the FASTR formats, see
http://fastr.readthedocs.io/en/stable/fastr.reference.html#ioplugin-reference.


You can off course have multiple images or ROIs per object, e.g. a liver
ROI and a tumor ROI. This can be easily done by appending to the
sources. For example:

.. code-block:: python

   images1 = {'patient1': vfs://example/MR.nii, 'patient2': vfs://example/MR.nii}
   segmentations1 = {'patient1': vfs://example/tumor.nii, 'patient2': vfs://example/tumor.nii}
   segmentations2 = {'patient1': vfs://example/liver.nii, 'patient2': vfs://example/liver.nii}

   network.images_train.append(images1)
   network.images_train.append(images1)

   network.segmentations_train.append(segmentations1)
   network.segmentations_train.append(segmentations2)



When using multiple sequences per patients (e.g. T1 and T2), the same
appending procedure can be used.


.. note:: You have to make sure the images and segmentation sources match in size.

.. note:: You have to supply a configuration file for each image or feature source you append.
          Thus, in above example, you need to append two configurations!
.. note:: When you use
          multiple image sequences, you can supply a ROI for each sequence by
          appending to to segmentations object. Alternatively, when you do not
          supply a segmentation for a specific sequence, WORC will use Elastix to
          align this sequence to another through image registration. It will then
          warp the segmentation from this sequence to the sequence for which you
          did not supply a segmentation. **WORC will always align these sequences with no segmentations to the first sequence, i.e. the first object in the images_train list.**
          Hence make sure you supply the sequence for which you have a ROI as the first object.



Attributes: Settings
--------------------


There are several attributes in WORC which define how your pipeline is
executed:



-  fastr_plugin
-  fastr_tmpdir
-  Tools: additional workflows are stored here. Currently only includes
   a pipeline for image registration without any Radiomics.
-  CopyMetadata: Whether to automatically copy the metadata info
   (e.g. direction of cosines) from the images to the segmentations
   before applying transformix.

An explanation of the FASTR settings is given below.



Attributes: Functions
---------------------

The WORC.configs() attribute contains the configparser files, which you
can easily edit. The WORC.set() function saves these objects in a
temporary folder and converts the filename into as FASTR source, which
is then put in the WORC.fastrconfigs() objects. Hence you do not need to
edit the fastrconfigs object manually.



Images and segmentations
~~~~~~~~~~~~~~~~~~~~~~~~



The minimal input for a Radiomics pipeline consists of either images
(plus a segmentation if you have not implemented an automatic
segmentation tool) or features plus a label file (and a configuration,
but you can just use the default one.

If you supply these, features will be computed within the segmentations
on the images. They are read out using SimpleITK, which supports various
image formats such as DICOM, NIFTI, TIFF, NRRD and MHD.



Semantics
~~~~~~~~~

Semantic features are used in the PREDICT CalcFeatures tool. You can
supply these as a .csv listing your features per patient. The first
column should always be named ``Patient`` and contain the Patient ID. The
other columns should contain a label for the feature and their values.
For example:



+----------+--------+--------+
| Patient  | Label1 | Label2 |
+==========+========+========+
| patient1 | 1      | 0      |
+----------+--------+--------+
| patient2 | 2      | 1      |
+----------+--------+--------+
| patient3 | 1      | 5      |
+----------+--------+--------+


Similar to the patient labels, the semantic features are matched to the
correct image/features by the name of the image/features. So in this
case, your sources should look as following:



.. code-block:: python

   images_train = {'patient1': 'source1.nii.gz', 'patient2': 'source2.nii.gz', ...}
   segmentations_train = {'patient1': 'seg1.nii.gz', 'patient2': 'seg2.nii.gz', ...}



Labels
~~~~~~

The labels are used in classification. For PREDICT, these should be
supplied as a .txt file. Similar to the semantics, the first column
should head ``Patient`` and contain the patient ID. The next columns can
contain things you want to predict. Hence the format is similar to the
semantics file.


Masks
-----------

WORC contains a segmentation preprocessing tool, called segmentix. This
tool is still under development. The idea is that you can manipulate
your segmentation, e.g. using dilation, then use a mask to make sure it
is still valid. Currently, you can only let it take a ring of a certain
radius around your ROI and mask it.



Features
--------

If you already computed your features, e.g. from a previous run, you can
directly supply the features instead of the images and segmentations and
skip the feature computation step. These should be stored in .hdf5 files
matching the PREDICT CalcFeatures format.


Metadata
--------

This source can be used if you want to use tags from the DICOM header as
features, e.g. patient age and sex. In this case, this source should
contain a single DICOM per patient from which the tags that are read.
Check the PREDICT.imagefeatures.patient_feature module for the currently
implemented tags.



Elastix_Para
------------

If you have multiple images for each patient, e.g. T1 and T2, but only a
single segmentation, you can use image registration to align and
transform the segmentation to the other modality. This is done in WORC
using Elastix http://elastix.isi.uu.nl/. In this source, you can supply
a parameter file for Elastix to be used in the registration in .txt.
format. Alternatively, you can use SimpleElastix to generate a parameter
map and pass this object to WORC. **Note: WORC assume your segmentation
is made on the first WORC.images source you supply. The segmentation
will be alingned to all other image sources.**



FASTR settings
--------------

There are two WORC attributes which contain settings on running FASTR.
In WORC.fastr_plugin, you can specify which Execution Plugin should be
used: see also
http://fastr.readthedocs.io/en/stable/fastr.reference.html#executionplugin-reference.

The default is the ProcessPollExecution plugin. The WORC.fastr_tempdir
sets the temporary directory used in your run.



Construction and execution commands
-----------------------------------



After supplying your sources, you need to build the FASTR network. This
can be done through the WORC.build() command. Depending on your sources,
several nodes will be added and linked. This creates the WORC.network()
object, which is a fastr.network() object. You can edit this network
freely, e.g. add another source or node. You can print the network with
the WORC.network.draw_network() command.


Next, we have to tell the network which sources should be used in the
source nodes. This can be done through the WORC.set() command. This will
put your supplied sources into the source nodes and also creates the
needed sink nodes. You can check these by looking at the created
WORC.source_data_data and WORC.sink objects.


Finally, after completing above steps, you can execute the network
through the WORC.execute() command.
