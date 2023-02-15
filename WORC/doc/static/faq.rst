.. _faq-chapter:

FAQ
=======================

Installation
-------------

Error: ``ModuleNotFoundError: No module named 'numpy'``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Some versions of several packages that WORC uses, such as PyWavelets and
PyRadiomics, require numpy during their installation. To solve this issue,
simply first install numpy before installing WORC or any of the dependencies
, i.e. ``pip install numpy`` or ``conda install numpy`` when using Anaconda.

Execution errors
----------------

My experiment crashed, where to begin looking for errors?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``fastr`` toolbox has a method to trace back errors. For more details,
see the `fastr documentation <https://fastr.readthedocs.io/en/stable/static/user_manual.html#debugging-a-network-run-with-errors>`_.
If you want to know the exact error that occured in a job, make sure you trace back to a single sink and single sample,
e.g. ``fastr trace $RUNDIR/__sink_data__.json --sinks sink_5 --sample sample_1_1 ``.


Error: ``File "H5FDsec2.c", line 941, in H5FD_sec2_lock unable to lock file, errno = 37, error message = 'No locks available'``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Known HDF5 error, see also https://github.com/h5py/h5py/issues/1101.
Can be solved by setting the HDF5_USE_FILE_LOCKING environment variable to 'FALSE',
e.g. adding export HDF5_USE_FILE_LOCKING='FALSE' to your ~..bashrc on Linux.

Error: ``Failed building wheel for cryptography`` (occurs often on BIGR cluster)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This bug can be caused when using pyOpenSSL 22.1.0 or recent cryptography versions on the BIGR cluster.
Cryptography 3.4.7 and PyOpenSSL 20.0.1 should work, so install those (in that order) before installing WORC.

Error: ``WORC.addexceptions.WORCValueError: First column in the file`` ``given to SimpleWORC().labels_from_this_file(**) needs to be named Patient.``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This means that your label file, i.e. in which the label to be predicted for
each patient is given, is not formatted correctly. Please see the
:ref:`Configuration chapter <config-chapter>`, or the WORC Tutorial Github
for an `example <https://github.com/MStarmans91/WORCTutorial/blob/master/Data/Examplefiles/pinfo_HN.csv/>`_.

Error: ``WORC.addexceptions.WORCKeyError: 'No entry found in labeling`` ``for feature file .../feat_out_0.hdf5.'``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This means for this specific file (../feat_out_0.hdf5), WORC could not
find a label in your label file. Please make sure that one of the Patient IDs
from your label file occurs in the filename of your inputs. For example,
when using the example label file from the `WORC tutorial <https://github.com/MStarmans91/WORCTutorial/blob/master/Data/Examplefiles/pinfo_HN.csv/>`_,
if your Patient ID is not listed in column 1, this error will occur.

Error: ``File "...\lib\site-packages\numpy\lib\function_base.py", line 4406, in delete keep[obj,] = False IndexError: arrays used as indices must be of integer (or boolean) type``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is an error in PyRadiomics 3.0, see also
`this issue <https://github.com/Radiomics/pyradiomics/issues/592/>`_. It has
currently to be manually solved by within the PyRadiomics package, in the
``glcm``, ``gldm``, ``glrlm``, ``glszm`` and ``ngtdm`` functions,
searching for the line starting with ``emptyGrayLevels =``. After that,
there will be a line similar to ``P_ngtdm = numpy.delete(P_ngtdm, emptyGrayLevels, 1)``.
Before that line, add a conditional ``if list(emptyGrayLevels):``, e.g.
for the NGTDM:

.. code-block:: python

  if list(emptyGrayLevels):
    P_ngtdm = numpy.delete(P_ngtdm, emptyGrayLevels, 1)

See also my fork of PyRadiomics, which you can also install to fix the issue:
https://github.com/MStarmans91/pyradiomics.

Other
-----

I am working on the BIGR cluster and would like some jobs to be submitted to different queues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Unfortunately, fastr does not support giving a queue argument per job. In
general, we assume you would like all your jobs to be run on the day queue,
which you can set as the default, and only the classify job on the week queue.
The only solution we currently have is to manually hack this into fastr:

1. Go to the installation of the fastr package in your (virtual) environment.
2. Open the fastr/resources/plugins/executionplugins/drmaaplugin.py script.
3. Search for the line ``if queue is None:`` and replace that if loop
  with the following:

.. code-block:: python

  if queue is None:
      if 'classify' in command:
          fastr.log.info('Detected classify in command: submitting to week queue')
          queue = 'week'
      elif any('classify' in a for a in arguments):
          fastr.log.info('Detected classify in arguments: submitting to week queue')
          queue = 'week'
      else:
          queue = self.default_queue

Can I use my own features instead of the standard ``WORC`` features?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``WORC`` also includes an option to use your own features instead of the default
features included. ``WORC`` will than simply start at the data mining
(e.g. classification, regression) step, and thus after the normal
feature extraction. This requires three things


1. Convert your features to the default ``WORC`` format
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
``WORC`` expects your features per patient in a .hdf5 file, containing a ``pandas`` series
with at least a ``feature_values`` and a ``feature_labels`` object. The
``feature_values`` object should be a list containing your feature values,
the ``feature_labels`` object a list with the corresponding featuree labels.
Below an example on how to create such a series.

.. code-block:: python

  # Dummy variables
  feature_values = [1, 1.5, 25, 8]
  feature_labels = ['label_feature_1', 'label_feature_2', 'label_feature_3',
                    'label_feature_4']

  # Output filename
  output = 'test.hdf5'

  # Converting features to pandas series and saving
  panda_data = pd.Series([feature_values,
                          feature_labels],
                         index=['feature_values', 'feature_labels'],
                         name='Image features'
                         )

  panda_data.to_hdf(output, 'image_features')

2. Alter feature selection on the feature labels
"""""""""""""""""""""""""""""""""""""""""""""""""""
``WORC`` by default includes groupwise feature selection, were groups of
features are randomly turned on or off. Since your feature labels are probably
not in the default included values, you should turn this of. This can be done
by setting the ``config['Featsel']['GroupwiseSearch']`` to ``"False"``.

Alternatively, you can use default feature labels in ``WORC`` and still use
the groupwise feature selection. This is relatively simple: for example,
shape features are recognized by looking for ``"sf_"`` in the feature label
name. To see which labels are exactly used, please see
:py:mod:`WORC.featureprocessing.SelectGroups` and the SelectFeatGroup section in the
:ref:`Config chapter <config-chapter>`.

3. Tell ``WORC`` to use your feature and not compute the default ones
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
To this end, ``SimpleWORC``, and therefore also ``BasicWORC``, include the
function ``features_from_this_directory()``. See also the
:ref:`quick start guide <quickstart-chapter>`. As explained in the WORCTutorial,
a default structure of your ``featuresdatadir`` folder is expected in this
function: there should be a subfolder for each patient, in which the feature
file should be. The feature file can have a fixed name, but wildcard are
allowed in the search, see also the documentation of the ``features_from_this_directory()``
function.

Altneratively, when using ``BasicWORC``, you can append dictionaries to the
``features_train`` object. Each dictionary you append should have as keys
the patient names, and as values the paths to the feature files, e.g.
``feature_dict = {'Patient1': '/path/to/featurespatient1.hdf5',
'Patient2': '/path/to/someotherrandandomfolderwith/featurespatient2.hdf5'...}``.

How to change the temporary and output folders?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``WORC`` makes use of the ``fastr`` workflow engine to manage and execute
the experiment, and thus also to manage and produce the output. These folders
can be configured in the ``fastr`` config (https://fastr.readthedocs.io/en/stable/static/file_description.html#config-file).
The ``fastr`` config files can be found in a hidden folder .fastr in your home folder.
``WORC`` adds an additional config file to the config.d folder of ``fastr``:
https://github.com/MStarmans91/WORC/blob/master/WORC/fastrconfig/WORC_config.py.

The two mounts that determine the temporary and output folders and thus which
you have to change are:
- Temporary output: ``mounts['tmp']`` in the ~/.fastr/config.py file
- Final output: ``mounts['output']`` in the ~/.fastr/config.d/WORC_config.py file

How can I get the performance on the validation dataset?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The performance of the top 1 workflow is stored in the fitted estimators in the estimator_all_0.hdf5 file:

.. code-block:: python

      data = pd.read_hdf("estimator_all_0.hdf5")
      data = data[list(data.keys())[0]]

      validation_performance = list()
      # Iterate over all train-test cross validations
      for clf in data.classifiers:
          validation_performance.append(clf.best_score_)


My jobs on the BIGR cluster get cancelled due to memory errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can adjust the memory for various jobs through changing the values in the ``WORC.fastr_memory_parameters`` dictionary 
(accesible in ``SimpleWORC`` and ``BasicWORC`` through ``_worc.fastr_memory_parameters``.) The fit_and_score job
memory can be adjusted through the WORC HyperOptimization config, see :ref:`Configuration chapter <config-chapter>`.
