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

Execution
-------------

My experiment crashed, where to begin looking for errors?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``fastr`` toolbox has a method to trace back errors. For more details,
see the `fastr documentation <https://fastr.readthedocs.io/en/stable/static/user_manual.html#debugging-a-network-run-with-errors/>`_.


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

Error: ``File "...\lib\site-packages\numpy\lib\function_base.py", line 4406,`` `` in delete keep[obj,] = False`` ``IndexError: arrays used as indices must be of integer (or boolean) type``
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

I am working on the BIGR cluster and would like some jobs to be submitted to different queues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
