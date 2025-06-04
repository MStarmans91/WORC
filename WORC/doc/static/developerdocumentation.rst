Developer documentation
=======================

Information on the `fastr`` workflow engine
---------------------------------------------
The `WORC` toolbox makes use of the `fastr` package [1]_, an automated workflow engine.
`fastr` does not provide any actual implementation of the required (radiomics) algorithms,
but serves as a computational workflow engine, which has several advantages.

Firstly, `fastr` requires workflows to be modular and split into standardized components
or *tools*, with standardized inputs and outputs. This nicely connects to the modular design of `WORC`, for which we therefore wrapped each component as a tool in `fastr`. Alternating between feature extraction toolboxes can be easily done by changing a single field in the `WORC` toolbox configuration.

Second, provenance is automatically tracked by `fastr` to facilitate repeatability and reproducibility.

Third, `fastr` offers support for multiple execution plugins in order to be able to
execute the same workflow on different computational resources or clusters. Examples
include linear execution, local threading on multiple CPUs, and SLURM [2]_.

Fourth, `fastr` is agnostic to software language. Hence, instead of restricting the
user to a single programming language, algorithms (e.g., feature toolboxes) can be
supplied in a variety of languages such as `Python`, `Matlab`, `R`, and command line executables.

Fifth, `fastr` provides a variety of import and export plugins for loading and saving
data. Besides using the local file storage, these include the use of `XNAT` [3]_.


Adding a feature processing toolbox
-----------------------------------
We suggest to use the wrapping we did around the PyRadiomics toolbox as an example.

1. Check if config type is in ``fastr.types``: else, make your own. See
   the `WORC datatypes <https://github.com/MStarmans91/WORC/tree/master/WORC/resources/fastr_types/>`_
   for examples.
2. Make a fastr tool, which basically is a XML wrapper around your tool telling fastr
   what the inputs and outputs are. You can take our wrappers around pyradiomics as example:
   see `the command line wrapper <https://github.com/MStarmans91/WORC/blob/master/WORC/resources/fastr_tools/pyradiomics/pyradiomics.xml/>`_
   for when your toolbox is command line executable, or the `Python wrapper <https://github.com/MStarmans91/WORC/blob/master/WORC/resources/fastr_tools/pyradiomics/CF_pyradiomics.xml/>`_
   for when your toolbox needs an interpreter such as Python. In the latter case, you will also need to include
   an actual script using the interpreter, which is by default placed in the bin folder together
   with the wrapper. The script will need to parse the arguments defined in the XML,
   see `here <https://github.com/MStarmans91/WORC/blob/master/WORC/resources/fastr_tools/pyradiomics/bin/CF_pyradiomics_tool.py/>`_
   for the one using PyRadiomics.

   For more information, visit `the fastr documentation <https://fastr.readthedocs.io/en/stable/static/user_manual.html#create-your-own-tool/>`_.
3. In WORC, go to the :py:mod:`WORC.WORC.WORC.add_feature_calculator` function and change the following:

  a. Add a converter for your config file if you do not use the .ini format WORC by default uses.
  b. If you followed a, also add a config safe function to :py:mod:`WORC.WORC.WORC.save_config`.
  c. Make sure you add both the sources and sinks for your tools.
  d. Link these sources and sinks to the fastr network in the :py:mod:`WORC.WORCWORC.set` function.
  e. Add part to feature converter

4. To convert the feature files from your toolbox to WORC compatible format,
   add the neccesary functionality to the :py:mod:`WORC.featureprocessing.FeatureConverter`
   function.

5. Tell WORC to use your feature calculator by changing the relevant config field: ``config['General']['FeatureCalculators]``.

6. Optionally, add the parameters specifically for your toolbox to the WORC
   configuration in :py:mod:`WORC.WORC.WORC.defaultconfig`.

Testing and example data
-------------------------
WIP

Adding methods to hyperoptimization
-----------------------------------
1. Add the parameters to the relevant parts of the WORC configuration in
   :py:mod:`WORC.WORC.WORC.defaultconfig`.

   Please add a description of these parameters and their potential values to
   the documentation, see https://github.com/MStarmans91/WORC/blob/master/WORC/doc/generate_config.py.

2. Make sure these fields are adequately parsed by adding parsing to
   :py:mod:`WORC.IOparser.config_io_classifier.load_config`.

3. Define how the parameters you just added are added to the search space
   in :py:mod:`WORC.classification.trainclassifier.trainclassifier`, or
   :py:mod:`WORC.classification.construct_classifier` both  the
   ``construct_classifier`` and ``create_param_grid`` functions for machine
   learning estimators (e.g. classifiers). For
   example, your parameters may define the actual options, or the min-max of
   a distribution (e.g. uniform, log-uniform). See the referred function
   for some examples. Make sure that the object is iterable.

4. These parameters will end up in the function fitting the workflow:
   :py:mod:`WORC.classification.fitandscore.fit_and_score`. Hence,
   add a part to that function to embed your method in the workflow. We advice
   you to embed your method in a sklearn compatible class, having init,
   fit and transform functions. See for example
   :py:mod:`WORC.featureprocessing.Preprocessor.Preprocessor`.

   In
   :py:mod:`WORC.classification.fitandscore.fit_and_score`, make sure
   that after fitting your object, the parameters used are deleted from the
   config, as is done for the other methods as well.

   Lastly, in :py:mod:`WORC.classification.fitandscore.fit_and_score`,
   make sure the fitted object is returned. We recommend looking at the
   ``imputer`` object and similarly including your object.

   This is given to various objects
   in the :py:mod:`WORC.classification.SearchCV` module. Therefore,
   add the returned object to all the parts were fitted objects are used: we
   recommend looking everywhere the ``imputer`` is stated in
   :py:mod:`WORC.classification.SearchCV`, copying those five statements
   and replace ``imputer`` with however you called your methods. You can see
   that this is also similar to e.g. the ``scaler``, ``pca``, and ``groupsel``
   objects.

5. If you want your new method to be used by the ``SimpleWORC`` or a child
   facade, check :py:mod:`WORC.facade.SimpleWORC` to see if you need to add it,
   e.g. whitelist a classifier.

Adding a (plotting) tool to the WORC evaluation pipeline
----------------------------------------------------------
We illustrate here how plotting the ROC curves is embedded in ``WORC``, and
to follow or even copy-paste this example to add your own tools. 

1. Write a script to perform the actual analysis, preferably stored in the 
   plotting subfolder. See for example the one of the
   `ROC curves <https://github.com/MStarmans91/WORC/blob/master/WORC/plotting/plot_ROC.py/>`_.
2. Make it command-line executable and able to parse input arguments. For the ROC curves, we did this 
   in the above script as well, but that is optional. It should be stored in 
   `the WORC fastr_tools folder <https://github.com/MStarmans91/WORC/tree/master/WORC/resources/fastr_tools/worc/bin/>`_,
   see also the `ROC script in that folder <https://github.com/MStarmans91/WORC/blob/master/WORC/resources/fastr_tools/worc/bin/PlotROC.py/>`_,
   for step 3. Make sure it both is able to take in input arguments (parameters, file names) and output arguments (file names)
   to store the results in.
3. Make a fastr tool, i.e., a wrapper around your ``main`` function that fastr can call. See 
   `the general fastr documentation on creating your own tool <https://fastr.readthedocs.io/en/stable/static/user_manual.html#create-your-own-tool/>`_,
   and the `WORC tool for the ROC curve plotting< https://github.com/MStarmans91/WORC/blob/master/WORC/resources/fastr_tools/worc/PlotROC.xml/``>_.
   You can see we call the script from step 2 for this. For the input and output files, you can run ``fastr.types`` to see which
   datatypes are in your current ``fastr`` installation. You an see we added
   several in ``WORC``, see also `the WORC fastr_types folder <https://github.com/MStarmans91/WORC/tree/master/WORC/resources/fastr_types/>`_.
4. Now add it the the `Evaluation part of WORC <https://github.com/MStarmans91/WORC/blob/master/WORC/tools/Evaluate.py/>`_.
   If you are new to creating fastr networks, you may want to check out
   `the fastr documentation <https://fastr.readthedocs.io/en/stable/static/quick_start.html#creating-a-simple-network/>`_,
   but in principle you can just copy paste again the parts of the plotting of the ROC curve. Make sure you add: the
   additional sources (inputs) your tool requires if they are not already in the rest of WORC, the actual tool you made,
   and sinks (outputs) so the output is also actually stored when your tool is done in an output folder.


.. _references:

References
==========

.. [1] Achterberg, H. C., Koek, M., & Niessen, W. J. (2016). *Fastr: A Workflow Engine for Advanced Data Flows in Medical Image Analysis*. Frontiers in ICT, 3, 15. https://doi.org/10.3389/fict.2016.00015

.. [2] Yoo, A. B., Jette, M. A., & Grondona, M. (2003). *SLURM: Simple Linux Utility for Resource Management*. Job Scheduling Strategies for Parallel Processing, Lecture Notes in Computer Science, 2862, 44–60. https://doi.org/10.1007/10968987_3

.. [3] Marcus, D. S., Olsen, T. R., Ramaratnam, M., & Buckner, R. L. (2007). *The extensible neuroimaging archive toolkit*. Neuroinformatics, 5(1), 11–33. https://doi.org/10.1385/NI:5:1:11