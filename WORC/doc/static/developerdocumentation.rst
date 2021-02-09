Developer documentation
=======================

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
   :py:mod:`WORC.WORC.IOparser.config_io_classifier.load_config`.

3. Define how the parameters you just added are added to the search space
   in :py:mod:`WORC.WORC.classification.trainclassifier.trainclassifier`. For
   example, your parameters may define the actual options, or the min-max of
   a distribution (e.g. uniform, log-uniform). See the referred function
   for some examples. Make sure that the object is iterable.

4. These parameters will end up in the function fitting the workflow:
   :py:mod:`WORC.WORC.classification.fitandscore.fit_and_score`. Hence,
   add a part to that function to embed your method in the workflow. We advice
   you to embed your method in a sklearn compatible class, having init,
   fit and transform functions. See for example
   py:mod:`WORC.WORC.featureprocessing.Preprocessor.Preprocessor`.

   In
   :py:mod:`WORC.WORC.classification.fitandscore.fit_and_score`, make sure
   that after fitting your object, the parameters used are deleted from the
   config, as is done for the other methods as well.

   Lastly, in :py:mod:`WORC.WORC.classification.fitandscore.fit_and_score`,
   make sure the fitted object is returned. We recommend looking at the
   ``imputer`` object and similarly including your object.

   This is given to various objects
   in the :py:mod:`WORC.WORC.classification.SearchCV` module. Therefore,
   add the returned object to all the parts were fitted objects are used: we
   recommend looking everywhere the ``imputer`` is stated in
   :py:mod:`WORC.WORC.classification.SearchCV`, copying those five statements
   and replace ``imputer`` with however you called your methods. You can see
   that this is also similar to e.g. the ``scaler``, ``pca``, and ``groupsel``
   objects.
