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


Preprocessing
--------------

Documentation WIP.

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
