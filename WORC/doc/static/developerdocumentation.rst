Developer documentation
============

Adding a feature processing toolbox
---------------------
1. Check if config type is in fastr typelist: else, make your own
2. Make FASTR tool: take pyradiomics as example.
3. In WORC: (all in add_feature_calculator)
  a. Add converter for your config file
  b. Add a config safe function to save_config
  c. Make sure you add both sources and sinks for your tools
  d. Link these sources and sinks to the fastr network in the WORC.set function
  e. Add part to feature converter
4. Add part to feature converter tool to convert your features to WORC format.

Testing and example data
-------------------------

Facades
--------
