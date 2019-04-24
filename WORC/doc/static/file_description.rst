Resource File Formats
=====================

This chapter describes the various files fastr uses. The function and format
of the files is described allowing the user to configure fastr and add
DataTypes and Tools.

.. _config-file:

Config file
-----------

Fastr reads the config files from ``$FASTRHOME/config.py`` by default. If the
``$FASTRHOME`` environment variable is not set it will default to ``~/.fastr``.
As a result it read:

* ``$FASTRHOME/config.py`` (if environment variable set)
* ``~/.fastr/config.py`` (otherwise)

Reading a new config file change or override settings, making the last config
file read have the highest priority. All settings have a default value, making
config files and all settings within optional.

.. note::
    To verify which config files have been read you can see
    ``fastr.config.read_config_files`` which contains a list
    of the read config files (in read order).

.. note::
    If ``$FASTRHOME`` is set, ``$FASTRHOME/tools`` is automatically added
    as a tool directory if it exists and ``$FASTRHOME/datatypes`` is automatically
    added as a type directory if it exists.

Splitting up config files
^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes it is nice to have config files split in multiple smaller files. Next
to the ``config.py`` you can also created a directory ``config.d`` and all
``.py`` files in this directory will be sourced in alphabetical order.

Given the following layout of the ``$FASTRHOME`` directory::

    ./config.d/a.py
    ./config.d/b.txt
    ./config.d/c.py
    ./config.py

The following files will be read in order:

1. ``./config.py``
2. ``./config.d/a.py``
3. ``./config.d/c.py``

Example config file
^^^^^^^^^^^^^^^^^^^

Here is a minimal config file::

  # Enable debugging output
  debug = False

  # Define the path to the tool definitions
  tools_path = ['/path/to/tools',
                '/path/to/other/tools'] + tools_path
  types_path = ['/path/to/datatypes',
                '/path/to/other/datatypes'] + types_path


  # Specify what your preferred output types are.
  preferred_types += ["NiftiImageFileCompressed",
                      "NiftiImageFile"]

  # Set the tmp mount
  mounts['tmp'] = '/path/to/tmpdir'


Format
^^^^^^

The config file is actually a python source file. The next syntax applies to
setting configuration values::

    # Simple values
    float_value = 1.0
    int_value = 1
    str_value = "Some value"
    other_str_value = 'name'.capitalize()

    # List-like values
    list_value = ['over', 'ride', 'values']
    other_list_value.prepend('first')
    other_list_value.append('list')

    # Dict-like values
    dict_value = {'this': 1, 'is': 2, 'fixed': 3}
    other_dict_value['added'] = 'this key'

.. note:: Dictionaries and list always have a default, so you can always append
          or assign elements to them and do not have to create them in a config
          file. Best practice is to only edit them unless you really want to
          block out the earliers config files.

Most operations will be assigning values, but for list and dict values
a special wrapper object is used that allows manipulations from the default.
This limits the operations allowed.

List values in the ``config.py`` have the following supported operators/methods:

* ``+``, ``__add__`` and ``__radd__``
* ``+=`` or ``__iadd__``
* ``append``
* ``prepend``
* ``extend``

Mapping (dict-like) values in the ``config.py`` have the following supported operators/methods:

* ``update``
* ``[]`` or ``__getitem__``, ``__setitem__`` and ``__delitem__``

Configuration fields
^^^^^^^^^^^^^^^^^^^^

This is a table the known config fields on the system:

.. include:: ../autogen/fastr.config.rst


:py:class:`Tool <fastr.core.tool.Tool>` description
---------------------------------------------------

.. _tool-schema:

:py:class:`Tools <fastr.core.tool.Tool>` are the building blocks in the fastr network. To add new
:py:class:`Tools <fastr.core.tool.Tool>` to fastr, XML/json files containing a :py:class:`Tool <fastr.core.tool.Tool>`
definition can be added. These files have the following layout:

+-------------------------------------------------+--------------------------------------------------------------------------------+
| Attribute                                       | Description                                                                    |
+=================================================+================================================================================+
| ``id``                                          | The id of this Tool (used internally in fastr)                                 |
+---------------+---------------------------------+--------------------------------------------------------------------------------+
| ``name``      |                                 | The name of the Tool, for human readability                                    |
+---------------+---------------------------------+--------------------------------------------------------------------------------+
| ``version``   |                                 | The version of the Tool wrapper (not the binary)                               |
+---------------+---------------------------------+--------------------------------------------------------------------------------+
| ``url``       |                                 | The url of the Tool wrapper                                                    |
+---------------+---------------------------------+--------------------------------------------------------------------------------+
| ``authors[]`` |                                 | List of authors of the Tools wrapper                                           |
|               +---------------------------------+--------------------------------------------------------------------------------+
|               | ``name``                        | Name of the author                                                             |
|               +---------------------------------+--------------------------------------------------------------------------------+
|               | ``email``                       | Email address of the author                                                    |
|               +---------------------------------+--------------------------------------------------------------------------------+
|               | ``url``                         | URL of the website of the author                                               |
+---------------+---------------------------------+--------------------------------------------------------------------------------+
| ``tags``      | ``tag[]``                       | List of tags describing the Tool                                               |
+---------------+---------------------------------+--------------------------------------------------------------------------------+
| ``command``   |                                 | Description of the underlying command                                          |
|               +---------------------------------+--------------------------------------------------------------------------------+
|               | ``version``                     | Version of the tool that is wrapped                                            |
|               +---------------------------------+--------------------------------------------------------------------------------+
|               | ``url``                         | Website where the tools that is wrapped can be obtained                        |
|               +---------------+-----------------+--------------------------------------------------------------------------------+
|               | ``targets[]`` |                 | Description of the target binaries/script of this Tool                         |
|               |               +-----------------+--------------------------------------------------------------------------------+
|               |               | ``os``          | OS targeted (windows, linux, macos or * (for any)                              |
|               |               +-----------------+--------------------------------------------------------------------------------+
|               |               | ``arch``        | Architecture targeted 32, 64 or * (for any)                                    |
|               |               +-----------------+--------------------------------------------------------------------------------+
|               |               | ...             | Extra variables based on the target used, see :ref:`Targets <target-ref>`      |
|               +---------------+-----------------+--------------------------------------------------------------------------------+
|               | ``description``                 | Description of the Tool                                                        |
|               +---------------------------------+--------------------------------------------------------------------------------+
|               | ``license``                     | License of the Tool, either full license or a clear name (e.g. LGPL, GPL v2)   |
|               +---------------+-----------------+--------------------------------------------------------------------------------+
|               | ``authors[]`` |                 | List of authors of the Tool (not the wrapper!)                                 |
|               |               +-----------------+--------------------------------------------------------------------------------+
|               |               | ``name``        | Name of the authors                                                            |
|               |               +-----------------+--------------------------------------------------------------------------------+
|               |               | ``email``       | Email address of the author                                                    |
|               |               +-----------------+--------------------------------------------------------------------------------+
|               |               | ``url``         | URL of the website of the author                                               | 
+---------------+---------------+-----------------+--------------------------------------------------------------------------------+
| ``interface``                                   | The interface definition see :ref:`Interfaces <interface-ref>`                 |
+-------------------------------------------------+--------------------------------------------------------------------------------+
| ``help``                                        | Help text explaining the use of the Tool                                       |
+-------------------------------------------------+--------------------------------------------------------------------------------+
| ``cite``                                        | Bibtext of the Citation(s) to reference when using this Tool for a publication |
+-------------------------------------------------+--------------------------------------------------------------------------------+


