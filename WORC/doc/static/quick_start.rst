Quick start guide
=================

This manual will show users how to install WORC, configure WORC and construct and run simple networks.

.. _installation-chapter:

Installation
------------

You can install WORC either using pip, or from the source code.

Installing via pip
``````````````````

You can simply install WORC using ``pip``:

.. code-block:: bash

    pip install WORC

.. note:: You might want to consider installing ``WORC`` in a `virtualenv <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_


Installing from source code
```````````````````````````

To install from source code, use git via the command-line:

.. code-block:: bash

    git clone https://github.com/MStarmans91/WORC.git  # for http
    git clone ssh://git@github.com:MStarmans91/WORC.git # for ssh

.. _subsec-installing:

To install to your current Python environment, run:

.. code-block:: bash

    cd WORC/
    pip install .

This installs the scripts and packages in the default system folders. For
Windows this is the python ``site-packages`` directory for the WORC python
library. For Ubuntu this is in the ``/usr/local/lib/python3.x/dist-packages/`` folder.

.. note:: If you want to develop WORC, you might want to use ``pip install -e .`` to get an editable install

.. note:: You might want to consider installing ``WORC`` in a `virtualenv <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_


Configuration
-------------

WORC has defaults for all settings so it can be run out of the box to test the examples.
However, you may want to alter the fastr configuration to your system settings, e.g.
to locate your input and output folders and how much you want to parallelize the execution.

Fastr will search for a config file named ``config.py`` in the ``$FASTRHOME`` directory
(which defaults to ``~/.fastr/`` if it is not set). So if ``$FASTRHOME`` is set the ``~/.fastr/``
will be ignored.

For a sample configuration file and a complete overview of the options in ``config.py`` see
the :ref:`Config file <config-chapter>` section.
