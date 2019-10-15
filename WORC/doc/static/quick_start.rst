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

.. note:: You might want to consider installing ``WORC`` in a
    `virtualenv <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_

Windows installation
````````````````````

On Windows, we strongly recommend to install python through the
`Anaconda distribution <https://www.anaconda.com/distribution/#windows>`_.

Regardless of your installation, you will need `Microsoft Visual Studio <https://visualstudio.microsoft.com/vs/features/python>`_: the Community
edition can be downloaded and installed for free.

If you still get an error similar to error: ``Microsoft Visual C++ 14.0 is required. Get it with``
`Microsoft Visual C++ Build Tools   <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019>`_
, please follow the respective link and install the requirements.


Hello World and Tutorials
-------------------------

To start out using WORC, we recommend you to follow the tutorials located in the
`WORCTutorial Github <https://github.com/MStarmans91/WORCTutorial/>`_. This repository
contains tutorials for an introduction to WORC, as well as more advanced workflows.

If you run into any issue, you can first debug your network using
`the fastr trace tool <https://fastr.readthedocs.io/en/stable/static/user_manual.html#debugging-a-network-run-with-errors/>`_.
If you're stuck, feel free to post an issue on the `WORC Github <https://github.com/MStarmans91/WORC/>`_.
