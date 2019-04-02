|Build Status|

WORC v2.1.2
===========

Workflow for Optimal Radiomics Classification
---------------------------------------------

WORC is an open-source python package for the easy execution of full
Radiomics pipelines.

We aim to establish a general Radiomics platform supporting easy
integration of other tools. With our modular build and support of
different software languages (python, MATLAB, ruby, java etc.), we want
to facilitate and stimulate collaboration, standardisation and
comparison of different Radiomics approaches. By combining this in a
single framework, we hope to find a universal Radiomics strategy that
can address various problems.

Disclaimer
----------

This package is still under development. We try to thoroughly test and
evaluate every new build and function, but bugs can off course still
occur. Please contact us through the channels below if you find any and
we will try to fix them as soon as possible, or create an issue on this
Github.

Tutorial
--------

The WORC tutorial is hosted in a `separate
repository <https://github.com/MStarmans91/WORCTutorial>`__.

Documentation
-------------

For more information, see our the Wiki on this Github.

Alternatively, you can generate the documentation by checking out the
master branch and running from the root directory:

::

    python setup.py build_sphinx

The documentation can then be viewed in a browser by opening
``PACKAGE_ROOT\build\sphinx\html\index.html``.

Installation
------------

WORC currently only supports Unix with Python 2 (>2.7.6) systems.
Windows is not supported, although WORC can still work under windows.

Please first install PREDICT:

::

      pip install PREDICT

The package can be installed through pip:

::

      pip install WORC

Alternatively, you can directly install WORC from this repository:

::

      python setup.py install

Make sure you install the requirements first:

::

      pip install -r requirements.txt

Several tools have some (mandatory) prerequisites which are listed
below. We highly recommend you to install these to maximally profit from
our toolbox.

PREDICT
~~~~~~~

Most of the default tools in WORC use
`PREDICT <https://github.com/Svdvoort/PREDICTFastr>`__, our in-house
feature extraction and classification toolbox. Currently, you do need to
manually install PREDICT from the Github or with pip:

::

      pip install PREDICT

Fastr Configuration
~~~~~~~~~~~~~~~~~~~

The installation will create a FASTR configuration file in the
$HOME/.fastr/config.d folder. This file is used for configuring fastr,
the pipeline execution toolbox we use. More information can be found at
`the FASTR
website <http://fastr.readthedocs.io/en/stable/static/file_description.html#config-file>`__.
In this file, so called mounts are defined, which are used to locate the
WORC tools and your inputs and outputs. Please inspect the mounts and
change them if neccesary.

Only if you are using FASTR < 1.3.0, you need to manually add the WORC
tools, datatypes and mounts to your FASTR configuration
($HOME/.fastr/config.py). This concerns the following additions:

Optional: Graphviz
~~~~~~~~~~~~~~~~~~

WORC can draw the network and save it as a SVG image using
`graphviz <https://www.graphviz.org/>`__. In order to do so, please make
sure you install graphviz:

::

      apt install graphiv

Optional: Elastix
~~~~~~~~~~~~~~~~~

Image registration is included in WORC through `elastix and
transformix <http://elastix.isi.uu.nl/>`__. In order to use elastix,
please download the binaries and place them in your
fastr.config.mounts['apps'] path. Check the elastix tool description for
the correct subdirectory structure. For example, on Linux, the binaries
and libraries should be in "../apps/elastix/4.8/install/" and
"../apps/elastix/4.8/install/lib" respectively.

Note: optionally, you can tell WORC to copy the metadata from the image
file to the segmentation file before applying the deformation field.
This requires ITK and ITKTools: see the `Install\_ITK
file <Install_ITK.md>`__ for installation instructions. More info on
using the copying of metadata can be found on our Github Wiki.

Optional: XNAT
~~~~~~~~~~~~~~

We use the XNATpy package to connect the toolbox to the XNAT online
database platforms. You will only need this when you want to download or
upload data from or to XNAT. We advise you to specify your account
settings in a .netrc file when using this feature, such that you do not
need to input them on every request:

::

    echo "machine images.xnat.org
    >     login admin
    >     password admin" > ~/.netrc
    chmod 600 ~/.netrc

3rd-party packages used in WORC:
--------------------------------

-  FASTR (Workflow design and building)
-  xnat (Collecting data from XNAT)
-  SimpleITK (Image loading and preprocessing)
-  `Pyradiomics <https://github.com/Radiomics/pyradiomics>`__
-  Our in-house package
   `PREDICT <https://github.com/Svdvoort/PREDICTFastr>`__

See for other requirements the `requirements file <requirements.txt>`__.

Start
-----

We suggest you start with the `WORC
Tutorial <https://github.com/MStarmans91/WORCTutorial>`__. Besides a
Jupter notebook with instructions, we provide there also an example
script for you to get started with. Make sure you input your own data as
the sources. Also, check out the unit tests of several tools in the
WORC/resources/fastr\_tests directory. The example is explained in more
detail in the Wiki on this Github.

WIP
---

-  We are working on improving the documentation.
-  We are working on organizing clinically relevant datasets for
   examples and unit tests.
-  We will merge to Python 3 support in the coming months (April 2019),
   as soon as FASTR moves to Python 3.

License
-------

This package is covered by the open source `APACHE 2.0
License <APACHE-LICENSE-2.0>`__.

When using WORC, please cite this repository.

Contact
-------

We are happy to help you with any questions. Please contact us on the
`WORC google
group <https://groups.google.com/forum/#!forum/worc-users>`__.

We welcome contributions to WORC. We will soon make some guidelines. For
the moment, converting your toolbox into a FASTR tool will be
satisfactory.

.. |Build Status| image:: https://travis-ci.com/MStarmans91/WORC.svg?token=qyvaeq7Cpwu7hJGB98Gp&branch=master
   :target: https://travis-ci.com/MStarmans91/WORC
