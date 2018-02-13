|Build Status|

WORC v2.0.0
===========

Workflow for Optimal Radiomics Classification
---------------------------------------------

WORC is an open-source python package for the easy execution of full
Radiomics pipelines.

We aim to establish a common Radiomics platform supporting easy
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

Documentation
~~~~~~~~~~~~~

For more information, see our Github Wiki.

Alternatively, you can generate the documentation by checking out the
master branch and running from the root directory:

::

    python setup.py build_sphinx

The documentation can then be viewed in a browser by opening
``PACKAGE_ROOT\build\sphinx\html\index.html``.

Installation
============

WORC currently only supports Unix with Python 2 (>2.7.6) systems.

The package can be installed by running the setup file:

::

      python setup.py install

Make sure you install the requirements first:

::

      pip install -r requirements.txt

Several tools have some (mandatory) prerequisites which are listed
below. We highly recommend you to install these to maximally profit from
our toolbox.

Fastr Configuration
~~~~~~~~~~~~~~~~~~~

The installation will create a FASTR configuration file in the
~/.fastr/config.d folder. This file is used for configuring fastr, the
pipeline execution toolbox we use. These mounts are used to locate the
WORC tools and your inputs and outputs. Please inspect the mounts and
change them if neccesary.

Note: We use the site package to automatically find the WORC
installation directory in this file. The site package does however not
work in virtual environments. You will therefore have to change the
packagedir directory manually to the folder your WORC installation is
located.

More information can be found at `the FASTR
website <http://fastr.readthedocs.io/en/stable/static/file_description.html#config-file>`__

If you are using FASTR < 1.3.0, you need to manually add the WORC tools,
datatypes and mounts to your FASTR configuration (~/.fastr/config.py).
This concerns the following additions:

::

    # Add the WORC FASTR tools and type paths
    packagedir = site.getsitepackages()[0]
    tools_path = [os.path.join(packagedir, 'WORC', 'resources', 'fastr_tools')] + tools_path
    types_path = [os.path.join(packagedir, 'WORC', 'resources', 'fastr_types')] + types_path

    # Mounts accessible to fastr virtual file system
    mounts['worc_example_data'] = os.path.join(packagedir, 'WORC', 'exampledata')
    mounts['apps'] = os.path.expanduser(os.path.join('~', 'apps'))
    mounts['output'] = os.path.expanduser(os.path.join('~', 'WORC', 'output'))
    mounts['test'] = os.path.join(packagedir, 'WORC', 'resources', 'fastr_tests')

Note that the Python site package does not work properly in virtual
environments. You must then manually locate the packagedir.

ITK and ITK tools
~~~~~~~~~~~~~~~~~

We use the ITKtools toolbox for the conversion between different image
types, which is by default embedded in the toolbox. As ITKtools requires
you to build ITK, you will also have to do so. PATH should be equal to
your fastr.config.mounts['apps'] path.

On Linux, we provide a script for automatic installation. Simply run:
""" ./install\_ITK.sh """

On Windows/MacOSx, follow the steps below.

1. Obtain the ITK sources, compile and install

   ::

       wget http://downloads.sourceforge.net/project/itk/itk/4.10/InsightToolkit-4.10.1.tar.gz?r=https%3A%2F%2Fitk.org%2FITK%2Fresources%2Fsoftware.html&ts=1477129065&use_mirror=kent PATH/ITK/itk.tar.gz
       mkdir PATH/ITK/ITK-src/ && tar -xzf PATH/ITK/itk.tar.gz -C PATH/ITK/ITK-src/ --strip-components=1
       cd PATH/ITK/ITK-bin/
       cmake -DModule_ITKReview=ON -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF PATH/ITK/ITK-src
       make

Alternative version:

::

    git clone https://itk.org/ITK.git
    mkdir ITK-build
    cd ITK-build
    cmake ../ITK
    make

2. Obtain and build ITKtools

   ::

       mkdir PATH/itktools
       cd PATH/itktools
       wget https://github.com/ITKTools/ITKTools/archive/master.zip
       unzip master.zip
       rsync -a ITKTools-master/ 0.3.2 && rm -rf ITKTools-master
       mkdir 0.3.2/install && cd 0.3.2/install
       cmake ../src -DITK_DIR=PATH/ITK/ITK-bin/
       make

Elastix
~~~~~~~

Image registration is included in WORC through `elastix and
transformix <http://elastix.isi.uu.nl/>`__. Download the binaries and,
similar to ITKtools, place them in the fastr.config.mounts['apps'] path.
Check the elastix tool description for the correct subdirectory
structure. For example, on Linux, the binaries and libraries should be
in "../apps/elastix/4.8/install/" and "../apps/elastix/4.8/install/lib"
respectively.

XNAT
~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  FASTR (Workflow design and building)
-  xnat (Collecting data from XNAT)
-  SimpleITK (Image loading and preprocessing)
-  Pyradiomics (Feature extractor)

Also, the PREDICT(Feature extractor and classifiers) package is used,
which currently needs to be installed manually from the `PREDICT Github
repository <https://github.com/Svdvoort/PREDICTFastr>`__.

See for other requirements the `requirements file <requirements.txt>`__.

Start
-----

We provide an example script for you to get started with. Make sure you
input your own data as the sources. Also, check out the unit tests of
several tools in the WORC/resources/fastr\_tests directory. The example
is explained in more detail in the Wiki on this Github.

WIP
---

-  We are working on improving the documentation.
-  We are working on the addition of different classifiers.
-  We are working on organizing clinically relevant datasets for
   examples and unit tests.
-  We will merge to Python 3 support in the coming months, as soon as
   FASTR moves to Python 3.
-  We have some issues with installing numpy and scipy in the
   requirements. There is now a workaround implemented.

License
-------

This package is covered by the open source `APACHE 2.0
License <APACHE-LICENSE-2.0>`__.

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
