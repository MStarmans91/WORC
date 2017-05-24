[![Build Status](https://travis-ci.com/MStarmans91/WORC.svg?token=qyvaeq7Cpwu7hJGB98Gp&branch=master)](https://travis-ci.com/MStarmans91/WORC)

# WORC v1.0.0

## Workflow for Optimal Radiomics Classification

This is an open-source python package for the easy execution of full Radiomics pipelines.

We aim to establish a common Radiomics platform supporting easy integration of other tools. With our modular build
and support of different software languages (python, MATLAB, ruby, java etc.), we want to facilitate and stimulate
collaboration and comparison of different Radiomics approaches. By combining this in a single framework,
we hope to find a universal Radiomics strategy that can address various problems.

## Disclaimer
This package is under heavy development. We try to thoroughly test and evaluate every new build and function, but
bugs can off course still occur. Please contact us through the channels below if you find any and we will try to fix
them as soon as possible.

### Documentation

For more information, see the sphinx generated documentation available [here](http://worc.readthedocs.io/).

Alternatively, you can generate the documentation by checking out the master branch and running from the root directory:

    python setup.py build_sphinx

The documentation can then be viewed in a browser by opening `PACKAGE_ROOT\build\sphinx\html\index.html`.

### Installation

WORC has currently only been tested on Unix with Python 2.7.
The package can be installed through pip:

      pip install WORC

The installation will create a FASTR configuration file in the ~/.fastr/config.d folder. Please inspect the mounts and change if neccesary.
More information can be found at [the FASTR website](http://fastr.readthedocs.io/en/stable/static/file_description.html#config-file)

Several tools have some prerequisites which are listed below. We highly recommend you to install these to
maximally profit from our toolbox.

# ITK and ITK tools
We use the ITKtools toolbox for the conversion between different image types, which is by default embedded in the toolbox.
As ITKtools requires you to build ITK, you should do this first. PATH should be equal to your fastr.config.mounts['apps'] path.


1. Obtain the ITK sources, compile and install
```
wget http://downloads.sourceforge.net/project/itk/itk/4.10/InsightToolkit-4.10.1.tar.gz?r=https%3A%2F%2Fitk.org%2FITK%2Fresources%2Fsoftware.html&ts=1477129065&use_mirror=kent PATH/ITK/itk.tar.gz
mkdir PATH/ITK/ITK-src/ && tar -xzf PATH/ITK/itk.tar.gz -C PATH/ITK/ITK-src/ --strip-components=1
cd PATH/ITK/ITK-bin/
cmake -DModule_ITKReview=ON -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF PATH/ITK/ITK-src
make
```

2.  Obtain and build ITKtools
```
mkdir PATH/itktools
cd PATH/itktools
wget https://github.com/ITKTools/ITKTools/archive/master.zip
unzip master.zip
rsync -a ITKTools-master/ 0.3.2 && rm -rf ITKTools-master
mkdir 0.3.2/install && cd 0.3.2/install
cmake ../src -DITK_DIR=PATH/ITK/ITK-bin/
make
```

# Elastix
Image registration is included in WORC through [elastix and transformix](http://elastix.isi.uu.nl/). Download the binaries and,
similar to ITKtools, place them in the fastr.config.mounts['apps'] path. Check the elastix tool description for the correct
subdirectory structure. For example, on Linux, the binaries and libraries should be in "../apps/elastix/4.8/install/"  and
"../apps/elastix/4.8/install/lib" respectively.

# XNAT
We use the XNATpy package to connect the toolbox to XNAT online database platforms. We advise you to specify
your account settings in a .netrc file when using this feature,  such that you do not need to input them on every request:

```
echo "machine images.xnat.org
>     login admin
>     password admin" > ~/.netrc
chmod 600 ~/.netrc
```

# FASTR
If you are using FASTR < 1.3.0, you need to manually add the WORC tools, datatypes and mounts to your FASTR configuration (~/.fastr/config.py). Check the fastrconfig/config.py in this repository for the necessary additions.

### 3rd-party packages used in WORC:

 - FASTR (Workflow design and building)
 - PREDICT(Feature extractor and classifiers)
 - xnat (Collecting data from XNAT)
 - SimpleITK (Image loading and preprocessing)
 - Pyradiomics (Feature extractor)

See also the [requirements file](requirements.txt).

### WIP
- We are working on improving the documentation.
- We are working on the addition of different classifiers.
- Examples and unit tests will be added.

### License
This package is covered by the open source [APACHE 2.0 License](APACHE-LICENSE-2.0).

### Contact
We are happy to help you with any questions. Please contact us on the [WORC email list](https://groups.google.com/forum/#!forum/worc-users).

We welcome contributions to WORC. We will soon make some guidelines. For the moment, converting your toolbox into FASTR
will be satisfactory.
