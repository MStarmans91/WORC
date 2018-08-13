### ITK and ITK tools
ITK can be used for several image processing operations, including the copying of metadata from one file to another.
WORC optionally can use this in Elastix.

As ITKtools requires you to build ITK, you will also have to do so. PATH should be equal to your fastr.config.mounts['apps'] path.

On Linux, we provide a script for automatic installation. Simply run:
"""
./install_ITK.sh
"""

On Windows/MacOSx, follow the steps below.

1. Obtain the ITK sources, compile and install
```
wget http://downloads.sourceforge.net/project/itk/itk/4.10/InsightToolkit-4.10.1.tar.gz?r=https%3A%2F%2Fitk.org%2FITK%2Fresources%2Fsoftware.html&ts=1477129065&use_mirror=kent PATH/ITK/itk.tar.gz
mkdir PATH/ITK/ITK-src/ && tar -xzf PATH/ITK/itk.tar.gz -C PATH/ITK/ITK-src/ --strip-components=1
cd PATH/ITK/ITK-bin/
cmake -DModule_ITKReview=ON -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF PATH/ITK/ITK-src
make
```

Alternative version:
```
git clone https://itk.org/ITK.git
mkdir ITK-build
cd ITK-build
cmake ../ITK
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
