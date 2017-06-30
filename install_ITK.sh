#!/bin/sh
N_cores=6

# Installation of ITK
cd ~
mkdir apps
cd apps
mkdir ITK
cd ITK

git clone https://itk.org/ITK.git ITK-src
cd ITK-src
git checkout tags/v4.10.1
cd ..
mkdir ITK-build
cd ITK-build
cmake -DModule_ITKReview=ON -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF ../ITK-src
make -j $N_cores

# Installation of ITK tools
cd ~/apps
git clone https://github.com/ITKTools/ITKTools.git 0.3.2/
cd 0.3.2/
git checkout tags/v0.3.2
mkdir install/
cd install/
cmake ../src/ -DITK_DIR=../../ITK-build
make -j $N_cores
cd ..







