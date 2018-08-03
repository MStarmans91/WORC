#!/bin/bash

# -----------------------------------------------------------------------
# WORC Installation
# -----------------------------------------------------------------------

# Install requirements: git, cmake, build-essential
echo -e "Installing git, pip, build-essential, graphviz, ipython and jupyter notebook requirements."
apt-get -y install git python-pip build-essential graphviz ipython jupyter-core
pip install jupyter

# Clone WORC repository from Github.
echo -e "Clone WORC repository from Github."
git clone http://Github.com/MStarmans91/WORC ~/Documents/WORC

# Install WORC requirements and package itself
echo -e "Installing WORC requirements and package itself."
cd ~/Documents/WORC
pip install -r requirements.txt
python setup.py install

# Install the PREDICT requirements and package manually
echo -e "Installing PREDICT requirements and package itself."
cd ~/Documents
git clone http://Github.com/Svdvoort/PREDICTFastr PREDICT
cd PREDICT
pip install -r requirements.txt
python setup.py install
