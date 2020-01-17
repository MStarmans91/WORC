# Check whether config.d folder contains WORC config: else copy
import os
import shutil

fastr_home = os.path.expanduser(os.path.join('~', '.fastr', 'config.d'))
if not os.path.exists(fastr_home):
    os.makedirs(fastr_home)

destination = os.path.join(fastr_home, 'WORC_config.py')

if not os.path.exists(destination):
    # Copy file
    current_path = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(current_path, 'fastrconfig', 'WORC_config.py')
    shutil.copyfile(source, destination)

# Actually import all WORC functions required
from WORC.WORC import WORC
from WORC import classification
from WORC import featureprocessing
from WORC import processing
from WORC import plotting
from WORC import exampledata
from .facade import SimpleWORC, BasicWORC
