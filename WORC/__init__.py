# Add the path in WORC where the fastr configuration is to the fastr configs
# environment variable
import os
this_directory = os.path.dirname(os.path.abspath(__file__))
WORC_config_dir = os.path.join(this_directory, 'fastrconfig')
key = "FASTR_CONFIG_DIRS"
if key in os.environ.keys():
    os.environ["FASTR_CONFIG_DIRS"] += os.pathsep + WORC_config_dir
else:
    os.environ["FASTR_CONFIG_DIRS"] = os.pathsep + WORC_config_dir

# Actually import all WORC functions required
from WORC.WORC import WORC
from WORC import classification
from WORC import featureprocessing
from WORC import processing
from WORC import plotting
from WORC import exampledata
from .facade import SimpleWORC
