# Copyright 2011-2022 Biomedical Imaging Group Rotterdam, Departments of
# Medical Informatics and Radiology, Erasmus MC, Rotterdam, The Netherlands
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pandas as pd
from tables.hdf5extension import HDF5ExtError
from fastr.datatypes import URLType


class HDF5(URLType):
    description = 'Pandas HDF5 file'
    extension = 'hdf5'
    
    def _validate(self):
        # Function to validate the filetype
        parsed_value = self.parsed_value

        if self.extension and not parsed_value.endswith(self.extension):
            return False

        if not os.path.isfile(parsed_value):
            return False

        try:
            # Read the file and extract features
            data = pd.read_hdf(parsed_value)
            return True
        
        except HDF5ExtError:
            # Not a valid hdf5 file
            return False
        
