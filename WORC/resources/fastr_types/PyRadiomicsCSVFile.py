# Copyright 2020-2020 Biomedical Imaging Group Rotterdam, Departments of
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
from fastr.datatypes import URLType


class PyRadiomicsCSVFile(URLType):
    description = 'PyRadiomics CSV file'
    extension = 'csv'

    def _validate(self):
        # Function to validate the filetype
        parsed_value = self.parsed_value

        if self.extension and not parsed_value.endswith(self.extension):
            return False

        if not os.path.isfile(parsed_value):
            return False

        try:
            # Read the file and extract features
            data = pd.read_csv(parsed_value)
            nrows, ncols = data.shape

            if ncols == 2:
                # Only mask and segmentation key, no valid feature file

                # As PyRadiomics cannot handle multiple exuctions well,
                # delete the file
                os.remove(parsed_value)

                return False

            if nrows != 1:
                # Ran multiple times, appended features, which is invalid

                # As PyRadiomics cannot handle multiple exuctions well,
                # delete the file
                os.remove(parsed_value)

                return False

            if ncols == 0 or nrows == 0:
                # No information, so invalid file

                # As PyRadiomics cannot handle multiple exuctions well,
                # delete the file
                os.remove(parsed_value)

                return False

            # No errors, so file is valid
            return True

        except OSError:
            # Not a valid feature file
            return False
