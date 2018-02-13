# Copyright 2011-2014 Biomedical Imaging Group Rotterdam, Departments of
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

from fastr.datatypes import URLType


class TransformixPointFile(URLType):
    description = 'Text file to store point coordinates'
    extension = 'txt'

    def _validate(self):
        # Special case, for no points, but entire field
        if not super(TransformixPointFile, self)._validate():
            return False

        try:
            # Extract lines
            lines = []
            with open(self.parsed_value) as fin:
                for line in fin:
                    lines.append(line)

            # Check first line is index or point
            if lines[0].strip() not in ['index', 'point']:
                return False

            # Check second line is number of points
            try:
                nrpoints = int(lines[1])
            except ValueError:
                return False

            # Check number of points in file
            if len(lines) != nrpoints + 2:
                return False
        except IOError:
            return False

        return True
