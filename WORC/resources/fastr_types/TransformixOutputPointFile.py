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

import re

from fastr.datatypes import URLType


class TransformixOutputPointFile(URLType):
    description = 'Resulting point file from transformix'
    extension = 'txt'

    def _validate(self):
        if not super(TransformixOutputPointFile, self)._validate():
            return False

        try:
            with open(self.parsed_value) as fin:
                for line in fin:
                    if not re.match('Point\s+(?P<id>\d+)\s+; InputIndex = \[(?P<inindex>[\s\d]+)\]\s+; InputPoint = \[(?P<inpoint>[\s\d\.-]+)\]\s+; OutputIndexFixed = \[(?P<outindex>[\s\d]+)\]\s+; OutputPoint = \[(?P<outpoint>[\s\d\.-]+)\]\s+; Deformation = \[(?P<deform>[\s\d\.-]+)\]', line):
                        return False
        except IOError:
            return False

        return True
