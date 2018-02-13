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

import os
import re
from fastr.datatypes import URLType


class ElastixTransformFile(URLType):
    _re_line_parse = re.compile(r'\s*(\(.+\s+.*\))?\s*(//.*)?')
    _re_keyvalue_parse = re.compile(r'\((\S+)\s+(.*)\)')
    _re_value_parse = re.compile(r'(".*?"|-?\d+\.?\d*)')

    description = 'Elastix Transform parameter file'
    extension = 'txt'

    def __eq__(self, other):
        if type(other) is not ElastixTransformFile:
            return NotImplemented

        return self._parse() == other._parse()

    def _parse(self, filename=None):
        data = dict()

        if filename is None:
            filename = self.parsed_value

        with open(filename, 'r') as input_file:
            for linenr, line in enumerate(input_file):
                match_obj = re.match(self._re_line_parse, line)

                if match_obj.group(1) is not None:
                    submatch = re.match(self._re_keyvalue_parse, match_obj.group(1))
                    key = submatch.group(1)
                    value = re.findall(self._re_value_parse, submatch.group(2))

                data[key] = tuple(self._parse_value(x) for x in value)

        initial_field = 'InitialTransformParametersFileName'
        if initial_field in data:
            initial_filename = data[initial_field][0]

            if initial_filename != 'NoInitialTransform':
                if initial_filename[0] not in '\\/':
                    initial_filename = os.path.join(os.path.dirname(filename), initial_filename)
                data[initial_field] = self._parse(initial_filename)

        return data

    def _parse_value(self, item):
        if item is None:
            return None

        if isinstance(item, (int, float)):
            return item

        if item[0] == '"' and item[-1] == '"':
            item = item[1:-1]
        else:
            try:
                item = int(item)
            except ValueError:
                try:
                    item = float(item)
                except ValueError:
                    item = str(item)

        return item
