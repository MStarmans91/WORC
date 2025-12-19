#!/usr/bin/env python

# Copyright 2016-2025 Biomedical Imaging Group Rotterdam, Departments of
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
import fastr
import importlib.util

# Find the WORC installation path
# packagedir = r"C:\Users\795023\Documents\GitHub\WORC"
spec = importlib.util.find_spec("WORC")
worc_path = os.path.dirname(spec.origin)

# Add the WORC FASTR tools and type paths
tools_path = [os.path.join(worc_path, 'resources', 'fastr_tools')] + tools_path
types_path = [os.path.join(worc_path, 'resources', 'fastr_types')] + types_path

# Mounts accessible to fastr virtual file system
mounts['worc_example_data'] = os.path.join(worc_path, 'exampledata')
mounts['apps'] = os.path.expanduser(os.path.join('~', 'apps'))
mounts['output'] = os.path.expanduser(os.path.join('~', 'WORC', 'output'))
mounts['test'] = os.path.join(worc_path, 'resources', 'fastr_tests')

# The ITKFile type requires a preferred type when no specification is given.
# We will set it to Nifti, but you may change this.
preferred_types += ["NiftiImageFileCompressed"]
