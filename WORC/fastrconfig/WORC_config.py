#!/usr/bin/env python

# Copyright 2016-2019 Biomedical Imaging Group Rotterdam, Departments of
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
import pkg_resources
import site
import sys

# Get directory in which packages are installed
working_set = pkg_resources.working_set
requirement_spec = pkg_resources.Requirement.parse('WORC')
egg_info = working_set.find(requirement_spec)
if egg_info is None:  # Backwards compatibility with WORC2
    try:
        packagedir = site.getsitepackages()[0]
    except AttributeError:
        # Inside virtualenvironment, so getsitepackages doesnt work.
        paths = sys.path
        for p in paths:
            if os.path.isdir(p) and os.path.basename(p) == 'site-packages':
                packagedir = p
else:
    packagedir = egg_info.location

# Add the WORC FASTR tools and type paths
tools_path = [os.path.join(packagedir, 'WORC', 'resources', 'fastr_tools')] + tools_path
types_path = [os.path.join(packagedir, 'WORC', 'resources', 'fastr_types')] + types_path

# Mounts accessible to fastr virtual file system
mounts['worc_example_data'] = os.path.join(packagedir, 'WORC', 'exampledata')
mounts['apps'] = os.path.expanduser(os.path.join('~', 'apps'))
mounts['output'] = os.path.expanduser(os.path.join('~', 'WORC', 'output'))
mounts['test'] = os.path.join(packagedir, 'WORC', 'resources', 'fastr_tests')

# The ITKFile type requires a preferred type when no specification is given.
# We will set it to Nifti, but you may change this.
preferred_types += ["NiftiImageFileCompressed"]
