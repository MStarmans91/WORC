#!/usr/bin/env python

# Copyright 2011-2020 Biomedical Imaging Group Rotterdam, Departments of
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
from sys import platform

# Go to directory of this folder.
os.chdir(os.path.dirname(os.path.abspath(__file__)))


if platform == "linux" or platform == "linux2":
    ex = 'python3'
elif platform == "win32":
    ex = 'python'

print(f'{ex} doc_clean.py')
os.system(f'{ex} doc_clean.py')
print(f'{ex} generate_modules.py ..{os.path.sep} -d .{os.path.sep} -s rst -f')
os.system(f'{ex} generate_modules.py ..{os.path.sep} -d .{os.path.sep} -s rst -f')
print(f'{ex} generate_config.py')
os.system(f'{ex} generate_config.py')

print('make html')
os.system('make html')
#os.system('make latexpdf')
