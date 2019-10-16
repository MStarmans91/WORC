#!/usr/bin/env python

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

import pathlib


def clean():
    """
    Clean the autogen and main directories of .rst and .txt files
    """
    doc_path = pathlib.Path(__file__).parent
    clean_dir(doc_path)

    autogen_path = doc_path / 'autogen'
    clean_dir(autogen_path)
    clean_dir(autogen_path / 'config')


def clean_dir(directory):
    for filepath in directory.iterdir():
        if filepath.suffix in ('.rst', '.txt'):
            filepath.unlink()
            print(f'removed {filepath}')


if __name__ == '__main__':
    clean()
