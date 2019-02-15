# Copyright 2011-2017 Biomedical Imaging Group Rotterdam, Departments of
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
import sys
import subprocess
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.test import test as TestCommand
from setuptools import setup, find_packages

with open('requirements.txt', 'r') as fh:
    _requires = fh.read().splitlines()

with open('README.rst', 'r') as fh:
    _description = fh.read()

with open('version', 'r') as fh:
    __version__ = fh.read().splitlines()[0]

with open('test_requirements.txt', 'r') as fh:
    _tests_require = fh.read().splitlines()

with open('requirements-setup.txt', 'r') as fp:
    setup_requirements = list(filter(bool, (line.strip() for line in fp)))


def scan_dir(path, prefix=None):
    if prefix is None:
        prefix = path

    # Scan resources package for files to include
    file_list = []
    for root, dirs, files in os.walk(path):
        # Strip this part as setup wants relative directories
        root = root.replace(prefix, '')
        root = root.lstrip('/\\')

        for filename in files:
            if filename[0:8] == '__init__':
                continue
            file_list.append(os.path.join(root, filename))

    return file_list


# Determine the extra resources and scripts to pack
resources_list = scan_dir('./WORC/resources')

print('[setup.py] called with: {}'.format(' '.join(sys.argv)))
if hasattr(sys, 'real_prefix'):
    print('[setup.py] Installing in virtual env {} (real prefix: {})'.format(sys.prefix, sys.real_prefix))
else:
    print('[setup.py] Not inside a virtual env!')


# Set the entry point
entry_points = {
    "console_scripts": [
        "WORC = WORC.WORC:main",
    ]
}

# Determine the fastr config path
USER_DIR = os.path.expanduser(os.path.join('~', '.fastr'))
config_d = os.path.join(USER_DIR, 'config.d')


class NoseTestCommand(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # Run nose ensuring that argv simulates running nosetests directly
        import nose
        nose.run_exit(argv=['nosetests'])


setup(
    name='WORC',
    version='2.1.1',
    description='Workflow for Optimal Radiomics Classification.',
    long_description=_description,
    url='https://github.com/MStarmans91/WORC',
    author='M. Starmans',
    author_email='m.starmans@erasmusmc.nl',
    license='Apache License, Version 2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Education',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: System :: Distributed Computing',
        'Topic :: System :: Logging',
        'Topic :: Utilities',
    ],
    keywords='bioinformatics radiomics features',
    packages=['WORC',
              'WORC.exampledata',
              'WORC.resources',
              'WORC.IOparser',
              'WORC.processing'],
    include_package_data=True,
    package_data={'fastr.resources': resources_list,
                  'WORC': ['versioninfo'],
                  # If any package contains *.ini files, include them
                  'src': ['IOparser/*.ini']},
    data_files=[(config_d, ['WORC/fastrconfig/WORC_config.py'])],
    install_requires=_requires,
    tests_require=_tests_require,
    test_suite='nose.collector',
    cmdclass={'test': NoseTestCommand},
    entry_points=entry_points,
    setup_requires=_requires
)
