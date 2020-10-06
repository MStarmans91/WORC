#!/usr/bin/env python

# Copyright 2020 Biomedical Imaging Group Rotterdam, Departments of
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

import WORC.addexceptions as WORCexceptions
import fastr


class Inference(object):
    """Build a network that can use an existing model for inference."""

    def __init__(self, fastr_plugin='LinearExecution',
                 name='Example'):
        """
        Initialize object.

        Parameters
        ----------
        network: fastr network, default None
                If you input a network, the evaluate network is added
                to the existing network.

        """

    def create_network(self):
        """Add evaluate components to network."""
