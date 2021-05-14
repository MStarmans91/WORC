#!/usr/bin/env python

# Copyright 2016-2021 Biomedical Imaging Group Rotterdam, Departments of
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

"""
This module contains all WORC-related Exceptions
"""

# pylint: disable=too-many-ancestors
# Because fo inheriting from FastrError and a common exception causes this
# exception, even though this behaviour is desired


class WORCError(Exception):
    """
    This is the base class for all WORC related exceptions. Catching this
    class of exceptions should ensure a proper execution of WORC.
    """
    pass


class WORCNotImplementedError(WORCError, NotImplementedError):
    """
    This function/method has not been implemented on purpose (e.g. should be
    overwritten in a sub-class)
    """
    pass


class WORCIOError(WORCError, IOError):
    """
    IOError in WORC
    """
    pass


class WORCTypeError(WORCError, TypeError):
    """
    TypeError in the WORC system
    """
    pass


class WORCValueError(WORCError, ValueError):
    """
    ValueError in the WORC system
    """
    pass


class WORCKeyError(WORCError, KeyError):
    """
    KeyError in the WORC system
    """
    pass


class WORCAssertionError(WORCError, AssertionError):
    """
    AssertionError in the WORC system
    """
    pass


class WORCIndexError(WORCError, IndexError):
    """
    IndexError in the WORC system
    """
    pass
