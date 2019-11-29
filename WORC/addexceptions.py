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

"""
This module contains all WORC-related Exceptions
"""

# import inspect
# import os
# import textwrap

# pylint: disable=too-many-ancestors
# Because fo inheriting from FastrError and a common exception causes this
# exception, even though this behaviour is desired


class WORCError(Exception):
    """
    This is the base class for all WORC related exceptions. Catching this
    class of exceptions should ensure a proper execution of WORC.
    """
    # def __init__(self, *args, **kwargs):
    #     """
    #     Constructor for all exceptions. Saves the caller object fullid (if
    #     found) and the file, function and line number where the object was
    #     created.
    #     """
    #     super(WORCError, self).__init__(*args, **kwargs)
    #
    #     frame = inspect.stack()[1][0]
    #     call_object = frame.f_locals.get('self', None)
    #     if call_object is not None and hasattr(call_object, 'fullid'):
    #         self.WORC_object = call_object.fullid
    #     else:
    #         self.WORC_object = None
    #
    #     info = inspect.getframeinfo(frame)
    #     self.filename = info.filename
    #     self.function = info.function
    #     self.linenumber = info.lineno
    #
    # def __str__(self):
    #     """
    #     String representation of the error
    #
    #     :return: error string
    #     :rtype: str
    #     """
    #     if self.WORC_object is not None:
    #         return '[{}] {}'.format(self.WORC_object, super(WORCError, self).__str__())
    #     else:
    #         return super(WORCError, self).__str__()
    #
    # def excerpt(self):
    #     """
    #     Return a excerpt of the Error as a tuple.
    #     """
    #     return type(self).__name__, self.message, self.filename, self.linenumber
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
