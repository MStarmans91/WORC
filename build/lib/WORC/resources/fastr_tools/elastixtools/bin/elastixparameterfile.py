#!/usr/bin/env python
from collections import OrderedDict, MutableMapping
import re


class ElastixParameterFile(MutableMapping):
    """
    ElastixParameterFile is a dictionary object to allow access to parameter file in a style like::

    >>> params = ElastixParameterFile('/path/to/file/params.txt')
    >>> params[key].value
    >>> params[key].value = 'new_value'
    >>> params[linenr].value
    >>> params[key].comment
    >>> params[key].comment = 'This inserts/changes the comment on this line'
    >>> params[linenr].comment
    >>> params[key].linenr

    ElastixParameterFile is a MutableMapping, meaning it offers all functionality of a dict object.
    You can index both on setting key as well as line number. The returns values are ElastixParameterLine
    objects that give access to a key, value, comment and linenr. This allow easy manipulation of certain
    lines in the parameter file.
    """
    class ElastixParameterLine(list):
        def __init__(self, data):
            if len(data) != 4:
                raise ValueError(
                    'ElastixParameterLine must have 4 components!')

            if (not isinstance(data[0], int) or
                not (isinstance(data[1], (float, int, str, tuple, list)) or data[1] is None) or
                not (isinstance(data[2], str) or data[2] is None) or
                    not isinstance(data[3], str)):
                raise TypeError('ElastixParameterLine must be of types [int, list/None, str/None, str], found [%s, %s, %s, %s]' % (type(data[0]).__name__, type(data[1]).__name__, type(data[2]).__name__, type(data[3]).__name__))

            if data[1] is not None:
                if isinstance(data[1], (int, float, str)):
                    data[1] = [data[1]]

                data[1] = [self.parse_value(v) for v in data[1]]

            super(ElastixParameterFile.ElastixParameterLine, self).__init__(data)

        def __repr__(self):
            if self.value is None:
                key_val_str = ''
            elif len(self.value) > 10:
                key_val_str = '({} [...({:d} elem)...])'.format(self.key, len(self.value))
            else:
                value_str = ' '.join([self.unparse_value(v) for v in self.value])
                if len(value_str) > 15:
                    value_str = '{}...'.format(value_str[:15])

                key_val_str = '({key} {value})'.format(key=self.key, value=value_str)

            if self.comment is None:
                comment_str = ''
            elif len(self.comment) > 15:
                comment_str = '//{}...'.format(self.comment[:15])
            else:
                comment_str = '//{}'.format(self.comment)

            return '<ElastixParameterLine {nr}: {keyval}{comment}>'.format(nr=self.linenr, keyval=key_val_str, comment=comment_str)

        def __str__(self):
            if self.value is None:
                return self.comment_str
            elif self.comment is None:
                return self.key_value_str
            else:
                output = '{key_value} {comment}'.format(
                    key_value=self.key_value_str,
                    comment=self.comment_str)

                return output

        @property
        def linenr(self):
            return self[0]

        @property
        def value(self):
            return self[1]

        @value.setter
        def value(self, value):
            # Make sure value is a list
            if isinstance(value, (str, float, int)):
                value = [value]
            elif not isinstance(value, list):
                try:
                    value = list(value)
                except TypeError:
                    value = [value]

            self[1] = value

        @property
        def key_value_str(self):
            if self.value is None:
                return ''

            return '({key} {value})'.format(key=self.key, value=' '.join([self.unparse_value(v) for v in self.value]))

        @property
        def comment(self):
            return self[2]

        @comment.setter
        def comment(self, value):
            if not isinstance(value, str):
                raise TypeError('Comments have to be of str type!')

            if value[0] != ' ':
                value = ' ' + value

            self[2] = value

        @property
        def comment_str(self):
            if self.comment is None:
                return ''

            return '//{comment}'.format(comment=self.comment)

        @property
        def key(self):
            return self[3]

        @staticmethod
        def parse_value(item):
            """
            Parse a value in an elastix parameter file and cast it into the right
            type. Supported types are string, integer and float.
            """
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

        @staticmethod
        def unparse_value(item):
            """
            Parse a python variable into the format used in an elastix parameter
            file. Supported types are string, integer and float.
            """
            if isinstance(item, str):
                item = '"%s"' % item
            else:
                item = str(item)

            return item

    """
    Create a ElastixParameterFile object. If filename is given the file is read
    and parsed into the object.
    """
    _re_line_parse = re.compile(r'\s*(\(.+\s+.*\))?\s*(//.*)?')
    _re_keyvalue_parse = re.compile(r'\((\S+)\s+(.*)\)')
    _re_value_parse = re.compile(r'(".*?"|-?\d+\.?\d*)')

    def __init__(self, filename=None):
        self._data = OrderedDict()

        if filename is not None:
            self.parse(filename)

    def parse(self, filename):
        """Load an elastrix parameter file and parse it into the ElastixParameterFile object.
        """
        self._data = OrderedDict()

        if filename is not None:
            with open(filename, 'r') as input_file:
                for linenr, line in enumerate(input_file):
                    match_obj = re.match(self._re_line_parse, line)

                    if match_obj.group(1) is None:
                        key = 'dummy_%d' % linenr
                        value = None
                    else:
                        submatch = re.match(self._re_keyvalue_parse, match_obj.group(1))
                        key = submatch.group(1)
                        value = re.findall(self._re_value_parse, submatch.group(2))

                    if match_obj.group(2) is None:
                        comment = None
                    else:
                        comment = match_obj.group(2)[2:]  # Strip of //

                    self._data[key] = self.ElastixParameterLine([linenr + 1, value, comment, key])

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._data[key]
        elif isinstance(key, int):
            out = next((value for value in self._data.itervalues() if value[0] == key), None)
            if out is None:
                raise KeyError('Line not found in parameter file')

            return out
        else:
            raise TypeError('Key is of incorrect type!')

    def __setitem__(self, key, value):
        comment = None

        if not isinstance(key, str):
            key = self[key].key

        # If we get a ElastixParameterLine, insert as is
        if isinstance(value, self.ElastixParameterLine):
            # Unpack the desired fields
            comment = value.comment
            value = value.value

        # Check to update value or insert a new line
        if key in self._data:
            self._data[key].value = value
        else:
            if not isinstance(value, list):
                value = [value]
            line = len(self._data) + 1
            self._data[key] = self.ElastixParameterLine([line, value, comment, key])

    def __delitem__(self, key):
        if isinstance(key, str):
            del self._data[key]
        else:
            # Try to get a proper str key
            key = self[key].key
            del self._data[key]

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return '<ElastixParameterFile containing {:d} lines>'.format(len(self._data))

    def __str__(self):
        output = '\n'.join([str(v) for v in self._data.values()])

        return output

    def addline(self, key=None, value=None, comment=None):
        if key is None and value is not None:
            raise ValueError('Adding a value with a dummy key seems like a BAD idea!')

        if key is None:
            key = 'dummy_{}'.format(len(self))

        if key in self._data:
            raise ValueError('Cannot add a line with an already existing key!')

        linenr = len(self) + 1

        self._data[key] = self.ElastixParameterLine([linenr, value, comment, key])

    """
    Write the ElastrixParameterFile object back to an actual file on the disk.
    """
    def write(self, filename):
        with open(filename, 'w') as outfile:
            outfile.write(str(self))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', '-i', metavar='input_transform.txt', required=True, help='the elastix parameter file to modify')
    parser.add_argument('--outfile', '-o', metavar='output_transform.txt', required=True, help='the resulting elastix parameter file')
    parser.add_argument('--set', '-s', metavar='key=val', nargs="*", help='the items to override in the parameter file')
    parser.add_argument('--verbose', '-v', action="store_true", help='use verbose output')

    args = parser.parse_args()

    input_file = args.infile
    output_file = args.outfile
    if args.set is not None:
        substitutions = OrderedDict([x.split('=', 1) for x in args.set])
    else:
        substitutions = OrderedDict()
    VERBOSE = args.verbose

    if VERBOSE:
        print('Loading input {}'.format(input_file))

    params = ElastixParameterFile(input_file)

    for key, value in substitutions.items():
        substitutions[key] = [params.ElastixParameterLine.parse_value(x) for x in value.split(',')]

    if VERBOSE:
        print('Substituting values')
        for key, value in substitutions.items():
            print('Set "{}" to {}'.format(key, value))

    params.update(substitutions)

    if VERBOSE:
        print('Writing result to {}'.format(output_file))

    params.write(output_file)
