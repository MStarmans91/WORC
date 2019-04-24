#!/usr/bin/env python
import os

import fastr
from fastr.helpers.rest_generation import create_rest_table


def generate_config(fields):
    field_names = []
    field_types = []
    field_defaults = []
    field_description = []

    for name, field in sorted(fields.items()):
        print(f'[generate_config.py] Documenting field {name}')
        field_names.append(name)
        field_types.append(field[0].__name__)
        field_description.append(field[2])

        if len(field) > 3:
            field_defaults.append(field[3] if isinstance(field[3], str) else repr(field[3]))
        else:
            field_defaults.append(repr(field[1]))

    data = [field_names, field_types, field_description, field_defaults]
    headers = ['name', 'type', 'description', 'default']

    return create_rest_table(data, headers)


def generate_config_doc():
    print('[generate_config.py] Generating config reference...')
    filename = os.path.join(os.path.dirname(__file__), 'autogen', 'fastr.config.rst')

    with open(filename, 'w') as fh_out:
        fh_out.write(generate_config(fastr.config.DEFAULT_FIELDS))

    print(f'[generate_config.py] Config reference saved to {filename}')


if __name__ == '__main__':
    generate_config_doc()
