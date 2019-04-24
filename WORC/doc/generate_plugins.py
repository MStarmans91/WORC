#!/usr/bin/env python
import os
import sys
import textwrap
from fastr.helpers.rest_generation import create_rest_table


if hasattr(sys, 'real_prefix'):
    print('[generate_plugins.py] Inside virtual env: {}'.format(sys.prefix))
else:
    print('[generate_plugins.py] Not inside a virtual env!')

# Add the fastr top level directory for importing without an install
#fastr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
#print('Source fastr from: {}'.format(fastr_dir))
#sys.path = [fastr_dir] + sys.path

import fastr


def generate_plugins(plugin_header, plugin_manager, plugin_type_name):
    plugin_type_id = plugin_type_name.lower()

    # The initial document
    doc = [plugin_header, None]
    table = []

    # Extract data from all IOPlugins
    print('[generate_plugins.py] Found Plugins: {}'.format(list(plugin_manager.keys())))
    for scheme, plugin in sorted(plugin_manager.items()):
        if isinstance(plugin, fastr.abc.baseplugin.Plugin):
            plugin = type(plugin)

        plugin_name = plugin.__name__
        underline = '^' * len(plugin_name)
        refname = '.. _{}-{}:'.format(plugin_type_id, plugin_name)
        docstring = plugin.__doc__
        if docstring is None:
            docstring = 'NOT DOCUMENTED!'

        docstring = textwrap.dedent(docstring)

        if len(plugin.configuration_fields) > 0:
            fields = plugin.configuration_fields

            field_names = list(fields.keys())
            field_types, field_defaults, field_description = list(zip(*list(fields.values())))

            # Convert fields to str
            field_types = [x.__name__ for x in field_types]
            field_defaults = [repr(x) for x in field_defaults]
            field_description = list(field_description)

            docstring += "\n\n\n**Configuration fields**\n\n"
            docstring += create_rest_table(data=[field_names,
                                                 field_types,
                                                 field_description,
                                                 field_defaults],
                                           headers=['name',
                                                    'type',
                                                    'description',
                                                    'default'])

        table.append((scheme, plugin_name))
        doc.append('{}\n\n{}\n{}\n\n{}\n'.format(refname,
                                                 plugin_name,
                                                 underline,
                                                 docstring.strip()))

    # Make links for plugins
    table = [(x[0], ':ref:`{0} <{1}-{0}>`'.format(x[1], plugin_type_id)) for x in table]

    table = create_rest_table(
        data=list(zip(*table)),
        headers=['scheme', ':py:class:`{name} <{mod}.{name}>`'.format(name=plugin_type_name,
                                                                      mod='fastr.plugins')]
    )

    doc[1] = table + '\n'

    # Join everything together
    doc = '\n'.join(doc)
    filename = os.path.join(os.path.dirname(__file__), 'autogen', 'fastr.ref.{}s.rst'.format(plugin_type_name.lower()))
    print('[generate_plugins.py] Writing {} reference to {} ({})'.format(plugin_type_name, filename, os.path.abspath(filename)))
    with open(filename, 'w') as output:
        output.write(doc)

    return filename


def find_parents(cls):
    if isinstance(cls, fastr.abc.baseplugin.Plugin):
        cls = type(cls)

    if not issubclass(cls, fastr.abc.baseplugin.Plugin):
        # Not a subclass of Plugin (other parent)
        print('[generate_plugins.py] Ignoring {}!'.format(cls))
        return []

    if fastr.abc.baseplugin.Plugin in cls.__bases__:
        print('[generate_plugins.py] Found {}'.format(cls))
        return [cls]
    else:
        return [y for x in cls.__bases__ for y in find_parents(x)]


def find_plugin_types():
    plugins = [base for plugin in fastr.plugin_manager.values() for base in find_parents(plugin)]
    return sorted(set(plugins), key=lambda x: x.id)


def generate_all():
    print('[generate_plugins.py] Start generating plugin references')
    plugin_types = find_plugin_types()
    print(f'[generate_plugins.py] Found plugin types to document: {plugin_types}')
    files = []

    for plugin_type in plugin_types:
        doc = plugin_type.__doc__ or 'NOT DOCUMENTED!'
        header = """\
.. _{id}-ref:

{ID} Reference
{underline}

{docstring}
        """.format(
            id=plugin_type.id.lower(),
            underline='-' * (len(plugin_type.id) + 10),
            ID=plugin_type.id,
            docstring=textwrap.dedent(doc)
        )

        print('[generate_plugins.py] Start generating {} reference'.format(plugin_type.id))
        filename = generate_plugins(header,
                                    {k: v for k, v in fastr.plugin_manager.items() if isinstance(v, plugin_type) or (isinstance(v, type) and issubclass(v, plugin_type))},
                                    plugin_type.id)
        files.append(filename)
        print('[generate_plugins.py] Finished generating {} reference'.format(plugin_type.id))

    filename = os.path.join(os.path.dirname(__file__), 'autogen', 'fastr.reference.rst')
    reference_header = os.path.join(os.path.dirname(__file__), 'static', 'reference.rst')

    with open(filename, 'w') as fh_out:
        with open(reference_header, 'r') as fh_in:
            fh_out.write(fh_in.read())

        for filepath in files:
            filepath = os.path.basename(filepath)
            fh_out.write('\n.. include:: {}\n'.format(filepath))


if __name__ == '__main__':
    generate_all()
