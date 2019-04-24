import os

from fastr.utils import cmd
from fastr.helpers.rest_generation import create_rest_table


def generate_command_reference():
    commands = sorted(cmd.find_commands())

    title = 'Command Line Tools'
    help_file = [
        '{}\n{}'.format(title, '=' * len(title)),
        "Fastr is shipped with a number of command line tools "
        "to perform common tasks and greatly simplify things "
        "such as debugging. The list of command line tools that "
        "is included in Fastr:",
    ]

    links = [":ref:`{c} <cmdline-{c}>`".format(c=c) for c in commands]
    doc_strings = []

    for command in commands:
        module = cmd.get_command_module(command)
        if hasattr(module.main, '__doc__'):
            # Use the first line of the doc string
            doc_strings.append(module.main.__doc__.strip().splitlines()[0])
        else:
            doc_strings.append('')

    help_file.append(create_rest_table(data=[links, doc_strings], headers=['command', 'description']))

    for command in commands:
        heading = "fastr {}".format(command)
        heading = '{}\n{}'.format(heading, '-' * len(heading))

        reference = '.. _cmdline-{}:'.format(command)

        help_text = \
"""
.. argparse::
   :module: fastr.utils.cmd.{c}
   :func: get_parser
   :prog: fastr {c}
""".format(c=command)

        help_file.append('\n\n'.join([
            reference,
            heading,
            help_text
            ])
        )

    filename = os.path.join(os.path.dirname(__file__), 'autogen', 'fastr.commandline.rst')

    with open(filename, 'w') as fh_out:
        fh_out.write('\n\n'.join(help_file))


if __name__ == '__main__':
    generate_command_reference()
