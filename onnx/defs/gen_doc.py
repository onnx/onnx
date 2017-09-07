#/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

from onnx import defs
from onnx.defs import OpSchema

def display_number(v):
    if defs.OpSchema.is_infinite(v):
        return '&#8734;'
    return str(v)


def main(args):
    args.output.write('## Operator Schemas\n')

    for op_type, schema in sorted(defs.get_all_schemas().items()):
        # If support level is experimental, then don't generate documentation.
        if schema.support_level == OpSchema.SupportType.EXPERIMENTAL:
            continue

        # op_type
        s = '* **{}**\n'.format(op_type)

        # doc
        if schema.doc:
            s += '\n'
            s += '\n'.join('  ' + line
                           for line in schema.doc.lstrip().splitlines())
            s += '\n'

        # attributes
        if schema.attributes:
            s += '  * **attribute**:\n'
            s += '    <dl>\n'
            for _, attr in sorted(schema.attributes.items()):
                s += '      <dt>{}</dt>\n'.format(attr.name)
                s += '      <dd>{}</dd>\n'.format(attr.description)
            s += '    </dl>\n'


        # inputs
        s += '  * **input**:'
        if schema.min_input != schema.max_input:
            s += '{} - {}'.format(display_number(schema.min_input),
                                  display_number(schema.max_input))
        s += '\n'
        if schema.input_desc:
            s += '    <dl>\n'
            for input_name, input_desc in schema.input_desc:
                s += '      <dt>{}</dt>\n'.format(input_name)
                s += '      <dd>{}</dd>\n'.format(input_desc)
            s += '    </dl>\n'

        # outputs
        s += '  * **output**:'
        if schema.min_output != schema.max_output:
            s += '{} - {}'.format(display_number(schema.min_output),
                                  display_number(schema.max_output))
        s += '\n'
        if schema.output_desc:
            s += '    <dl>\n'
            for output_name, output_desc in schema.output_desc:
                s += '      <dt>{}</dt>\n'.format(output_name)
                s += '      <dd>{}</dd>\n'.format(output_desc)
            s += '    </dl>\n'

        s += '\n\n'
        args.output.write(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('gen_doc')
    parser.add_argument('-o', '--output', type=argparse.FileType('w'),
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
                            'docs',
                            'Operators.md'),
                        help='output path (default: %(default)s)')
    main(parser.parse_args())
