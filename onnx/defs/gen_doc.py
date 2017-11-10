#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import argparse
import inspect
from textwrap import dedent
import os

from onnx import defs
from onnx.defs import OpSchema
from onnx.backend.test.case.node import collect_snippets

SNIPPETS = collect_snippets()


def display_number(v):
    if defs.OpSchema.is_infinite(v):
        return '&#8734;'
    return str(v)


def display_attr_type(v):
    assert isinstance(v, OpSchema.AttrType)
    s = str(v)
    s = s[s.rfind('.')+1:].lower()
    if s[-1] == 's':
        s = 'list of ' + s
    return s


def support_level_str(level):
    return \
        "<sub>experimental</sub> " if level == OpSchema.SupportType.EXPERIMENTAL else ""


def main(args):

    with io.open(args.output, 'w', newline='') as fout:

        fout.write('## Operator Schemas\n')
        fout.write(
            "*This file is automatically generated from the\n"
            "            [def files](/onnx/defs) via [this script](/onnx/defs/gen_doc.py).\n"
            "            Do not modify directly and instead edit operator definitions.*\n")


        sorted_ops = sorted(
            (int(schema.support_level), op_type, schema)
            for (op_type, schema) in defs.get_all_schemas().items())

        fout.write('\n')

        # Table of contents
        for _, op_type, schema in sorted_ops:
            s = '* <a href="#{}">{}{}</a>\n'.format(
                op_type, support_level_str(schema.support_level), op_type)
            fout.write(s)

        fout.write('\n')

        for _, op_type, schema in sorted_ops:
            # op_type
            s = '### <a name="{}"></a><a name="{}">**{}{}**</a>\n'.format(
                op_type, op_type.lower(), support_level_str(schema.support_level),
                op_type)

            # doc
            if schema.doc:
                s += '\n'
                s += '\n'.join('  ' + line
                               for line in schema.doc.lstrip().splitlines())
                s += '\n'

            # attributes
            if schema.attributes:
                s += '\n#### Attributes\n\n'
                s += '<dl>\n'
                for _, attr in sorted(schema.attributes.items()):
                    s += '<dt><tt>{}</tt> : {}{}</dt>\n'.format(
                        attr.name,
                        display_attr_type(attr.type),
                        ' (required)' if attr.required else '')
                    s += '<dd>{}</dd>\n'.format(attr.description)
                s += '</dl>\n'


            # inputs
            s += '\n#### Inputs'
            if schema.min_input != schema.max_input:
                s += ' ({} - {})'.format(display_number(schema.min_input),
                                      display_number(schema.max_input))
            s += '\n\n'
            if schema.inputs:
                s += '<dl>\n'
                for input in schema.inputs:
                    s += '<dt><tt>{}</tt>{} : {}</dt>\n'.format(input.name, ' (optional)' if input.optional else '', input.typeStr)
                    s += '<dd>{}</dd>\n'.format(input.description)
                s += '</dl>\n'

            # outputs
            s += '\n#### Outputs'
            if schema.min_output != schema.max_output:
                s += ' ({} - {})'.format(display_number(schema.min_output),
                                       display_number(schema.max_output))
            s += '\n\n'

            if schema.outputs:
                s += '<dl>\n'
                for output in schema.outputs:
                    s += '<dt><tt>{}</tt> : {}</dt>\n'.format(output.name, output.typeStr)
                    s += '<dd>{}</dd>\n'.format(output.description)
                s += '</dl>\n'

            # type constraints
            s += '\n#### Type Constraints'
            s += '\n\n'
            if schema.type_constraints:
                s += '<dl>\n'
                for type_constraint in schema.type_constraints:
                    allowedTypes = type_constraint.allowed_type_strs
                    if (len(allowedTypes) > 0):
                        allowedTypeStr = allowedTypes[0]
                    for allowedType in allowedTypes[1:]:
                        allowedTypeStr += ', ' + allowedType
                    s += '<dt><tt>{}</tt> : {}</dt>\n'.format(type_constraint.type_param_str, allowedTypeStr)
                    s += '<dd>{}</dd>\n'.format(type_constraint.description)
                s += '</dl>\n'

            s += '\n\n'

            if op_type in SNIPPETS:
                s += '#### Examples\n\n'
                for summary, code in sorted(SNIPPETS[op_type]):
                    s += '<details>\n'
                    s += '<summary>{}</summary>\n\n'.format(summary)
                    s += '```python\n{}\n```\n\n'.format(code)
                    s += '</details>\n'
                    s += '\n\n'
            fout.write(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('gen_doc')
    parser.add_argument('-o', '--output', type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
                            'docs',
                            'Operators.md'),
                        help='output path (default: %(default)s)')
    main(parser.parse_args())
