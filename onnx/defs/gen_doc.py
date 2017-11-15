#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import inspect
import io
from textwrap import dedent
import os
import sys
import json

from onnx import defs
from onnx.defs import OpSchema
from onnx.backend.test.case.node import collect_snippets

SNIPPETS = collect_snippets()

def generate_markdown_display_number(v):
    if defs.OpSchema.is_infinite(v):
        return '&#8734;'
    return str(v)

def generate_markdown_display_attr_type(v):
    assert isinstance(v, OpSchema.AttrType)
    s = str(v)
    s = s[s.rfind('.')+1:].lower()
    if s[-1] == 's':
        s = 'list of ' + s
    return s

def generate_markdown_support_level_str(level):
    return \
        "<sub>experimental</sub> " if level == OpSchema.SupportType.EXPERIMENTAL else ""

def generate_markdown(sorted_ops, file):
    with io.open(file, 'w', newline='') as fout:
      fout.write('## Operator Schemas\n')
      fout.write(
          "*This file is automatically generated from the\n"
          "            [def files](/onnx/defs) via [this script](/onnx/defs/gen_doc.py).\n"
          "            Do not modify directly and instead edit operator definitions.*\n")

      fout.write('\n')

      # Table of contents
      for _, op_type, schema in sorted_ops:
          s = '* <a href="#{}">{}{}</a>\n'.format(
              op_type, generate_markdown_support_level_str(schema.support_level), op_type)
          fout.write(s)

      fout.write('\n')

      for _, op_type, schema in sorted_ops:
          # op_type
          s = '### <a name="{}"></a><a name="{}">**{}{}**</a>\n'.format(
              op_type, op_type.lower(), generate_markdown_support_level_str(schema.support_level),
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
                      generate_markdown_display_attr_type(attr.type),
                      ' (required)' if attr.required else '')
                  s += '<dd>{}</dd>\n'.format(attr.description)
              s += '</dl>\n'


          # inputs
          s += '\n#### Inputs'
          if schema.min_input != schema.max_input:
              s += ' ({} - {})'.format(generate_markdown_display_number(schema.min_input),
                                  generate_markdown_display_number(schema.max_input))
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
              s += ' ({} - {})'.format(generate_markdown_display_number(schema.min_output),
                                   generate_markdown_display_number(schema.max_output))
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

def generate_json_attr_type(type):
    assert isinstance(type, OpSchema.AttrType)
    s = str(type)
    s = s[s.rfind('.')+1:].lower()
    if s[-1] == 's':
        s = 'list of ' + s
    return s

def generate_json_support_level_name(support_level):
    assert isinstance(support_level, OpSchema.SupportType)
    s = str(support_level)
    return s[s.rfind('.')+1:].lower()

def generate_json_types(types):
    r = []
    for type in types:
        r.append(type)
    r = sorted(r)
    return r

def generate_json(sorted_ops, file):
    json_root = []
    for _, op_type, schema in sorted_ops:
        json_schema = {}
        json_schema['support_level'] = generate_json_support_level_name(schema.support_level)
        if schema.doc:
            json_schema['doc'] = schema.doc.lstrip();
        if schema.inputs:
            json_schema['inputs'] = []
            for input in schema.inputs:
                json_schema['inputs'].append({ 
                    'name': input.name, 
                    'description': input.description,
                    'optional': input.optional,
                    'typeStr': input.typeStr,
                    'types': generate_json_types(input.types) })
        json_schema['min_input'] = schema.min_input;
        json_schema['max_input'] = schema.max_input;
        if schema.outputs:
            json_schema['outputs'] = []
            for output in schema.outputs:
                json_schema['outputs'].append({ 
                    'name': output.name, 
                    'description': output.description,
                    'optional': output.optional,
                    'typeStr': output.typeStr,
                    'types': generate_json_types(output.types) })
        json_schema['min_output'] = schema.min_output;
        json_schema['max_output'] = schema.max_output;
        if schema.attributes:
            json_schema['attributes'] = []
            for _, attribute in sorted(schema.attributes.items()):
                json_schema['attributes'].append({
                    'name' : attribute.name,
                    'description': attribute.description,
                    'type': generate_json_attr_type(attribute.type),
                    'required': attribute.required })
        if schema.type_constraints:
            json_schema["type_constraints"] = []
            for type_constraint in schema.type_constraints:
                json_schema['type_constraints'].append({
                    'description': type_constraint.description,
                    'type_param_str': type_constraint.type_param_str,
                    'allowed_type_strs': type_constraint.allowed_type_strs
                })
        if op_type in SNIPPETS:
            json_schema['snippets'] = []
            for summary, code in sorted(SNIPPETS[op_type]):
                json_schema['snippets'].append({
                    'summary': summary,
                    'code': code
                })
        json_root.append({
            "op_type": op_type,
            "schema": json_schema })
    with io.open(file, 'w', newline='') as fout:
        json_root = json.dumps(json_root, sort_keys=True, indent=2)
        for line in json_root.splitlines():
            line = line.rstrip()
            if sys.version_info[0] < 3:
                line = unicode(line)
            fout.write(line)
            fout.write('\n')

def main(args):
    sorted_ops = sorted(
        (int(schema.support_level), op_type, schema)
        for (op_type, schema) in defs.get_all_schemas().items())
    # If no format is specified, generate /docs/Operator.md and /docs/Operator.json by default
    if args.format == '':
        docs_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'docs')
        generate_markdown(sorted_ops, os.path.join(docs_root, 'Operators.md'))
        generate_json(sorted_ops, os.path.join(docs_root, 'Operators.json'))
    if args.format == 'markdown' and args.output and len(args.output) > 0:
        generate_markdown(sorted_ops, args.output)
    if args.format == 'json' and args.output and len(args.output) > 0:
        generate_json(sorted_ops, args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('gen_doc')
    parser.add_argument('-o', '--output', type=str, default='',
                        help='output path (default: %(default)s)')
    parser.add_argument('-f', '--format', type=str, default='',
                        help='output format (default: %(default)s)')
    main(parser.parse_args())
