#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import io
import os
import sys

import numpy as np  # type: ignore

from onnx import defs, FunctionProto, helper, OperatorStatus
from onnx.defs import OpSchema, ONNX_DOMAIN, ONNX_ML_DOMAIN
from onnx.backend.test.case import collect_snippets
from onnx.backend.sample.ops import collect_sample_implementations
from typing import Any, Text, Sequence, Dict, List, Type, Set, Tuple


SNIPPETS = collect_snippets()
SAMPLE_IMPLEMENTATIONS = collect_sample_implementations()
ONNX_ML = not bool(os.getenv('ONNX_ML') == '0')


if ONNX_ML:
    ext = '-ml.md'
else:
    ext = '.md'


def display_number(v):  # type: (int) -> Text
    if defs.OpSchema.is_infinite(v):
        return '&#8734;'
    return Text(v)


def should_render_domain(domain):  # type: (Text) -> bool
    if domain == ONNX_ML_DOMAIN and not ONNX_ML:
        return False
    elif ONNX_ML and domain != ONNX_ML_DOMAIN:
        return False
    return True


def format_name_with_domain(domain, schema_name):  # type: (Text, Text) -> Text
    if domain:
        return '{}.{}'.format(domain, schema_name)
    else:
        return schema_name


def display_attr_type(v):  # type: (OpSchema.AttrType) -> Text
    assert isinstance(v, OpSchema.AttrType)
    s = Text(v)
    s = s[s.rfind('.') + 1:].lower()
    if s[-1] == 's':
        s = 'list of ' + s
    return s


def display_domain(domain):  # type: (Text) -> Text
    if domain:
        return "the '{}' operator set".format(domain)
    else:
        return "the default ONNX operator set"


def display_domain_short(domain):  # type: (Text) -> Text
    if domain:
        return domain
    else:
        return 'ai.onnx (default)'


def display_version_link(name, version):  # type: (Text, int) -> Text
    changelog_md = 'Changelog' + ext
    name_with_ver = '{}-{}'.format(name, version)
    return '<a href="{}#{}">{}</a>'.format(changelog_md, name_with_ver, name_with_ver)


def display_schema(schema, versions):  # type: (OpSchema, Sequence[OpSchema]) -> Text
    s = ''

    # doc
    if schema.doc:
        s += '\n'
        s += '\n'.join('  ' + line
                       for line in schema.doc.lstrip().splitlines())
        s += '\n'

    # since version
    s += '\n#### Version\n'
    if schema.support_level == OpSchema.SupportType.EXPERIMENTAL:
        s += '\nNo versioning maintained for experimental ops.'
    else:
        s += '\nThis version of the operator has been ' + ('deprecated' if schema.deprecated else 'available') + ' since version {}'.format(schema.since_version)
        s += ' of {}.\n'.format(display_domain(schema.domain))
        if len(versions) > 1:
            # TODO: link to the Changelog.md
            s += '\nOther versions of this operator: {}\n'.format(
                ', '.join(display_version_link(format_name_with_domain(v.domain, v.name),
                                               v.since_version) for v in versions[:-1]))

    # If this schema is deprecated, don't display any of the following sections
    if schema.deprecated:
        return s

    # attributes
    if schema.attributes:
        s += '\n#### Attributes\n\n'
        s += '<dl>\n'
        for _, attr in sorted(schema.attributes.items()):
            # option holds either required or default value
            opt = ''
            if attr.required:
                opt = 'required'
            elif attr.default_value.name:
                default_value = helper.get_attribute_value(attr.default_value)

                def format_value(value):  # type: (Any) -> Text
                    if isinstance(value, float):
                        value = np.round(value, 5)
                    if isinstance(value, (bytes, bytearray)) and sys.version_info[0] == 3:
                        value = value.decode('utf-8')
                    return str(value)

                if isinstance(default_value, list):
                    default_value = [format_value(val) for val in default_value]
                else:
                    default_value = format_value(default_value)
                opt = 'default is {}'.format(default_value)

            s += '<dt><tt>{}</tt> : {}{}</dt>\n'.format(
                attr.name,
                display_attr_type(attr.type),
                ' ({})'.format(opt) if opt else '')
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
            option_str = ""
            if OpSchema.FormalParameterOption.Optional == input.option:
                option_str = " (optional)"
            elif OpSchema.FormalParameterOption.Variadic == input.option:
                if input.isHomogeneous:
                    option_str = " (variadic)"
                else:
                    option_str = " (variadic, heterogeneous)"
            s += '<dt><tt>{}</tt>{} : {}</dt>\n'.format(input.name, option_str, input.typeStr)
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
            option_str = ""
            if OpSchema.FormalParameterOption.Optional == output.option:
                option_str = " (optional)"
            elif OpSchema.FormalParameterOption.Variadic == output.option:
                if output.isHomogeneous:
                    option_str = " (variadic)"
                else:
                    option_str = " (variadic, heterogeneous)"
            s += '<dt><tt>{}</tt>{} : {}</dt>\n'.format(output.name, option_str, output.typeStr)
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
            s += '<dt><tt>{}</tt> : {}</dt>\n'.format(
                type_constraint.type_param_str, allowedTypeStr)
            s += '<dd>{}</dd>\n'.format(type_constraint.description)
        s += '</dl>\n'

    # Function Body
    if schema.has_function:  # type: ignore
        s += '\n#### Function\n'
        s += '\nThe Function can be represented as a function.\n'

    return s


def support_level_str(level):  # type: (OpSchema.SupportType) -> Text
    return \
        "<sub>experimental</sub> " if level == OpSchema.SupportType.EXPERIMENTAL else ""


def main(args):  # type: (Type[Args]) -> None
    with io.open(args.changelog, 'w', newline='') as fout:
        fout.write('## Operator Changelog\n')
        fout.write(
            "*This file is automatically generated from the\n"
            "            [def files](/onnx/defs) via [this script](/onnx/defs/gen_doc.py).\n"
            "            Do not modify directly and instead edit operator definitions.*\n")

        # domain -> version -> [schema]
        dv_index = defaultdict(lambda: defaultdict(list))  # type: Dict[Text, Dict[int, List[OpSchema]]]
        for schema in defs.get_all_schemas_with_history():
            dv_index[schema.domain][schema.since_version].append(schema)

        fout.write('\n')

        for domain, versionmap in sorted(dv_index.items()):
            if not should_render_domain(domain):
                continue

            s = '# {}\n'.format(display_domain_short(domain))

            for version, unsorted_schemas in sorted(versionmap.items()):
                s += '## Version {} of {}\n'.format(version, display_domain(domain))
                for schema in sorted(unsorted_schemas, key=lambda s: s.name):
                    name_with_ver = '{}-{}'.format(format_name_with_domain(domain, schema.name),
                                                   schema.since_version)
                    s += ('### <a name="{}"></a>**{}**' + (' (deprecated)' if schema.deprecated else '') + '</a>\n').format(name_with_ver, name_with_ver)
                    s += display_schema(schema, [schema])
                    s += '\n'

            fout.write(s)

    with io.open(args.output, 'w', newline='', encoding="utf-8") as fout:
        fout.write('## Operator Schemas\n')
        fout.write(
            "*This file is automatically generated from the\n"
            "            [def files](/onnx/defs) via [this script](/onnx/defs/gen_doc.py).\n"
            "            Do not modify directly and instead edit operator definitions.*\n")

        # domain -> support level -> name -> [schema]
        index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # type: Dict[Text, Dict[int, Dict[Text, List[OpSchema]]]]
        for schema in defs.get_all_schemas_with_history():
            index[schema.domain][int(schema.support_level)][schema.name].append(schema)

        fout.write('\n')

        # Preprocess the Operator Schemas
        # [(domain, [(support_level, [(schema name, current schema, all versions schemas)])])]
        operator_schemas = list()  # type: List[Tuple[Text, List[Tuple[int, List[Tuple[Text, OpSchema, List[OpSchema]]]]]]]
        exsting_ops = set()  # type: Set[Text]
        for domain, _supportmap in sorted(index.items()):
            if not should_render_domain(domain):
                continue

            processed_supportmap = list()
            for _support, _namemap in sorted(_supportmap.items()):
                processed_namemap = list()
                for n, unsorted_versions in sorted(_namemap.items()):
                    versions = sorted(unsorted_versions, key=lambda s: s.since_version)
                    schema = versions[-1]
                    if schema.name in exsting_ops:
                        continue
                    exsting_ops.add(schema.name)
                    processed_namemap.append((n, schema, versions))
                processed_supportmap.append((_support, processed_namemap))
            operator_schemas.append((domain, processed_supportmap))

        # Table of contents
        for domain, supportmap in operator_schemas:
            s = '* {}\n'.format(display_domain_short(domain))
            fout.write(s)
            function_ops = list()
            for _, namemap in supportmap:
                for n, schema, versions in namemap:
                    if schema.has_function:  # type: ignore
                        function_ops.append((n, schema, versions))
                        continue
                    s = '  * {}<a href="#{}">{}</a>\n'.format(
                        support_level_str(schema.support_level),
                        format_name_with_domain(domain, n),
                        format_name_with_domain(domain, n))
                    fout.write(s)
            if len(function_ops):
                fout.write('\n')
                fout.write('  **Operators with function registered:**\n')
                for n, schema, versions in function_ops:
                    s = '  * {}<a href="#{}">{}</a>\n'.format(
                        support_level_str(schema.support_level),
                        format_name_with_domain(domain, n),
                        format_name_with_domain(domain, n))
                    fout.write(s)

        fout.write('\n')

        for domain, supportmap in operator_schemas:
            s = '## {}\n'.format(display_domain_short(domain))
            fout.write(s)

            for _, namemap in supportmap:
                for op_type, schema, versions in namemap:
                    # op_type
                    s = ('### {}<a name="{}"></a><a name="{}">**{}**' + (' (deprecated)' if schema.deprecated else '') + '</a>\n').format(
                        support_level_str(schema.support_level),
                        format_name_with_domain(domain, op_type),
                        format_name_with_domain(domain, op_type.lower()),
                        format_name_with_domain(domain, op_type))

                    s += display_schema(schema, versions)

                    s += '\n\n'

                    if op_type in SNIPPETS:
                        s += '#### Examples\n\n'
                        for summary, code in sorted(SNIPPETS[op_type]):
                            s += '<details>\n'
                            s += '<summary>{}</summary>\n\n'.format(summary)
                            s += '```python\n{}\n```\n\n'.format(code)
                            s += '</details>\n'
                            s += '\n\n'
                    if op_type.lower() in SAMPLE_IMPLEMENTATIONS:
                        s += '#### Sample Implementation\n\n'
                        s += '<details>\n'
                        s += '<summary>{}</summary>\n\n'.format(op_type)
                        s += '```python\n{}\n```\n\n'.format(SAMPLE_IMPLEMENTATIONS[op_type.lower()])
                        s += '</details>\n'
                        s += '\n\n'

                    fout.write(s)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    docs_dir = os.path.join(base_dir, 'docs')

    class Args(object):
        output = os.path.join(docs_dir, 'Operators' + ext)
        changelog = os.path.join(docs_dir, 'Changelog' + ext)
    main(Args)
