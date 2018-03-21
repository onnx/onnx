#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import os
from collections import defaultdict

from onnx import defs
from onnx.defs import OpSchema
from onnx.backend.test.case import collect_snippets

SNIPPETS = collect_snippets()
ONNX_ML = bool(os.getenv('ONNX_ML') == '1')

if ONNX_ML:
    ext = '-ml.md'
else:
    ext = '.md'


def display_number(v):
    if defs.OpSchema.is_infinite(v):
        return '&#8734;'
    return str(v)


def should_render_domain(domain):
    if domain == 'ai.onnx.ml' and not ONNX_ML:
        return False
    elif ONNX_ML and domain != 'ai.onnx.ml':
        return False
    return True


def display_attr_type(v):
    assert isinstance(v, OpSchema.AttrType)
    s = str(v)
    s = s[s.rfind('.') + 1:].lower()
    if s[-1] == 's':
        s = 'list of ' + s
    return s


def display_domain(domain):
    if domain:
        return "operator set '{}'".format(domain)
    else:
        return "the default ONNX operator set"


def display_version_link(name, version):
    changelog_md = 'Changelog' + ext
    name_with_ver = '{}-{}'.format(name, version)
    return '<a href="{}#{}">{}</a>'.format(changelog_md, name_with_ver, name_with_ver)


def display_schema(schema, versions):
    s = ''

    if schema.domain:
        domain_prefix = '{}.'.format(schema.domain)
    else:
        domain_prefix = ''

    # doc
    if schema.doc:
        s += '\n'
        s += '\n'.join('  ' + line
                       for line in schema.doc.lstrip().splitlines())
        s += '\n'

    # since version
    s += '\n#### Versioning\n'
    s += '\nThis operator is used if you are using version {} '.format(schema.since_version)
    s += 'of {} until the next BC-breaking change to this operator; e.g., it will be used if your protobuf has:\n\n'.format(
        display_domain(schema.domain))
    s += '~~~~\n'
    s += 'opset_import {\n'
    s += '  version = {}\n'.format(schema.since_version)
    if schema.domain:
        s += "  domain = '{}'\n".format(schema.domain)
    s += '}\n'
    s += '~~~~\n'
    if len(versions) > 1:
        # TODO: link to the Changelog.md
        s += '\nOther versions of this operator: {}\n'.format(
            ', '.join(display_version_link(domain_prefix + s.name, s.since_version) for s in versions[:-1]))

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
            option_str = ""
            if OpSchema.FormalParameterOption.Optional == input.option:
                option_str = " (optional)"
            elif OpSchema.FormalParameterOption.Variadic == input.option:
                option_str = " (variadic)"
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
                option_str = " (variadic)"
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

    return s


def support_level_str(level):
    return \
        "<sub>experimental</sub> " if level == OpSchema.SupportType.EXPERIMENTAL else ""


def main(args):
    with io.open(args.changelog, 'w', newline='') as fout:
        fout.write('## Operator Changelog\n')
        fout.write(
            "*This file is automatically generated from the\n"
            "            [def files](/onnx/defs) via [this script](/onnx/defs/gen_doc.py).\n"
            "            Do not modify directly and instead edit operator definitions.*\n")

        # domain -> version -> [schema]
        index = defaultdict(lambda: defaultdict(list))
        for schema in defs.get_all_schemas_with_history():
            index[schema.domain][schema.since_version].append(schema)

        fout.write('\n')

        for domain, versionmap in sorted(index.items()):
            if not should_render_domain(domain):
                continue

            if domain:
                s = '# {}\n'.format(domain)
                domain_prefix = '{}.'.format(domain)
            else:
                s = '# ai.onnx (default)\n'
                domain_prefix = ''

            for version, unsorted_schemas in sorted(versionmap.items()):
                s += '## Version {} of {}\n'.format(version, display_domain(domain))
                for schema in sorted(unsorted_schemas, key=lambda s: s.name):
                    name_with_ver = '{}-{}'.format(domain_prefix +
                                                   schema.name, schema.since_version)
                    s += '### <a name="{}"></a>**{}**</a>\n'.format(name_with_ver, name_with_ver)
                    s += display_schema(schema, [schema])
                    s += '\n'

            fout.write(s)

    with io.open(args.output, 'w', newline='') as fout:
        fout.write('## Operator Schemas\n')
        fout.write(
            "*This file is automatically generated from the\n"
            "            [def files](/onnx/defs) via [this script](/onnx/defs/gen_doc.py).\n"
            "            Do not modify directly and instead edit operator definitions.*\n")

        # domain -> support level -> name -> [schema]
        index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for schema in defs.get_all_schemas_with_history():
            index[schema.domain][int(schema.support_level)][schema.name].append(schema)

        fout.write('\n')

        # Table of contents
        for domain, supportmap in sorted(index.items()):
            if not should_render_domain(domain):
                continue

            if domain:
                s = '* {}\n'.format(domain)
                domain_prefix = '{}.'.format(domain)
            else:
                s = '* ai.onnx (default)\n'
                domain_prefix = ''
            fout.write(s)

            for _, namemap in sorted(supportmap.items()):
                for n, unsorted_versions in sorted(namemap.items()):
                    versions = sorted(unsorted_versions, key=lambda s: s.since_version)
                    schema = versions[-1]
                    s = '  * {}<a href="#{}">{}</a>\n'.format(
                        support_level_str(schema.support_level),
                        domain_prefix + n, domain_prefix + n)
                    fout.write(s)

        fout.write('\n')

        for domain, supportmap in sorted(index.items()):
            if not should_render_domain(domain):
                continue

            if domain:
                s = '## {}\n'.format(domain)
                domain_prefix = '{}.'.format(domain)
            else:
                s = '## ai.onnx (default)\n'
                domain_prefix = ''
            fout.write(s)

            for _support, namemap in sorted(supportmap.items()):
                for op_type, unsorted_versions in sorted(namemap.items()):
                    versions = sorted(unsorted_versions, key=lambda s: s.since_version)
                    schema = versions[-1]

                    # op_type
                    s = '### {}<a name="{}"></a><a name="{}">**{}**</a>\n'.format(
                        support_level_str(schema.support_level),
                        domain_prefix + op_type, domain_prefix + op_type.lower(),
                        domain_prefix + op_type)

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
                    fout.write(s)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    docs_dir = os.path.join(base_dir, 'docs')

    class Args(object):
        output = os.path.join(docs_dir, 'Operators' + ext)
        changelog = os.path.join(docs_dir, 'Changelog' + ext)
    main(Args)
