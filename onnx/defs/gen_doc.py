#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import os
from collections import defaultdict

from onnx import defs
from onnx.defs import OpSchema, FunctionProto
from onnx.backend.test.case import collect_snippets
from typing import Text, Sequence, Dict, List, Type, Set


SNIPPETS = collect_snippets()
ONNX_ML = bool(os.getenv('ONNX_ML') == '1')
ONNX_ML_DOMAIN = 'ai.onnx.ml'


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


def display_version_link(name, version):  # type: (Text, int) -> Text
    changelog_md = 'Changelog' + ext
    name_with_ver = '{}-{}'.format(name, version)
    return '<a href="{}#{}">{}</a>'.format(changelog_md, name_with_ver, name_with_ver)


def display_function_version_link(name, version):  # type: (Text, int) -> Text
    changelog_md = 'FunctionsChangelog' + ext
    name_with_ver = '{}-{}'.format(name, version)
    return '<a href="{}#{}">{}</a>'.format(changelog_md, name_with_ver, name_with_ver)


def display_schema(schema, versions):  # type: (OpSchema, Sequence[OpSchema]) -> Text
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
    s += '\n#### Version\n'
    s += '\nThis version of the operator has been available since version {}'.format(schema.since_version)
    s += ' of {}.\n'.format(display_domain(schema.domain))
    if len(versions) > 1:
        # TODO: link to the Changelog.md
        s += '\nOther versions of this operator: {}\n'.format(
            ', '.join(display_version_link(domain_prefix + v.name, v.since_version) for v in versions[:-1]))

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


def display_function(function, versions, domain=""):  # type: (FunctionProto, List[int], Text) -> Text
    s = ''

    if domain:
        domain_prefix = '{}.'.format(ONNX_ML_DOMAIN)
    else:
        domain_prefix = ''

    # doc
    if function.doc_string:
        s += '\n'
        s += '\n'.join('  ' + line
                       for line in function.doc_string.lstrip().splitlines())
        s += '\n'

    # since version
    s += '\n#### Version\n'
    s += '\nThis version of the function has been available since version {}'.format(function.since_version)
    s += ' of {}.\n'.format(display_domain(domain_prefix))
    if len(versions) > 1:
        s += '\nOther versions of this function: {}\n'.format(
            ', '.join(display_function_version_link(domain_prefix + function.name, v) for v in versions if v != function.since_version))

    # inputs
    s += '\n#### Inputs'
    s += '\n\n'
    if function.inputs:
        s += '<dl>\n'
        for input in function.inputs:
            s += '<dt>{}; </dt>\n'.format(input)
        s += '<br/></dl>\n'

    # outputs
    s += '\n#### Outputs'
    s += '\n\n'
    if function.outputs:
        s += '<dl>\n'
        for output in function.outputs:
            s += '<dt>{}; </dt>\n'.format(output)
        s += '<br/></dl>\n'

        # attributes
    if function.attribute:
        s += '\n#### Attribute:\n\n'
        s += '<dl>\n'
        for attr in function.attribute:
            s += '<dt>{};<br/></dt>\n'.format(attr)
        s += '</dl>\n'

    # nodes
    s += '\n#### Nodes'
    s += '\n\n'
    if function.nodes:
        s += '<dl>\n'
        for node in function.nodes:
            s += '<dd><b>{}: </b></dd><br/>'.format(node.name)
            s += '<dd>Input(s):</dd>'
            for input in node.inputs:
                s += '<dd> {};</dd>'.format(input)
            s += '<br/>\n'
            s += '<dd>Output(s):</dd>'
            for output in node.outputs:
                s += '<dd> {};</dd>'.format(output)
            s += '<br/>\n'
        s += '</dl>\n'

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

    with io.open(args.fn_changelog, 'w', newline='') as fout:
        fout.write('## Function Changelog\n')
        fout.write(
            "*This file is automatically generated from the\n"
            "            [def files](/onnx/defs) via [this script](/onnx/defs/gen_doc.py).\n"
            "            Do not modify directly and instead edit function definitions.*\n")

        if os.getenv('ONNX_ML'):
            all_functions = defs.get_functions(ONNX_ML_DOMAIN)
        else:
            all_functions = defs.get_functions('')

        for fn_name, functions in sorted(all_functions.items()):
            if os.getenv('ONNX_ML'):
                s = '## {}\n'.format(ONNX_ML_DOMAIN)
                domain_display_name = ONNX_ML_DOMAIN
                domain_prefix = '{}.'.format(ONNX_ML_DOMAIN)
            else:
                s = '# ai.onnx (default)\n'
                domain_display_name = 'ai.onnx (default)'
                domain_prefix = ''

            sorted_functions = sorted(functions, key=lambda s: s.since_version)
            available_versions = [func.since_version for func in sorted_functions]
            for function in sorted_functions:
                s += '## Version {} of domain {}\n'.format(sorted_functions.index(function) + 1, domain_display_name)
                name_with_ver = '{}-{}'.format(domain_prefix +
                                               fn_name, function.since_version)
                s += '### <a name="{}"></a>**{}**</a>\n'.format(name_with_ver, name_with_ver)
                s += display_function(function, available_versions, domain_prefix)
                s += '\n'

            fout.write(s)

    with io.open(args.output, 'w', newline='') as fout:
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

        # Table of contents
        exsting_ops = set()  # type: Set[Text]
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
                    if schema.name in exsting_ops:
                        continue
                    exsting_ops.add(schema.name)
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

    with io.open(args.function_output, 'w', newline='') as fout:
        fout.write('## Functions\n')
        fout.write(
            "*This file is automatically generated from the\n"
            "            [def files](/onnx/defs) via [this script](/onnx/defs/gen_doc.py).\n"
            "            Do not modify directly and instead edit function definitions.*\n")

        if os.getenv('ONNX_ML'):
            all_functions = defs.get_functions(ONNX_ML_DOMAIN)
        else:
            all_functions = defs.get_functions('')

        if all_functions:
            if os.getenv('ONNX_ML'):
                s = '## {}\n'.format(ONNX_ML_DOMAIN)
                domain_prefix = '{}.'.format(ONNX_ML_DOMAIN)
            else:
                s = '## ai.onnx (default)\n'
                domain_prefix = ''
            fout.write(s)

            existing_functions = set()  # type: Set[Text]
            for function_name, functions in sorted(all_functions.items()):
                available_versions = [func.since_version for func in functions]
                latest_version = sorted(available_versions)[-1]

                for function in sorted(functions, key=lambda s: s.since_version, reverse=True):
                    if function.name in existing_functions:
                        continue
                    existing_functions.add(function.name)
                    s = '  * {}<a href="#{}">{}</a>\n'.format(
                        "<sub>experimental</sub>" if latest_version == function.since_version else "",
                        domain_prefix + function.name, domain_prefix + function.name)
                    fout.write(s)

                fout.write('\n')

            fout.write('\n\n')

            for function_name, functions in sorted(all_functions.items()):
                available_versions = [func.since_version for func in functions]
                function = sorted(functions, key=lambda s: s.since_version, reverse=True)[0]
                s = '### {}<a name="{}"></a><a name="{}">**{}**</a>\n'.format(
                    "<sub>experimental</sub> " if latest_version == function.since_version else "",
                    domain_prefix + function.name, domain_prefix + function.name.lower(),
                    domain_prefix + function.name)

                s += display_function(function, available_versions, domain_prefix)
                s += '\n\n'
                fout.write(s)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    docs_dir = os.path.join(base_dir, 'docs')

    class Args(object):
        output = os.path.join(docs_dir, 'Operators' + ext)
        function_output = os.path.join(docs_dir, 'Functions' + ext)
        changelog = os.path.join(docs_dir, 'Changelog' + ext)
        fn_changelog = os.path.join(docs_dir, 'FunctionsChangelog' + ext)
    main(Args)
