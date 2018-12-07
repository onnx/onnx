#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from onnx import defs, load, AttributeProto
from onnx.backend.test.case import collect_snippets
from onnx.backend.test.runner import Runner
from onnx.backend.test.loader import load_model_tests
from typing import Any, IO, Sequence, Text, Dict, List


def is_ml(schemas):  # type: (Sequence[defs.OpSchema]) -> bool
    for s in schemas:
        if s.domain == 'ai.onnx.ml':
            return True
    return False


def gen_outlines(f, ml):  # type: (IO[Any], bool) -> None
    f.write('# Test Coverage Report')
    if ml:
        f.write(' (ONNX-ML Operators)\n')
    else:
        f.write(' (ONNX Core Operators)\n')
    f.write('## Outlines\n')
    f.write('* [Node Test Coverage](#node-test-coverage)\n')
    f.write('* [Model Test Coverage](#model-test-coverage)\n')
    f.write('* [Overall Test Coverage](#overall-test-coverage)\n')


common_covered = []  # type: Sequence[Text]
experimental_covered = []  # type: Sequence[Text]


def gen_node_test_coverage(schemas, f, ml):
    # type: (Sequence[defs.OpSchema], IO[Any], bool) -> None
    global common_covered
    global experimental_covered
    generators = set({
        'Multinomial',
        'RandomNormal',
        'RandomNormalLike',
        'RandomUniform',
        'RandomUniformLike',
    })
    node_tests = collect_snippets()
    common_covered = sorted([s.name for s in schemas
            if s.name in node_tests
            and s.support_level == defs.OpSchema.SupportType.COMMON
            and (s.domain == 'ai.onnx.ml') == ml])
    common_no_cover = sorted([s.name for s in schemas
            if s.name not in node_tests
            and s.support_level == defs.OpSchema.SupportType.COMMON
            and (s.domain == 'ai.onnx.ml') == ml])
    common_generator = sorted([name for name in common_no_cover
            if name in generators])
    experimental_covered = sorted([s.name for s in schemas
            if s.name in node_tests
            and s.support_level == defs.OpSchema.SupportType.EXPERIMENTAL
            and (s.domain == 'ai.onnx.ml') == ml])
    experimental_no_cover = sorted([s.name for s in schemas
            if s.name not in node_tests
            and s.support_level == defs.OpSchema.SupportType.EXPERIMENTAL
            and (s.domain == 'ai.onnx.ml') == ml])
    experimental_generator = sorted([name for name in experimental_no_cover
            if name in generators])
    num_common = len(common_covered) + len(common_no_cover) \
            - len(common_generator)
    num_experimental = len(experimental_covered) + len(experimental_no_cover) \
            - len(experimental_generator)
    f.write('# Node Test Coverage\n')
    f.write('## Summary\n')
    if num_common:
        f.write('Node tests have covered {}/{} ({:.2f}%, {} generators excluded) '
                'common operators.\n\n'.format(
                    len(common_covered), num_common,
                    (len(common_covered) / float(num_common) * 100),
                    len(common_generator)))
    else:
        f.write('Node tests have covered 0/0 (N/A) common operators. \n\n')
    if num_experimental:
        f.write('Node tests have covered {}/{} ({:.2f}%, {} generators excluded) '
                'experimental operators.\n\n'.format(
                    len(experimental_covered), num_experimental,
                    (len(experimental_covered) / float(num_experimental) * 100),
                    len(experimental_generator)))
    else:
        f.write('Node tests have covered 0/0 (N/A) experimental operators.\n\n')
    titles = ['&#x1F49A;Covered Common Operators',
              '&#x1F494;No Cover Common Operators',
              '&#x1F49A;Covered Experimental Operators',
              '&#x1F494;No Cover Experimental Operators',
              ]
    all_lists = [common_covered, common_no_cover,
            experimental_covered, experimental_no_cover]
    for t in titles:
        f.write('* [{}](#{})\n'.format(t[9:], t[9:].lower().replace(' ', '-')))
    f.write('\n')
    for t, l in zip(titles, all_lists):
        f.write('## {}\n'.format(t))
        for s in l:
            f.write('### {}'.format(s))
            if s in node_tests:
                f.write('\nThere are {} test cases, listed as following:\n'.format(
                    len(node_tests[s])))
                for summary, code in sorted(node_tests[s]):
                    f.write('<details>\n')
                    f.write('<summary>{}</summary>\n\n'.format(summary))
                    f.write('```python\n{}\n```\n\n'.format(code))
                    f.write('</details>\n')
            else:
                if s in generators:
                    f.write(' (random generator operator)\n')
                else:
                    f.write(' (call for test cases)\n')
            f.write('\n\n')
        f.write('<br/>\n\n')


def gen_model_test_coverage(schemas, f, ml):
    # type: (Sequence[defs.OpSchema], IO[Any], bool) -> None
    f.write('# Model Test Coverage\n')
    # Process schemas
    schema_dict = dict()
    for schema in schemas:
        schema_dict[schema.name] = schema
    # Load models from each model test using Runner._prepare_model_data
    # Need to grab associated nodes
    attrs = dict()  # type: Dict[Text, Dict[Text, List[Any]]]
    model_paths = []  # type: List[Any]
    for rt in load_model_tests(kind='real'):
        model_dir = Runner._prepare_model_data(rt)
        model_paths.append(os.path.join(model_dir, 'model.onnx'))
    model_paths.sort()
    model_written = False
    for model_pb_path in model_paths:
        model = load(model_pb_path)
        if ml:
            ml_present = False
            for opset in model.opset_import:
                if opset.domain == 'ai.onnx.ml':
                    ml_present = True
            if not ml_present:
                continue
            else:
                model_written = True
        f.write('## {}\n'.format(model.graph.name))
        # Deconstruct model
        num_covered = 0
        for node in model.graph.node:
            if node.op_type in common_covered or node.op_type in experimental_covered:
                num_covered += 1
                # Add details of which nodes are/aren't covered
                # Iterate through and store each node's attributes
                for attr in node.attribute:
                    if node.op_type not in attrs:
                        attrs[node.op_type] = dict()
                    if attr.name not in attrs[node.op_type]:
                        attrs[node.op_type][attr.name] = []
                    if attr.type == AttributeProto.FLOAT:
                        if attr.f not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.f)
                    elif attr.type == AttributeProto.INT:
                        if attr.i not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.i)
                    elif attr.type == AttributeProto.STRING:
                        if attr.s not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.s)
                    elif attr.type == AttributeProto.TENSOR:
                        if attr.t not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.t)
                    elif attr.type == AttributeProto.GRAPH:
                        if attr.g not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.g)
                    elif attr.type == AttributeProto.FLOATS:
                        if attr.floats not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.floats)
                    elif attr.type == AttributeProto.INTS:
                        if attr.ints not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.ints)
                    elif attr.type == AttributeProto.STRINGS:
                        if attr.strings not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.strings)
                    elif attr.type == AttributeProto.TENSORS:
                        if attr.tensors not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.tensors)
                    elif attr.type == AttributeProto.GRAPHS:
                        if attr.graphs not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.graphs)
        f.write('\n{} has {} nodes. Of these, {} are covered by node tests ({}%)\n\n\n'.format(
            model.graph.name, num_covered, len(model.graph.node), 100.0 * float(
                num_covered) / float(len(model.graph.node))))
        # Iterate through attrs, print
        f.write('<details>\n')
        f.write('<summary>nodes</summary>\n\n')
        for op in sorted(attrs):
            f.write('<details>\n')
            # Get total number of attributes for node schema
            f.write('<summary>{}: {} out of {} attributes covered</summary>\n\n'
                    .format(op, len(attrs[op].keys()), len(schema_dict[op]
                        .attributes)))
            for attribute in sorted(schema_dict[op].attributes):
                if attribute in attrs[op]:
                    f.write('{}: {}\n'.format(attribute, len(attrs[op][attribute])))
                else:
                    f.write('{}: 0\n'.format(attribute))
            f.write('</details>\n')
        f.write('</details>\n\n\n')
    if not model_written and ml:
        f.write('No model tests present for selected domain\n')


def gen_overall_test_coverage(schemas, f, ml):
    # type: (Sequence[defs.OpSchema], IO[Any], bool) -> None
    f.write('# Overall Test Coverage\n')
    f.write('## To be filled.\n')


def main():
    # type: () -> None
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.realpath(__file__)))))
    docs_dir = os.path.join(base_dir, 'docs')
    schemas = defs.get_all_schemas()

    ml = is_ml(schemas)
    if ml:
        fname = os.path.join(docs_dir, 'TestCoverage-ml.md')
    else:
        fname = os.path.join(docs_dir, 'TestCoverage.md')

    with open(fname, 'w+') as f:
        gen_outlines(f, ml)
        gen_node_test_coverage(schemas, f, ml)
        gen_model_test_coverage(schemas, f, ml)
        gen_overall_test_coverage(schemas, f, ml)


if __name__ == '__main__':
    main()
