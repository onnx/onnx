#!/usr/bin/env python

# SPDX-License-Identifier: Apache-2.0

import os
import io

from onnx import defs, load, AttributeProto
from onnx.backend.test.case import collect_snippets
from onnx.backend.test.runner import Runner
from onnx.backend.test.loader import load_model_tests
from typing import Any, IO, Sequence, Dict, List


def is_ml(schemas: Sequence[defs.OpSchema]) -> bool:
    for s in schemas:
        if s.domain == 'ai.onnx.ml':
            return True
    return False


def gen_outlines(f: IO[Any], ml: bool) -> None:
    f.write('# Test Coverage Report')
    if ml:
        f.write(' (ONNX-ML Operators)\n')
    else:
        f.write(' (ONNX Core Operators)\n')
    f.write('## Outlines\n')
    f.write('* [Node Test Coverage](#node-test-coverage)\n')
    f.write('* [Model Test Coverage](#model-test-coverage)\n')
    f.write('* [Overall Test Coverage](#overall-test-coverage)\n')


common_covered: Sequence[str] = []
experimental_covered: Sequence[str] = []


def gen_node_test_coverage(schemas: Sequence[defs.OpSchema], f: IO[Any], ml: bool) -> None:
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
    common_covered = sorted(s.name for s in schemas
            if s.name in node_tests
            and s.support_level == defs.OpSchema.SupportType.COMMON
            and (s.domain == 'ai.onnx.ml') == ml)
    common_no_cover = sorted(s.name for s in schemas
            if s.name not in node_tests
            and s.support_level == defs.OpSchema.SupportType.COMMON
            and (s.domain == 'ai.onnx.ml') == ml)
    common_generator = sorted(name for name in common_no_cover
            if name in generators)
    experimental_covered = sorted(s.name for s in schemas
            if s.name in node_tests
            and s.support_level == defs.OpSchema.SupportType.EXPERIMENTAL
            and (s.domain == 'ai.onnx.ml') == ml)
    experimental_no_cover = sorted(s.name for s in schemas
            if s.name not in node_tests
            and s.support_level == defs.OpSchema.SupportType.EXPERIMENTAL
            and (s.domain == 'ai.onnx.ml') == ml)
    experimental_generator = sorted(name for name in experimental_no_cover
            if name in generators)
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
        f.write(f"* [{t[9:]}](#{t[9:].lower().replace(' ', '-')})\n")
    f.write('\n')
    for t, l in zip(titles, all_lists):
        f.write(f'## {t}\n')
        for s in l:
            f.write(f'### {s}')
            if s in node_tests:
                f.write('\nThere are {} test cases, listed as following:\n'.format(
                    len(node_tests[s])))
                for summary, code in sorted(node_tests[s]):
                    f.write('<details>\n')
                    f.write(f'<summary>{summary}</summary>\n\n')
                    f.write(f'```python\n{code}\n```\n\n')
                    f.write('</details>\n')
            else:
                if s in generators:
                    f.write(' (random generator operator)\n')
                else:
                    f.write(' (call for test cases)\n')
            f.write('\n\n')
        f.write('<br/>\n\n')


def gen_model_test_coverage(schemas: Sequence[defs.OpSchema], f: IO[Any], ml: bool) -> None:
    f.write('# Model Test Coverage\n')
    # Process schemas
    schema_dict = dict()
    for schema in schemas:
        schema_dict[schema.name] = schema
    # Load models from each model test using Runner.prepare_model_data
    # Need to grab associated nodes
    attrs: Dict[str, Dict[str, List[Any]]] = dict()
    model_paths: List[Any] = []
    for rt in load_model_tests(kind='real'):
        model_dir = Runner.prepare_model_data(rt)
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
        f.write(f'## {model.graph.name}\n')
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
                    f.write(f'{attribute}: {len(attrs[op][attribute])}\n')
                else:
                    f.write(f'{attribute}: 0\n')
            f.write('</details>\n')
        f.write('</details>\n\n\n')
    if not model_written and ml:
        f.write('No model tests present for selected domain\n')


def gen_overall_test_coverage(schemas: Sequence[defs.OpSchema], f: IO[Any], ml: bool) -> None:
    f.write('# Overall Test Coverage\n')
    f.write('## To be filled.\n')


def gen_spdx(f: IO[Any]) -> None:
    f.write('<!--- SPDX-License-Identifier: Apache-2.0 -->\n')


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.realpath(__file__)))))
    docs_dir = os.path.join(base_dir, 'docs')
    schemas = defs.get_all_schemas()

    has_ml = is_ml(schemas)
    fname = os.path.join(docs_dir, 'TestCoverage.md')
    with open(fname, 'w+', newline='', encoding="utf-8") as f:  # type: ignore
        gen_spdx(f)
        gen_outlines(f, False)
        gen_node_test_coverage(schemas, f, False)
        gen_model_test_coverage(schemas, f, False)
        gen_overall_test_coverage(schemas, f, False)

    if has_ml:
        fname = os.path.join(docs_dir, 'TestCoverage-ml.md')
        with open(fname, 'w+', newline='', encoding="utf-8") as f:  # type: ignore
            gen_spdx(f)
            gen_outlines(f, True)
            gen_node_test_coverage(schemas, f, True)
            gen_model_test_coverage(schemas, f, True)
            gen_overall_test_coverage(schemas, f, True)


if __name__ == '__main__':
    main()
