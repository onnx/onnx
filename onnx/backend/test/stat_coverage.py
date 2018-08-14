#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from onnx import defs
from onnx.backend.test.case import collect_snippets
from typing import Any, IO, Sequence, Text


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


def gen_node_test_coverage(schemas, f, ml):
    # type: (Sequence[defs.OpSchema], IO[Any], bool) -> None
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
    f.write('## To be filled.\n')


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
