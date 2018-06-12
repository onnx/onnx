import io
import os

from onnx import defs
from onnx.backend.test.case import collect_snippets


def is_ml(schemas):
    for s in schemas:
        if s.domain == 'ai.onnx.ml':
            return True
    return False


def gen_node_test_coverage(schemas, fname):
    node_tests = collect_snippets()
    with open(fname, 'w+') as f:
        common_covered = sorted([s.name for s in schemas
                if s.name in node_tests
                and s.support_level == defs.OpSchema.SupportType.COMMON])
        common_no_cover = sorted([s.name for s in schemas
                if s.name not in node_tests
                and s.support_level == defs.OpSchema.SupportType.COMMON])
        experimental_covered = sorted([s.name for s in schemas
                if s.name in node_tests
                and s.support_level == defs.OpSchema.SupportType.EXPERIMENTAL])
        experimental_no_cover = sorted([s.name for s in schemas
                if s.name not in node_tests
                and s.support_level == defs.OpSchema.SupportType.EXPERIMENTAL])
        num_common = len(common_covered) + len(common_no_cover)
        num_experimental = len(experimental_covered) + len(experimental_no_cover)
        f.write('# Node Test Coverage\n')
        f.write('## Summary\n')
        f.write('Node tests have covered {}/{} ({:.2f}%) common operators.\n\n'.format(
            len(common_covered), num_common, (len(common_covered) / num_common)))
        f.write('Node tests have covered {}/{} ({:.2f}%) experimental operators.\n\n'.format(
            len(experimental_covered), num_experimental,
            (len(experimental_covered) / num_experimental)))
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
                f.write('### {}\n'.format(s))
                if s in node_tests:
                    f.write('There are {} test cases, listed as following:\n'.format(
                        len(node_tests[s])))
                    for summary, code in sorted(node_tests[s]):
                        f.write('<details>\n')
                        f.write('<summary>{}</summary>\n\n'.format(summary))
                        f.write('```python\n{}\n```\n\n'.format(code))
                        f.write('</details>\n')
                        f.write('\n\n')
            f.write('<br/>\n\n')
        
        print(common_covered)
        print(common_no_cover)
        print(experimental_covered)
        print(experimental_no_cover)
   

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.realpath(__file__)))))
    docs_dir = os.path.join(base_dir, 'docs')
    schemas = defs.get_all_schemas()
    
    if is_ml(schemas):
        fname = os.path.join(docs_dir, 'TestCoverage-ml.md')
    else:
        fname = os.path.join(docs_dir, 'TestCoverage.md')

    gen_node_test_coverage(schemas, fname)


if __name__ == '__main__':
    main()
