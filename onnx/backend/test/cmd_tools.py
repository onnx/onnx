from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import os
import shutil

import onnx.backend.test.case.node as node_test
import onnx.backend.test.case.model as model_test
from onnx import numpy_helper

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
DATA_DIR = os.path.join(TOP_DIR, 'data')


def generate_data(args):

    def prepare_dir(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    # node tests
    node_cases = node_test.collect_testcases()
    for case in node_cases:
        output_dir = os.path.join(args.output, 'node', case.name)
        prepare_dir(output_dir)
        with open(os.path.join(output_dir, 'node.pb'), 'wb') as f:
            f.write(case.node.SerializeToString())
        for i, input_np in enumerate(case.inputs):
            tensor = numpy_helper.from_array(input_np, case.node.input[i])
            with open(os.path.join(output_dir, 'input_{}.pb'.format(i)), 'wb') as f:
                f.write(tensor.SerializeToString())
        for i, output_np in enumerate(case.outputs):
            tensor = numpy_helper.from_array(output_np, case.node.output[i])
            with open(os.path.join(output_dir, 'output_{}.pb'.format(i)), 'wb') as f:
                f.write(tensor.SerializeToString())

    # model tests
    model_cases = model_test.collect_testcases()
    for case in model_cases:
        output_dir = os.path.join(
            args.output, 'model', case.kind, case.name)
        prepare_dir(output_dir)
        if case.kind == 'real':
            with open(os.path.join(output_dir, 'data.json'), 'w') as f:
                json.dump({
                    'url': case.url,
                    'model_name': case.model_name,
                }, f, sort_keys=True)
        else:
            with open(os.path.join(output_dir, 'model.onnx'), 'wb') as f:
                f.write(case.model.SerializeToString())
            for i, (inputs, outputs) in enumerate(case.data_sets):
                data_set_dir = os.path.join(
                    output_dir, 'test_data_set_{}'.format(i))
                prepare_dir(data_set_dir)
                for j, input_np in enumerate(inputs):
                    tensor = numpy_helper.from_array(
                        input_np, case.model.graph.input[i].name)
                    with open(os.path.join(
                            data_set_dir, 'input_{}.pb'.format(j)), 'wb') as f:
                        f.write(tensor.SerializeToString())
                for j, output_np in enumerate(outputs):
                    tensor = numpy_helper.from_array(
                        output_np, case.model.graph.output[i].name)
                    with open(os.path.join(
                            data_set_dir, 'output_{}.pb'.format(i)), 'wb') as f:
                        f.write(tensor.SerializeToString())


def parse_args():
    parser = argparse.ArgumentParser('backend-test-tools')
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser('generate-data', help='convert testcases to test data')
    subparser.add_argument('-o', '--output', default=DATA_DIR,
                           help='output directory (default: %(default)s)')
    subparser.set_defaults(func=generate_data)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
