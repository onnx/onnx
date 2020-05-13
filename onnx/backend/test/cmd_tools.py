from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import os
import shutil

import numpy as np
import onnx.backend.test.case.node as node_test
import onnx.backend.test.case.model as model_test
from onnx import numpy_helper
from typing import Text


TOP_DIR = os.path.realpath(os.path.dirname(__file__))
DATA_DIR = os.path.join(TOP_DIR, 'data')


def generate_data(args):  # type: (argparse.Namespace) -> None

    def prepare_dir(path):  # type: (Text) -> None
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    cases = model_test.collect_testcases() + node_test.collect_testcases()
    for case in cases:
        output_dir = os.path.join(
            args.output, case.kind, case.name)
        prepare_dir(output_dir)
        if case.kind == 'real':
            with open(os.path.join(output_dir, 'data.json'), 'w') as fi:
                json.dump({
                    'url': case.url,
                    'model_name': case.model_name,
                    'rtol': case.rtol,
                    'atol': case.atol,
                }, fi, sort_keys=True)
        else:
            with open(os.path.join(output_dir, 'model.onnx'), 'wb') as f:
                f.write(case.model.SerializeToString())
            for i, (inputs, outputs) in enumerate(case.data_sets):
                data_set_dir = os.path.join(
                    output_dir, 'test_data_set_{}'.format(i))
                prepare_dir(data_set_dir)
                for j, input in enumerate(inputs):
                    if isinstance(input, dict):
                        arr = numpy_helper.from_dict_to_map(
                            input, case.model.graph.input[j].name)
                    elif isinstance(input, list):
                        arr = numpy_helper.from_list_to_sequence(
                            input, case.model.graph.input[j].name)
                    elif isinstance(input, np.ndarray):
                        arr = numpy_helper.from_array(
                            input, case.model.graph.input[j].name)
                    else:
                        raise TypeError(
                            "Your input is not a sequence (list), dictionary (map), or tensor (array) "
                            "and cannot be processed accordingly.", input)
                    with open(os.path.join(
                            data_set_dir, 'input_{}.pb'.format(j)), 'wb') as f:
                        f.write(arr.SerializeToString())
                for j, output in enumerate(outputs):
                    if isinstance(output, dict):
                        arr = numpy_helper.from_dict_to_map(
                            output, case.model.graph.output[j].name)
                    elif isinstance(output, list):
                        arr = numpy_helper.from_list_to_sequence(
                            output, case.model.graph.output[j].name)
                    elif isinstance(output, np.ndarray):
                        arr = numpy_helper.from_array(
                            output, case.model.graph.output[j].name)
                    else:
                        raise TypeError(
                            "Your output is not a sequence (list), dictionary (map), or tensor (array) ",
                            "and cannot be processed accordingly.", output)
                    with open(os.path.join(
                            data_set_dir, 'output_{}.pb'.format(j)), 'wb') as f:
                        f.write(arr.SerializeToString())


def parse_args():  # type: () -> argparse.Namespace
    parser = argparse.ArgumentParser('backend-test-tools')
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser('generate-data', help='convert testcases to test data')
    subparser.add_argument('-o', '--output', default=DATA_DIR,
                           help='output directory (default: %(default)s)')
    subparser.set_defaults(func=generate_data)

    return parser.parse_args()


def main():  # type: () -> None
    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
