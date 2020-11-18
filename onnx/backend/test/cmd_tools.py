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
from onnx import TensorProto
from typing import Text
import numpy as np


TOP_DIR = os.path.realpath(os.path.dirname(__file__))
DATA_DIR = os.path.join(TOP_DIR, 'data')


def generate_data(args):  # type: (argparse.Namespace) -> None

    def prepare_dir(path):  # type: (Text) -> None
        if not os.path.exists(path):
            os.makedirs(path)

    def write_proto(proto_filename, numpy_proto, name):
        update_file = True
        if os.path.exists(proto_filename):
            f = open(proto_filename, 'rb+')
        else:
            f = open(proto_filename, 'wb')
        if isinstance(numpy_proto, dict):
            print('dict')
            proto = numpy_helper.from_dict(numpy_proto, name)
        elif isinstance(numpy_proto, list):
            print('list')
            proto = numpy_helper.from_list(numpy_proto, name)
        else:
            tensor = TensorProto()
            tensor.ParseFromString(f.read())
            tensor_array = numpy_helper.to_array(tensor)
            try:
                if tensor_array.dtype == np.object:
                    np.testing.assert_array_equal(tensor_array, output)
                else:
                    np.testing.assert_allclose(tensor_array, output, rtol=1e-3, atol=1e-5)
                update_file = False
            except:
                proto = numpy_helper.from_array(numpy_proto, name)
        if update_file:
            f.seek(0)
            f.write(proto.SerializeToString())
            f.truncate()
        f.close()

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
                    input_name = os.path.join(data_set_dir, 'input_{}.pb'.format(j))
                    write_proto(input_name, input, case.model.graph.input[j].name)
                for j, output in enumerate(outputs):
                    output_name = os.path.join(data_set_dir, 'output_{}.pb'.format(j))
                    write_proto(output_name, output, case.model.graph.output[j].name)


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
