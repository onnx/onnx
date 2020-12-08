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
from onnx import TensorProto, SequenceProto, MapProto
from typing import Text, Any
import numpy as np  # type: ignore


TOP_DIR = os.path.realpath(os.path.dirname(__file__))
DATA_DIR = os.path.join(TOP_DIR, 'data')


def generate_data(args):  # type: (argparse.Namespace) -> None

    def prepare_dir(path):  # type: (Text) -> None
        if not os.path.exists(path):
            os.makedirs(path)

    def write_proto(proto_path, numpy_proto, name):  # type: (Text, Any, Text) -> None
        # write the produced proto to the target path
        def is_different_proto(ref_proto, proto):  # type: (Any, Any) -> bool
            # check whether the produced proto is different from the existing one
            try:
                if ref_proto.dtype == np.object:
                    np.testing.assert_array_equal(ref_proto, proto)
                else:
                    np.testing.assert_allclose(ref_proto, proto, rtol=1e-3, atol=1e-5)
                return False
            except:
                return True
        need_update = True
        # if exist, it needs to be both readable and writable
        f = open(proto_path, 'rb+') if os.path.exists(proto_path) else open(proto_path, 'wb')
        # load the existing proto and do comparison
        if isinstance(numpy_proto, dict):
            dic = MapProto()
            dic.ParseFromString(f.read())
            dic_array = numpy_helper.to_dict(dic)
            need_update = is_different_proto(dic_array, numpy_proto)
            if need_update:
                proto = numpy_helper.from_dict(numpy_proto, name)  # type: ignore
        elif isinstance(numpy_proto, list):
            sequence = SequenceProto()
            sequence.ParseFromString(f.read())
            sequence_array = numpy_helper.to_list(sequence)
            need_update = is_different_proto(sequence_array, numpy_proto)
            if need_update:
                proto = numpy_helper.from_list(numpy_proto, name)  # type: ignore
        else:
            tensor = TensorProto()
            tensor.ParseFromString(f.read())
            tensor_array = numpy_helper.to_array(tensor)
            need_update = is_different_proto(tensor_array, numpy_proto)
            if need_update:
                proto = numpy_helper.from_array(numpy_proto, name)  # type: ignore
        # update if they are different
        if need_update:
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
