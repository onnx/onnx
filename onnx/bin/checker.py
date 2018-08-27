from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx
from onnx import defs, load, checker, NodeProto
from onnx.defs import OpSchema
import onnx.onnx_cpp2py_export.checker as c_checker
import onnx.shape_inference

import argparse


def check_model():  # type: () -> None
    parser = argparse.ArgumentParser('check-model')
    parser.add_argument('model_pb', type=argparse.FileType('rb'))
    args = parser.parse_args()

    model = load(args.model_pb)
    checker.check_model(model)


def check_node():  # type: () -> None
    parser = argparse.ArgumentParser('check-node')
    parser.add_argument('node_pb', type=argparse.FileType('rb'))
    args = parser.parse_args()

    node = NodeProto()
    node.ParseFromString(args.node_pb.read())
    checker.check_node(node)


def main():  # type: () -> None
    parser = argparse.ArgumentParser(
        description='Validates ONNX and ONNX-ML model files.')
    parser.add_argument('-m', '--ml', action='store_true', help='ML mode')
    parser.add_argument('files', nargs='*',
                        help='list of ONNX files ')
    args = parser.parse_args()

    for file in args.files:
        try:
            m = onnx.load(file)
            print('\n==== Validating ' + file + ' ====\n')
            if m.domain != '':
                print("Domain: " + m.domain)
            if m.producer_name != '':
                print("Producer name: " + m.producer_name)
            if m.producer_version != '':
                print("Producer version: " + m.producer_version)
            for entry in m.metadata_props:
                print(entry.key + ': ' + entry.value)

            onnx.checker.check_model(m)
        except c_checker.ValidationError as error:
            print(str(error))
        else:
            print('No errors found.')

    print('\n')


if __name__ == '__main__':
    main()
