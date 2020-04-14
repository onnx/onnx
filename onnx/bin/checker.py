from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

from onnx import load, checker, NodeProto


def _get_arguments(prog, arg_name, file_type):
    parser = argparse.ArgumentParser(prog)
    parser.add_argument(arg_name, type=argparse.FileType(file_type))
    return parser.parse_args()


def check_model():  # type: () -> None
    args = _get_arguments('check-model', 'model_pb', 'rb')

    model = load(args.model_pb)
    checker.check_model(model)


def check_node():  # type: () -> None
    args = _get_arguments('check-node', 'node_pb', 'rb')

    node = NodeProto()
    node.ParseFromString(args.node_pb.read())
    checker.check_node(node)
