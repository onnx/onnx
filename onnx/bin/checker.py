from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

from onnx import load, checker, NodeProto


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
