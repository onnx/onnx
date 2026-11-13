# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

from onnx import NodeProto, checker, load


def check_model() -> None:
    parser = argparse.ArgumentParser("check-model")
    parser.add_argument("model_pb", type=argparse.FileType("rb"))
    parser.add_argument(
        "--no-check-external-data-files",
        dest="check_external_data_files",
        action="store_false",
        help="validate external data references without accessing the files",
    )
    args = parser.parse_args()

    if args.check_external_data_files:
        model = load(args.model_pb)
        checker.check_model(model)
    else:
        # Check by path: external data locations are relative to the model
        # directory, and loading the model would read the external data.
        args.model_pb.close()
        checker.check_model(args.model_pb.name, check_external_data_files=False)


def check_node() -> None:
    parser = argparse.ArgumentParser("check-node")
    parser.add_argument("node_pb", type=argparse.FileType("rb"))
    args = parser.parse_args()

    node = NodeProto()
    node.ParseFromString(args.node_pb.read())
    checker.check_node(node)
