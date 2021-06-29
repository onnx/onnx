# SPDX-License-Identifier: Apache-2.0

from onnx import GraphProto, ModelProto, onnx_cpp2py_export, load_from_string
from typing import Text


class ParseError(Exception):
    pass


def parse_model(model_text):  # type: (Text) -> ModelProto
    (success, msg, model_proto_str) = onnx_cpp2py_export.parser.parse_model(model_text)
    if success:
        return load_from_string(model_proto_str)
    else:
        raise ParseError(msg)


def parse_graph(graph_text):  # type: (Text) -> GraphProto
    (success, msg, graph_proto_str) = onnx_cpp2py_export.parser.parse_graph(graph_text)
    if success:
        G = GraphProto()
        G.ParseFromString(graph_proto_str)
        return G
    else:
        raise ParseError(msg)
