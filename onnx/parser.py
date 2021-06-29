# SPDX-License-Identifier: Apache-2.0

import onnx
from typing import Text


class ParseError(Exception):
    pass


def parse_model(model_text):  # type: (Text) -> onnx.ModelProto
    (success, msg, model_proto_str) = onnx.onnx_cpp2py_export.parser.parse_model(model_text)
    if success:
        return onnx.load_from_string(model_proto_str)
    else:
        raise ParseError(msg)


def parse_graph(graph_text):  # type: (Text) -> onnx.GraphProto
    (success, msg, graph_proto_str) = onnx.onnx_cpp2py_export.parser.parse_graph(graph_text)
    if success:
        G = onnx.GraphProto()
        G.ParseFromString(graph_proto_str)
        return G
    else:
        raise ParseError(msg)
