# SPDX-License-Identifier: Apache-2.0

import onnx
import onnx.onnx_cpp2py_export.parser as C

class ParseError(Exception):
    pass

def parse_model(model_text):  # type: (bytes) -> ModelProto
    (success, msg, model_proto_str) = C.parse_model(model_text)
    if success:
        return onnx.load_from_string(model_proto_str)
    else:
        raise ParseError(msg)

def parse_graph(graph_text):  # type: (bytes) -> GraphProto
    (success, msg, graph_proto_str) = C.parse_graph(graph_text)
    if success:
        G = onnx.GraphProto()
        G.ParseFromString(graph_proto_str)
        return G
    else:
        raise ParseError(msg)
        