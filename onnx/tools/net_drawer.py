# SPDX-License-Identifier: Apache-2.0
# A library and utility for drawing ONNX nets. Most of this implementation has
# been borrowed from the caffe2 implementation
# https://github.com/caffe2/caffe2/blob/master/caffe2/python/net_drawer.py
#
# The script takes two required arguments:
#   -input: a path to a serialized ModelProto .pb file.
#   -output: a path to write a dot file representation of the graph
#
# Given this dot file representation, you can-for example-export this to svg
# with the graphviz `dot` utility, like so:
#
#   $ dot -Tsvg my_output.dot -o my_output.svg

import argparse
from collections import defaultdict
import json
from onnx import ModelProto, GraphProto, NodeProto
import pydot  # type: ignore
from typing import Text, Any, Callable, Optional, Dict


OP_STYLE = {
    'shape': 'box',
    'color': '#0F9D58',
    'style': 'filled',
    'fontcolor': '#FFFFFF'
}

BLOB_STYLE = {'shape': 'octagon'}

_NodeProducer = Callable[[NodeProto, int], pydot.Node]


def _escape_label(name: Text) -> Text:
    # json.dumps is poor man's escaping
    return json.dumps(name)


def _form_and_sanitize_docstring(s: Text) -> Text:
    url = 'javascript:alert('
    url += _escape_label(s).replace('"', '\'').replace('<', '').replace('>', '')
    url += ')'
    return url


def GetOpNodeProducer(embed_docstring: bool = False, **kwargs: Any) -> _NodeProducer:
    def ReallyGetOpNode(op: NodeProto, op_id: int) -> pydot.Node:
        if op.name:
            node_name = '%s/%s (op#%d)' % (op.name, op.op_type, op_id)
        else:
            node_name = '%s (op#%d)' % (op.op_type, op_id)
        for i, input in enumerate(op.input):
            node_name += '\n input' + str(i) + ' ' + input
        for i, output in enumerate(op.output):
            node_name += '\n output' + str(i) + ' ' + output
        node = pydot.Node(node_name, **kwargs)
        if embed_docstring:
            url = _form_and_sanitize_docstring(op.doc_string)
            node.set_URL(url)
        return node
    return ReallyGetOpNode


def GetPydotGraph(
    graph: GraphProto,
    name: Optional[Text] = None,
    rankdir: Text = 'LR',
    node_producer: Optional[_NodeProducer] = None,
    embed_docstring: bool = False,
) -> pydot.Dot:
    if node_producer is None:
        node_producer = GetOpNodeProducer(embed_docstring=embed_docstring, **OP_STYLE)
    pydot_graph = pydot.Dot(name, rankdir=rankdir)
    pydot_nodes: Dict[Text, pydot.Node] = {}
    pydot_node_counts: Dict[Text, int] = defaultdict(int)
    for op_id, op in enumerate(graph.node):
        op_node = node_producer(op, op_id)
        pydot_graph.add_node(op_node)
        for input_name in op.input:
            if input_name not in pydot_nodes:
                input_node = pydot.Node(
                    _escape_label(
                        input_name + str(pydot_node_counts[input_name])),
                    label=_escape_label(input_name),
                    **BLOB_STYLE
                )
                pydot_nodes[input_name] = input_node
            else:
                input_node = pydot_nodes[input_name]
            pydot_graph.add_node(input_node)
            pydot_graph.add_edge(pydot.Edge(input_node, op_node))
        for output_name in op.output:
            if output_name in pydot_nodes:
                pydot_node_counts[output_name] += 1
            output_node = pydot.Node(
                _escape_label(
                    output_name + str(pydot_node_counts[output_name])),
                label=_escape_label(output_name),
                **BLOB_STYLE
            )
            pydot_nodes[output_name] = output_node
            pydot_graph.add_node(output_node)
            pydot_graph.add_edge(pydot.Edge(op_node, output_node))
    return pydot_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX net drawer")
    parser.add_argument(
        "--input",
        type=Text, required=True,
        help="The input protobuf file.",
    )
    parser.add_argument(
        "--output",
        type=Text, required=True,
        help="The output protobuf file.",
    )
    parser.add_argument(
        "--rankdir", type=Text, default='LR',
        help="The rank direction of the pydot graph.",
    )
    parser.add_argument(
        "--embed_docstring", action="store_true",
        help="Embed docstring as javascript alert. Useful for SVG format.",
    )
    args = parser.parse_args()
    model = ModelProto()
    with open(args.input, 'rb') as fid:
        content = fid.read()
        model.ParseFromString(content)
    pydot_graph = GetPydotGraph(
        model.graph,
        name=model.graph.name,
        rankdir=args.rankdir,
        node_producer=GetOpNodeProducer(
            embed_docstring=args.embed_docstring,
            **OP_STYLE
        ),
    )
    pydot_graph.write_dot(args.output)


if __name__ == '__main__':
    main()
