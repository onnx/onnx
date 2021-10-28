# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List, Tuple, Text, Optional
from onnx import GraphProto

def merge(
        g1,  # type: GraphProto
        g2,  # type: GraphProto
        io_map,  # type: List[Tuple[Text, Text]]
        name=None,  # type: Optional[Text]
        doc_string=None,  # type: Optional[Text]
):  # type: (...) -> GraphProto
    """Combines two ONNX graphs into a single one.

    The combined graph is defined by connecting the specified set of outputs/inputs.

    Arguments:
        g1 (GraphProto): First graph
        g2 (GraphProto): Second graph
        io_map (list of pairs of strings): The pairs of names [(out0, in0), (out1, in1), ...]
                                           representing outputs of the first graph and inputs of the second
                                           to be connected
    """
    if type(g1) is not GraphProto:
        raise ValueError("g1 argument is not an ONNX graph")
    if type(g2) is not GraphProto:
        raise ValueError("g2 argument is not an ONNX graph")

    g1_outs = [io[0] for io in io_map]
    g2_ins = [io[1] for io in io_map]

    g = GraphProto()

    g.node.extend(g1.node)
    g.node.extend(g2.node)

    # Connecting outputs of the first graph with the inputs of the second
    for io_pair in io_map:
        g1_out_name = io_pair[0]
        g2_in_name = io_pair[1]
        found = False
        for node in g.node:
            if found:
                break
            for index, name in enumerate(node.input):
                if name == g2_in_name:
                    node.input[index] = g1_out_name
                    found = True
                    break
        if not found:
            raise ValueError(f"Could not find an input named \"{g2_in_name}\" in g2")

    g.input.extend(g1.input)
    g.input.extend([item for item in g2.input if item.name not in g2_ins])

    g.output.extend([item for item in g1.output if item.name not in g1_outs])
    g.output.extend(g2.output)

    g.initializer.extend(g1.initializer)
    g.initializer.extend(g2.initializer)

    g.sparse_initializer.extend(g1.sparse_initializer)
    g.sparse_initializer.extend(g2.sparse_initializer)

    g.value_info.extend(g1.value_info)
    g.value_info.extend(g2.value_info)

    if name:
        g.name = name
    else:
        g.name = g1.name + "__" + g2.name

    if doc_string:
        g.doc_string = doc_string
    else:
        g.doc_string = f"Graph combining {g1.name} and {g2.name}\n" + \
            g1.name + "\n\n" + g1.doc_string + "\n\n" + g2.name + "\n\n" + g2.doc_string

    return g
