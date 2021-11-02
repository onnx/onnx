# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from typing import List, Tuple, Text, Optional
from onnx import GraphProto, helper
from onnx import TensorProto as tp

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
        g1_out_name, g2_in_name = io_pair
        found = False
        for node in g.node:
            for index, name in enumerate(node.input):
                if name == g2_in_name:
                    node.input[index] = g1_out_name
                    found = True
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


def add_prefix(
        g,  # type: GraphProto
        prefix,  # type: Text
        rename_nodes=True,  # type: Optional[bool]
        rename_edges=True,  # type: Optional[bool]
        rename_inputs=True,  # type: Optional[bool]
        rename_outputs=True,  # type: Optional[bool]
):  # type: (...) -> GraphProto
    """Adds a prefix to names of elements in a graph: Nodes, Edges, Inputs, Outputs

    It can be used as a utility before merging graphs that have overlapping names.
    Empty names are not prefixed.

    Arguments:
        g (GraphProto): Graph
        prefix (Text): Prefix to be added to each name in the graph
        rename_nodes (bool): Whether to prefix node names
        rename_edges (bool): Whether to prefix node edge names
        rename_inputs (bool): Whether to prefix input names
        rename_outputs (bool): Whether to prefix output names
    """
    g = copy.deepcopy(g)

    def prefixed(prefix, name):
        return prefix + name if len(name) > 0 else name

    if rename_nodes:
        for n in g.node:
            n.name = prefixed(prefix, n.name)

    if rename_edges:
        def clear(lst):
            for _ in range(len(lst)):
                lst.pop()

        for n in g.node:
            out_names = [prefixed(prefix, name) for name in n.output]
            clear(n.output)
            n.output.extend(out_names)

            in_names = [prefixed(prefix, name) for name in n.input]
            clear(n.input)
            n.input.extend(in_names)

    if rename_inputs:
        for i in g.input:
            i.name = prefixed(prefix, i.name)

    if rename_outputs:
        for o in g.output:
            o.name = prefixed(prefix, o.name)

    return g


def expand_out_dim(
        g,  # type: GraphProto
        dim_idx,  # type: int
):  # type: (...) -> GraphProto
    """Inserts an extra dimension with extent 1 to each output in the graph.

    Inserts and Unsqueeze node for each output. It can be used as a utility before merging graphs,
    for example when the second one expects a batch dimension.

    Arguments:
        g (GraphProto): Graph
        dim_idx (int): Index of the dimension to be inserted.
                       A negative value means counting dimensions from the back.
    """
    g = copy.deepcopy(g)

    expand_dim_k = g.name + "_expand_dim_idx"
    g.node.append(
        helper.make_node(
            'Constant', inputs=[], outputs=[expand_dim_k], name=f"{expand_dim_k}-constant",
            value=helper.make_tensor(name="{expand_dim_k}-value", data_type=tp.INT64,
                                     dims=[1,], vals=[dim_idx,]))
    )

    for _ in range(len(g.output)):
        o = g.output.pop(0)
        new_name = o.name + '_expanded'
        g.node.append(
            helper.make_node('Unsqueeze', inputs=[o.name, expand_dim_k],
                             outputs=[new_name], name=f'unsqueeze-{o.name}')
        )
        new_shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
        new_shape.insert(dim_idx, 1)
        g.output.append(
            helper.make_tensor_value_info(new_name, o.type.tensor_type.elem_type, new_shape))
    return g
