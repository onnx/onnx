# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from typing import List, Tuple, Text, Optional, MutableMapping
from google.protobuf.internal.containers import RepeatedScalarFieldContainer
from onnx import ModelProto, GraphProto, OperatorSetIdProto, helper, checker
from onnx import TensorProto as tp


def merge_graphs(
        g1,  # type: GraphProto
        g2,  # type: GraphProto
        io_map,  # type: List[Tuple[Text, Text]]
        name=None,  # type: Optional[Text]
        doc_string=None,  # type: Optional[Text]
):  # type: (...) -> GraphProto
    """Combines two ONNX graphs into a single one.

    The combined graph is defined by connecting the specified set of outputs/inputs. Those inputs/outputs
    not specified in the io_map argument will remain as inputs/outputs of the combined graph.

    Arguments:
        g1 (GraphProto): First graph
        g2 (GraphProto): Second graph
        io_map (list of pairs of string): The pairs of names [(out0, in0), (out1, in1), ...]
                                          representing outputs of the first graph and inputs of the second
                                          to be connected
        name (string): Optional name for the combined graph
                       By default, the name is g1.name and g2.name concatenated with an undescore delimiter
        doc_string (string): Optional docstring for the combined graph
                             If not provided, a default docstring with the concatenation of g1 and g2 docstrings is used
    """
    if type(g1) is not GraphProto:
        raise ValueError("g1 argument is not an ONNX graph")
    if type(g2) is not GraphProto:
        raise ValueError("g2 argument is not an ONNX graph")

    io_map_g1_outs = [io[0] for io in io_map]
    io_map_g2_ins = [io[1] for io in io_map]
    g1_outs = [o.name for o in g1.output]
    g2_ins = [i.name for i in g2.input]

    for g1_out_name, g2_in_name in io_map:
        if g1_out_name not in g1_outs:
            raise ValueError(f"Output {g1_out_name} not present in g1")
        if g2_in_name not in g2_ins:
            raise ValueError(f"Input {g2_in_name} not present in g2")

    g = GraphProto()

    g.node.extend(g1.node)
    g.node.extend(g2.node)

    # Connecting outputs of the first graph with the inputs of the second
    for g1_out_name, g2_in_name in io_map:
        for node in g.node:
            for index, name in enumerate(node.input):
                if name == g2_in_name:
                    node.input[index] = g1_out_name

    g.input.extend(g1.input)
    g.input.extend(
        [item for item in g2.input if item.name not in io_map_g2_ins])

    g.output.extend(
        [item for item in g1.output if item.name not in io_map_g1_outs])
    g.output.extend(g2.output)

    g.initializer.extend(g1.initializer)
    g.initializer.extend(g2.initializer)

    g.sparse_initializer.extend(g1.sparse_initializer)
    g.sparse_initializer.extend(g2.sparse_initializer)

    g.value_info.extend(g1.value_info)
    g.value_info.extend(g2.value_info)

    g.name = name if name is not None else "_".join([g1.name, g2.name])

    if doc_string is None:
        doc_string = f"Graph combining {g1.name} and {g2.name}\n" + \
            g1.name + "\n\n" + g1.doc_string + "\n\n" + g2.name + "\n\n" + g2.doc_string
    g.doc_string = doc_string

    return g


def merge_models(
        m1,  # type: ModelProto
        m2,  # type: ModelProto
        io_map,  # type: List[Tuple[Text, Text]]
        name=None,  # type: Optional[Text]
        doc_string=None,  # type: Optional[Text]
        producer_name='onnx.compose',  # type: Optional[Text]
        ir_version=None,  # type: Optional[int]
):  # type: (...) -> ModelProto
    """Combines two ONNX models into a single one.

    The combined model is defined by connecting the specified set of outputs/inputs. Those inputs/outputs
    not specified in the io_map argument will remain as inputs/outputs of the combined model.

    Arguments:
        m1 (ModelProto): First model
        m2 (ModelProto): Second model
        io_map (list of pairs of string): The pairs of names [(out0, in0), (out1, in1), ...]
                                          representing outputs of the first graph and inputs of the second
                                          to be connected
        name (string): Optional name for the combined graph
                       By default, the name is g1.name and g2.name concatenated with an undescore delimiter
        doc_string (string): Optional docstring for the combined graph
                             If not provided, a default docstring with the concatenation of g1 and g2 docstrings is used
        producer_name (string): Optional producer name for the combined model. Default: 'onnx.compose'
        ir_version (int): Optional target IR version. By default the highest IR version from the two graphs will be used
                          The target IR version should be compatible with the Operator Set versions used by
                          the models to be combined.
    """
    if type(m1) is not ModelProto:
        raise ValueError("m1 argument is not an ONNX model")
    if type(m2) is not ModelProto:
        raise ValueError("m2 argument is not an ONNX model")

    opset_import_map = {}  # type: MutableMapping[Text, int]
    opset_imports = \
        [entry for entry in m1.opset_import] + \
        [entry for entry in m2.opset_import]

    for entry in opset_imports:
        if entry.domain in opset_import_map:
            found_version = opset_import_map[entry.domain]
            if entry.version != found_version:
                raise ValueError(
                    "Can't merge two models with different operator set ids for a given domain. "
                    f"Got: {m1.opset_import} and {m2.opset_import}")
        else:
            opset_import_map[entry.domain] = entry.version

    min_ir_version = helper.find_min_ir_version_for(
        [entry for entry in opset_imports])
    if ir_version is None:
        ir_version = max(min_ir_version, max(
            m1.ir_version, m2.ir_version))
    if ir_version < min_ir_version:
        raise ValueError(
            f"IR version {ir_version} is not sufficient to support "
            f"the models operator set ids: {opset_imports}")

    graph = merge_graphs(m1.graph, m2.graph, io_map, name, doc_string)
    model = helper.make_model(graph, producer_name=producer_name,
                              opset_imports=opset_imports,
                              ir_version=ir_version)
    checker.check_model(model)
    return model


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
    if type(g) is not GraphProto:
        raise ValueError("g argument is not an ONNX graph")

    g = copy.deepcopy(g)

    def prefixed(prefix, name):  # type: (Text, Text) -> Text
        return prefix + name if len(name) > 0 else name

    if rename_nodes:
        for n in g.node:
            n.name = prefixed(prefix, n.name)

    if rename_edges:
        def clear(lst):  # type: (RepeatedScalarFieldContainer[Text]) -> None
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
    if type(g) is not GraphProto:
        raise ValueError("g argument is not an ONNX graph")

    g = copy.deepcopy(g)

    expand_dim_k = g.name + "_expand_dim_idx"
    g.node.append(
        helper.make_node(
            'Constant', inputs=[], outputs=[expand_dim_k], name=f"{expand_dim_k}-constant",
            value=helper.make_tensor(name=f"{expand_dim_k}-value", data_type=tp.INT64,
                                     dims=[1, ], vals=[dim_idx, ]))
    )

    for _ in range(len(g.output)):
        o = g.output.pop(0)
        new_name = o.name + '_expanded'
        g.node.append(
            helper.make_node('Unsqueeze', inputs=[o.name, expand_dim_k],
                             outputs=[new_name], name=f"unsqueeze-{o.name}")
        )
        new_shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
        new_shape.insert(dim_idx, 1)
        g.output.append(
            helper.make_tensor_value_info(new_name, o.type.tensor_type.elem_type, new_shape))
    return g
