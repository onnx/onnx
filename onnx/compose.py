# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List, Tuple, Text, Optional, MutableMapping
from onnx import ModelProto, GraphProto, helper, checker
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
        producer_name='onnx.compose.merge_models',  # type: Optional[Text]
        producer_version="1.0",  # type: Optional[Text]
        domain="",  # type: Optional[Text]
        model_version=1  # type: Optional[int]
):  # type: (...) -> ModelProto
    """Combines two ONNX models into a single one.

    The combined model is defined by connecting the specified set of outputs/inputs. Those inputs/outputs
    not specified in the io_map argument will remain as inputs/outputs of the combined model.

    Both models should have the same IR version, and same operator sets imported.

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
        producer_version (string): Optional producer version for the combined model. Default: "1.0"
        domain (string): Optional domain of the combined model. Default: ""
        model_version (int): Optional version of the graph encoded. Default: 1
    """
    if type(m1) is not ModelProto:
        raise ValueError("m1 argument is not an ONNX model")
    if type(m2) is not ModelProto:
        raise ValueError("m2 argument is not an ONNX model")

    if m1.ir_version != m2.ir_version:
        raise ValueError(
            f"IR version mismatch {m1.ir_version} != {m2.ir_version}."
            " Both models should have have the same IR version")
    ir_version = m1.ir_version

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

    graph = merge_graphs(m1.graph, m2.graph, io_map, name, doc_string)
    model = helper.make_model(graph,
                              producer_name=producer_name,
                              producer_version=producer_version,
                              domain=domain,
                              model_version=model_version,
                              opset_imports=opset_imports,
                              ir_version=ir_version)

    # Merging model metadata props
    model_props = {}
    for meta_entry in m1.metadata_props:
        model_props[meta_entry.key] = meta_entry.value
    for meta_entry in m2.metadata_props:
        if meta_entry.key in model_props:
            value = model_props[meta_entry.key]
            if value != meta_entry.value:
                raise ValueError(
                    "Can't merge models with different values for the same model metadata property."
                    f" Found: property = {meta_entry.key}, with values {value} and {meta_entry.value}."
                )
        else:
            model_props[meta_entry.key] = meta_entry.value
    helper.set_model_props(model, model_props)

    # Merging functions
    model.functions.MergeFrom(m1.functions)
    model.functions.MergeFrom(m2.functions)

    checker.check_model(model)
    return model


def add_prefix(
        model,  # type: ModelProto
        prefix,  # type: Text
        rename_nodes=True,  # type: Optional[bool]
        rename_edges=True,  # type: Optional[bool]
        rename_inputs=True,  # type: Optional[bool]
        rename_outputs=True,  # type: Optional[bool]
        inplace=False,  # type: Optional[bool]
):  # type: (...) -> ModelProto
    """Adds a prefix to names of elements in a graph: Nodes, Edges, Inputs, Outputs

    It can be used as a utility before merging graphs that have overlapping names.
    Empty names are not _prefixed.

    Arguments:
        g (ModelProto): Model
        prefix (Text): Prefix to be added to each name in the graph
        rename_nodes (bool): Whether to prefix node names
        rename_edges (bool): Whether to prefix node edge names
        rename_inputs (bool): Whether to prefix input names
        rename_outputs (bool): Whether to prefix output names
        inplace (bool): If True, mutates the model directly.
                        Otherwise, a copy will be created
    """
    if type(model) is not ModelProto:
        raise ValueError("model argument is not an ONNX model")

    if not inplace:
        m = ModelProto()
        m.CopyFrom(model)
        model = m

    g = model.graph

    def _prefixed(prefix, name):  # type: (Text, Text) -> Text
        return prefix + name if len(name) > 0 else name

    name_map = {}
    if rename_edges:
        for n in g.node:
            for e in n.input:
                name_map[e] = _prefixed(prefix, e)
            for e in n.output:
                name_map[e] = _prefixed(prefix, e)
    else:
        if rename_outputs:
            for entry in g.output:
                name_map[entry.name] = _prefixed(prefix, entry.name)
        if rename_inputs:
            for entry in g.input:
                name_map[entry.name] = _prefixed(prefix, entry.name)

    if rename_nodes:
        for n in g.node:
            n.name = _prefixed(prefix, n.name)

    for n in g.node:
        for i in range(len(n.output)):
            if n.output[i] in name_map:
                n.output[i] = name_map[n.output[i]]
        for i in range(len(n.input)):
            if n.input[i] in name_map:
                n.input[i] = name_map[n.input[i]]

    for in_desc in g.input:
        if in_desc.name in name_map:
            in_desc.name = name_map[in_desc.name]
    for out_desc in g.output:
        if out_desc.name in name_map:
            out_desc.name = name_map[out_desc.name]

    return model


def expand_out_dim(
        model,  # type: ModelProto
        dim_idx,  # type: int
        inplace=False,  # type: Optional[bool]
):  # type: (...) -> ModelProto
    """Inserts an extra dimension with extent 1 to each output in the graph.

    Inserts and Unsqueeze node for each output. It can be used as a utility before merging graphs,
    for example when the second one expects a batch dimension.

    Arguments:
        model (ModelProto): Model
        dim_idx (int): Index of the dimension to be inserted.
                       A negative value means counting dimensions from the back.
        inplace (bool): If True, mutates the model directly.
                        Otherwise, a copy will be created
    """
    if type(model) is not ModelProto:
        raise ValueError("m argument is not an ONNX model")

    if not inplace:
        m = ModelProto()
        m.CopyFrom(model)
        model = m

    g = model.graph

    orig_out_names = [output.name for output in g.output]

    for n in g.node:
        for i in range(len(n.output)):
            if n.output[i] in orig_out_names:
                n.output[i] = n.output[i] + f'_collapsed_dim_{dim_idx}'
        for i in range(len(n.input)):
            if n.input[i] in orig_out_names:
                n.input[i] = n.input[i] + f'_collapsed_dim_{dim_idx}'

    expand_dim_k = g.name + "_expand_out_dim_idx"
    g.node.append(
        helper.make_node(
            'Constant', inputs=[], outputs=[expand_dim_k], name=f"{expand_dim_k}-constant",
            value=helper.make_tensor(name=f"{expand_dim_k}-value", data_type=tp.INT64,
                                     dims=[1, ], vals=[dim_idx, ]))
    )

    for _ in range(len(g.output)):
        o = g.output.pop(0)
        prev_output = o.name + f'_collapsed_dim_{dim_idx}'
        g.node.append(
            helper.make_node('Unsqueeze', inputs=[prev_output, expand_dim_k],
                             outputs=[o.name], name=f"unsqueeze-{o.name}")
        )
        new_shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
        new_shape.insert(dim_idx, 1)
        g.output.append(
            helper.make_tensor_value_info(o.name, o.type.tensor_type.elem_type, new_shape))
    return model
