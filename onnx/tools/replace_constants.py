# SPDX-License-Identifier: Apache-2.0
# pylint: disable=too-many-statements,too-many-branches
from typing import Optional, Union
import numpy as np
from onnx import AttributeProto, FunctionProto, GraphProto, ModelProto, TensorProto
from onnx.helper import (
    make_attribute,
    make_graph,
    make_model,
    make_node,
    make_tensor_value_info,
    set_model_props,
    tensor_dtype_to_np_dtype,
)
from onnx.numpy_helper import from_array


def replace_initializer_by_constant_of_shape(
    onx: Union[FunctionProto, GraphProto, ModelProto],
    threshold: int = 128,
    ir_version: Optional[int] = None,
):
    """
    Replaces initializers by nodes *ConstantOfShape* to reduce
    the size and still write a unit test.

    :param onx: ModelProto
    :param threshold: every initializer under
        this threshold is not impacted
    :param ir_version: initializer must be specified as input for `ir_version <= 3`,
        this must be specified if onx is :class:`FunctionProto` or :class:`GraphProto`
    :return: onx, modified ModelProto
    """
    if isinstance(onx, FunctionProto):
        for node in onx.node:
            if node.op_type == "Constant":
                raise NotImplementedError(f"Node {node.op_type!r} is not handled yet.")
        return onx
    if isinstance(onx, ModelProto):
        new_graph = replace_initializer_by_constant_of_shape(
            onx.graph, ir_version=ir_version or onx.ir_version, threshold=threshold
        )
        new_functions = [
            replace_initializer_by_constant_of_shape(
                f, threshold=threshold, ir_version=ir_version or onx.ir_version
            )
            for f in onx.functions
        ]
        model = make_model(
            new_graph,
            functions=new_functions,
            producer_name=onx.producer_name,
            producer_version=onx.producer_version,
            ir_version=ir_version or onx.ir_version,
            doc_string=onx.doc_string,
            domain=onx.domain,
            model_version=onx.model_version,
        )
        if len(onx.metadata_props) > 0:  # pragma: no cover
            values = {p.key: p.value for p in onx.metadata_props}
            set_model_props(model, values)

        del model.opset_import[:]  # pylint: disable=E1101
        for oimp in onx.opset_import:
            op_set = model.opset_import.add()  # pylint: disable=E1101
            if oimp.domain == "" and oimp.version < 9:
                raise RuntimeError(
                    f"ConstantOfShape was introduced in "
                    f"opset 9 but opset is {oimp.version}."
                )
            op_set.domain = oimp.domain
            op_set.version = oimp.version
        return model

    if not isinstance(onx, GraphProto):
        raise TypeError(f"onx should be a GraphProto as this stage not {type(onx)}.")

    new_nodes = []
    removed = set()
    additional_inputs = []

    new_inits: Sequence[TensorProto] = []
    for init in onx.initializer:
        dims = tuple(init.dims)
        size = np.prod(dims)
        if size <= threshold:
            new_inits.append(init)
            continue
        new_name = f"{init.name}__SHAPE"
        new_inits.append(
            from_array(np.array(list(dims), dtype=np.int64), name=new_name)
        )
        dtype = tensor_dtype_to_np_dtype(init.data_type)
        node = make_node(
            "ConstantOfShape",
            [new_name],
            [init.name],
            value=from_array(np.array([0.5], dtype=dtype)),
        )
        new_nodes.append(node)
        removed.add(init.name)
        if ir_version is not None and ir_version <= 3:
            additional_inputs.append(
                make_tensor_value_info(new_name, TensorProto.INT64, [len(dims)])
            )

    new_sparse_inits: Sequence[SparseTensorProto] = []
    for init in onx.sparse_initializer:
        dims = tuple(init.dims)
        size = np.prod(dims)
        if size <= threshold:
            new_sparse_inits.append(init)
            continue
        raise NotImplementedError(
            f"This feature is not yet implemented for sparse initializer"
            f"(name={init.name!r})."
        )

    for node in onx.node:
        if node.op_type == "Constant":
            raise NotImplementedError(f"Node {node.op_type!r} is not handled yet.")
        modified = False
        atts = []
        for att in node.attribute:
            if (
                att.type == AttributeProto.GRAPH
                and hasattr(att, "g")
                and att.g is not None
            ):
                modified = True
                g = replace_initializer_by_constant_of_shape(
                    att.g, threshold=threshold, ir_version=ir_version
                )
                att = make_attribute(att.name, g)
            atts.append(att)
        if modified:
            new_node = make_node(node.op_type, node.input, node.output)
            new_node.attribute.extend(atts)
            new_nodes.append(node)
        else:
            new_nodes.append(node)

    graph = make_graph(
        new_nodes,
        onx.name,
        [i for i in onx.input if i.name not in removed] + additional_inputs,
        onx.output,
        initializer=new_inits,
        sparse_initializer=new_sparse_inits,
    )
    return graph
