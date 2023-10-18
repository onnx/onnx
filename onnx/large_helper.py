# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Optional, Sequence, Tuple

from onnx import IR_VERSION, FunctionProto, GraphProto, ModelProto, OperatorSetIdProto
from onnx.large_proto import LargeModelProto
from onnx.helper import make_model


def make_large_model(
    graph: GraphProto,
    large_initializers: Optional[List[Tuple[GraphProto, str, Any]]] = None,
    **kwargs: Any,
) -> ModelProto:
    """Construct a LargeModelProto

    Arguments:
        graph: *make_graph* returns
        large_initializers: list of tuple(graph, name, large tensor),
            graph is the GraphProto instance the initializer belongs to,
            name is the name of the initializer (dense),
            large tensor is any python object supporting the DLPack protocol,
            the ownership the tensor is transfered to the LargeModelProto
        **kwargs: any attribute to add to the returned instance
    Returns:
        ModelProto
    """
    model = make_model(graph, **kwargs)
    large_model = LargeModelProto()
    large_model.set_model_proto(model)
    if large_initializers:
        raise NotImplementedError("Large initializer are not implemented yet.")
    return large_model
