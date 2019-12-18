from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six import string_types
from typing import Any, Dict, List, Optional, Set, Text, Union
from onnx import ModelProto, ValueInfoProto

import onnx.checker


def update_inputs_outputs_dims(model, input_dims, output_dims):  # type: (ModelProto, Dict[Text, List[Any]], Dict[Text, List[Any]]) -> ModelProto
    """
        This function updates the dimension sizes of the model's inputs and outputs to the values
        provided in input_dims and output_dims. if the dim value provided is negative, a unique dim_param
        will be set for that dimension.

        Example. if we have the following shape for inputs and outputs:
                shape(input_1) = ('b', 3, 'w', 'h')
                shape(input_2) = ('b', 4)
                and shape(output)  = ('b', 'd', 5)

            The parameters can be provided as:
                input_dims = {
                    "input_1": ['b', 3, 'w', 'h'],
                    "input_2": ['b', 4],
                }
                output_dims = {
                    "output": ['b', -1, 5]
                }

            Putting it together:
                model = onnx.load('model.onnx')
                updated_model = update_inputs_outputs_dims(model, input_dims, output_dims)
                onnx.save(updated_model, 'model.onnx')
    """
    dim_param_set = set()  # type: Set[Text]

    def init_dim_param_set(dim_param_set, value_infos):  # type: (Set[Text], List[ValueInfoProto]) -> None
        for info in value_infos:
            shape = info.type.tensor_type.shape
            for dim in shape.dim:
                if dim.HasField('dim_param'):
                    dim_param_set.add(dim.dim_param)  # type: ignore

    init_dim_param_set(dim_param_set, model.graph.input)  # type: ignore
    init_dim_param_set(dim_param_set, model.graph.output)  # type: ignore
    init_dim_param_set(dim_param_set, model.graph.value_info)  # type: ignore

    def update_dim(tensor, dim, j, name):  # type: (ValueInfoProto, Any, int, Text) -> None
        dim_proto = tensor.type.tensor_type.shape.dim[j]
        if isinstance(dim, int):
            if dim >= 0:
                if dim_proto.HasField('dim_value') and dim_proto.dim_value != dim:
                    raise ValueError('Unable to set dimension value to {} for axis {} of {}. Contradicts existing dimension value {}.'
                        .format(dim, j, name, dim_proto.dim_value))
                dim_proto.dim_value = dim
            else:
                generated_dim_param = name + '_' + str(j)
                if generated_dim_param in dim_param_set:
                    raise ValueError('Unable to generate unique dim_param for axis {} of {}. Please manually provide a dim_param value.'
                        .format(j, name))
                dim_proto.dim_param = generated_dim_param
        elif isinstance(dim, string_types):
            dim_proto.dim_param = dim
        else:
            raise ValueError('Only int or str is accepted as dimension value, incorrect type: {}'.format(type(dim)))

    for input in model.graph.input:
        input_name = input.name
        input_dim_arr = input_dims[input_name]
        for j, dim in enumerate(input_dim_arr):
            update_dim(input, dim, j, input_name)

    for output in model.graph.output:
        output_name = output.name
        output_dim_arr = output_dims[output_name]
        for j, dim in enumerate(output_dim_arr):
            update_dim(output, dim, j, output_name)

    onnx.checker.check_model(model)
    return model


def partial_update_dims(model, input_dim=None, output_dim=None):  # type: (ModelProto, Optional[Dict[int, Union[int, Text]]], Optional[Dict[int, Union[int, Text]]]) -> ModelProto
    """
    This function updates the dimension sizes of the model's inputs and outputs to the values
    provided in input_dims and output_dims. if the dim value provided is negative, a unique dim_param
    will be set for that dimension.

    The difference with update_inputs_outputs_dims is that only the dimensions that should be changed need
    to be provided. This simplifies making models symbolic (e.g., with respect to batch or length parameters) or
    instantiating them.

        Example. if we have the following shape for inputs and outputs:
                shape(input_1) = (128, 'l', 64, 64)
                shape(input_2) = (128, 'w')
                and shape(output)  = (128, 'c', 5)
                The parameters can be provided as:
                input_dim = {
                    0: 'b',
                    1: 20
                }
                output_dim = {
                    0: -1
                }
                producing the following result:
                shape(input_1) = ('b', 20, 64, 64)
                shape(input_2) = ('b', 20)
                and shape(output)  = ('output.0', 'c', 5)

    :param model: Onnx Protobuf model to be modified
    :param input_dim: Dictionary of zero-based dimension indices and corresponding values (int or string) to be set. -1 for unique string
    :param output_dim: Dictionary of zero-based dimension indices and corresponding values (int or string) to be set. -1 for unique string
    :return: Model with modified inputs and outputs
    """
    input_dim = input_dim or {}
    output_dim = output_dim or {}
    if not (bool(input_dim) and bool(output_dim)):
        return model

    # Separate graph inputs from initializers from graph.input set
    inputs = set(i.name for i in model.graph.input)
    initializers = set(i.name for i in model.graph.initializer)
    graph_inputs = inputs - initializers

    outputs = model.graph.output

    # Create complete dictionary of input dimensions. Only Graph inputs are modified as specified in input_dim
    inp = {}
    for i in inputs:
        # Gather input i, inputs only contains strings
        input = [j for j in model.graph.input if j.name == i][0]
        dimensions = input.type.tensor_type.shape.dim

        ls = []

        if len(input_dim) == 0:
            break

        for d in dimensions:
            if d.HasField('dim_param'):
                ls.append(d.dim_param)
            if d.HasField('dim_value'):
                ls.append(d.dim_value)

        # Only change graph_input dimensions
        if input.name in graph_inputs:
            # Throw error in case specified dimension is too large (does not exist in inputs)
            if len(ls) <= max(input_dim.keys()):
                ValueError('Input {} has only {} dimensions, less than {} that are given'.format(input.name, len(ls),
                                                                                                 max(input_dim.keys())))
            for k in input_dim.keys():
                ls[k] = input_dim[k]

        inp[input.name] = ls

    # Create complete dictionary of output dimensions. All outputs are considered to be Graph outputs
    out = {}
    for o in outputs:
        dimensions = o.type.tensor_type.shape.dim

        ls = []

        if len(output_dim) == 0:
            break

        for d in dimensions:
            if d.HasField('dim_param'):
                ls.append(d.dim_param)
            if d.HasField('dim_value'):
                ls.append(d.dim_value)

        if len(ls) <= max(output_dim.keys()):
            ValueError('Input {} has only {} Dimensions, less than {} that are given'.format(o.name, len(ls),
                                                                                             max(output_dim.keys())))
        for k in output_dim.keys():
            ls[k] = output_dim[k]

        out[o.name] = ls

    return update_inputs_outputs_dims(model, inp, out)
