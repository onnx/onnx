from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import re

from typing import List, Text, Sequence, Any, Union
import numpy as np  # type: ignore

import onnx
import onnx.mapping

from ..utils import import_recursive
from ..test_case import TestCase


_NodeTestCases = []

from onnx.onnx_pb import NodeProto, AttributeProto
from onnx.onnx_operators_pb import FunctionProto


# FIXME(TMVector): Any reason we can't get rid of this and use the C++ helper directly?
def function_expand_helper(node,  # type: NodeProto
                           function_proto,  # type: FunctionProto
                           op_prefix  # type:  Text
                           ):  # type:  (...) -> List[NodeProto]
    node_list = []
    io_names_map = dict()
    attribute_map = dict((a.name, a) for a in node.attribute)

    for idx in range(len(function_proto.input)):
        io_names_map[function_proto.input[idx]] = node.input[idx] \
            if idx in range(len(node.input)) else ""

    for idx in range(len(function_proto.output)):
        # Even if the node has been created with optional outputs missing, we
        # can't assume that the function body handles this correctly, such as in
        # the case that output is also an intermediate value.
        # So we only add a name mapping if the output is present. An internal
        # name will be generated if the missing output is used, the same as any
        # other internal tensor.
        if idx in range(len(node.output)) and node.output[idx] != "":
            io_names_map[function_proto.output[idx]] = node.output[idx]

    for internal_node in function_proto.node:
        new_node = NodeProto()
        new_node.CopyFrom(internal_node)
        new_node.ClearField("input")
        new_node.ClearField("output")
        new_node.ClearField("attribute")
        for internal_name in internal_node.input:
            if internal_name in io_names_map:
                new_node.input.append(io_names_map[internal_name])
            else:
                new_node.input.append(op_prefix + internal_name)
        for internal_name in internal_node.output:
            if internal_name in io_names_map:
                new_node.output.append(io_names_map[internal_name])
            else:
                new_node.output.append(op_prefix + internal_name)
        for attr in internal_node.attribute:
            if attr.HasField("ref_attr_name"):
                if attr.ref_attr_name in attribute_map:
                    new_attr = AttributeProto()
                    new_attr.CopyFrom(attribute_map[attr.ref_attr_name])  # type: ignore
                    new_attr.name = attr.name
                    new_node.attribute.extend([new_attr])
            else:
                new_attr = AttributeProto()
                new_attr.CopyFrom(attr)
                new_node.attribute.extend([new_attr])
        node_list.append(new_node)
    return node_list


def function_testcase_helper(node, name):  # type: (NodeProto, Text) -> List[NodeProto]
    test_op = node.op_type
    op_prefix = test_op + "_" + name + "_expanded_function"
    schema = onnx.defs.get_schema(test_op, node.domain)

    if schema.has_function:    # type: ignore
        function_proto = schema.function_body  # type: ignore
    elif schema.has_context_dependent_function:    # type: ignore
        function_proto_str = schema.get_context_dependent_function(node.SerializeToString())  # type: ignore
        function_proto = FunctionProto()
        function_proto.ParseFromString(function_proto_str)
    else:
        return []

    for attr in schema.attributes:
        if attr in [a.name for a in node.attribute]:
            continue
        if schema.attributes[attr].default_value:
            node.attribute.extend([schema.attributes[attr].default_value])

    # function_proto.attributes
    node_list = function_expand_helper(node, function_proto, op_prefix)
    return node_list


def _extract_value_info(input, name, ele_type=None):  # type: (Union[List[Any], np.ndarray], Text, np.dtype) -> onnx.ValueInfoProto
    if isinstance(input, list):
        # TODO: Account for recursive sequence case. Right now, this function supports
        # Sequences of Tensors.
        return onnx.helper.make_sequence_value_info(
            name=name,
            elem_type=ele_type if ele_type else onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input[0].dtype],
            shape=None
        )
    return onnx.helper.make_tensor_value_info(
        name=name,
        elem_type=ele_type if ele_type else onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input.dtype],
        shape=input.shape)


def expect(node,  # type: onnx.NodeProto
           inputs,  # type: Sequence[np.ndarray]
           outputs,  # type: Sequence[np.ndarray]
           name,  # type: Text
           **kwargs  # type: Any
           ):  # type: (...) -> None
    present_inputs = [x for x in node.input if (x != '')]
    present_outputs = [x for x in node.output if (x != '')]
    input_types = [None] * len(inputs)
    if 'input_types' in kwargs:
        input_types = kwargs[str('input_types')]
        del kwargs[str('input_types')]
    output_types = [None] * len(outputs)
    if 'output_types' in kwargs:
        output_types = kwargs[str('output_types')]
        del kwargs[str('output_types')]
    inputs_vi = [_extract_value_info(arr, arr_name, input_type)
                 for arr, arr_name, input_type in zip(inputs, present_inputs, input_types)]
    outputs_vi = [_extract_value_info(arr, arr_name, output_type)
                  for arr, arr_name, output_type in zip(outputs, present_outputs, output_types)]
    graph = onnx.helper.make_graph(
        nodes=[node],
        name=name,
        inputs=inputs_vi,
        outputs=outputs_vi)
    kwargs[str('producer_name')] = 'backend-test'
    model = onnx.helper.make_model(graph, **kwargs)

    _NodeTestCases.append(TestCase(
        name=name,
        model_name=name,
        url=None,
        model_dir=None,
        model=model,
        data_sets=[(inputs, outputs)],
        kind='node',
        rtol=1e-3,
        atol=1e-7,
    ))

    expanded_function_nodes = function_testcase_helper(node, name)
    if expanded_function_nodes:
        function_test_name = name + '_expanded'
        graph = onnx.helper.make_graph(
            nodes=expanded_function_nodes,
            name=function_test_name,
            inputs=inputs_vi,
            outputs=outputs_vi)
        kwargs[str('producer_name')] = 'backend-test'
        model = onnx.helper.make_model(graph, **kwargs)
        _NodeTestCases.append(TestCase(
            name=function_test_name,
            model_name=function_test_name,
            url=None,
            model_dir=None,
            model=model,
            data_sets=[(inputs, outputs)],
            kind='node',
            rtol=1e-3,
            atol=1e-7,
        ))


def collect_testcases():  # type: () -> List[TestCase]
    '''Collect node test cases defined in python/numpy code.
    '''
    import_recursive(sys.modules[__name__])
    return _NodeTestCases
