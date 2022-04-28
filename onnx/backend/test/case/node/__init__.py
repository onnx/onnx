# SPDX-License-Identifier: Apache-2.0

import sys
import re

from typing import List, Text, Sequence, Any, Union, Optional
import numpy as np  # type: ignore

import onnx
import onnx.mapping
from onnx import ModelProto, GraphProto

from ..utils import import_recursive
from ..test_case import TestCase


_NodeTestCases = []
_TargetOpType = None
_TargetOpsetVersion = None


from onnx.onnx_pb import NodeProto, AttributeProto, TypeProto, FunctionProto


# FIXME(TMVector): Any reason we can't get rid of this and use the C++ helper directly?
def function_expand_helper(node: NodeProto,
                           function_proto: FunctionProto,
                           op_prefix: Text
                           ) -> List[NodeProto]:
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


def function_testcase_helper(node: NodeProto, input_types: List[TypeProto], name: Text) -> List[NodeProto]:
    test_op = node.op_type
    op_prefix = test_op + "_" + name + "_expanded_function"
    schema = onnx.defs.get_schema(test_op, node.domain)

    if schema.has_function:    # type: ignore
        function_proto = schema.function_body  # type: ignore
    elif schema.has_context_dependent_function:    # type: ignore
        function_proto_str = schema.get_context_dependent_function(node.SerializeToString(), [t.SerializeToString() for t in input_types])  # type: ignore
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


def _extract_value_info(input: Union[List[Any], np.ndarray, None], name: Text, type_proto: Optional[TypeProto] = None) -> onnx.ValueInfoProto:
    if type_proto is None:
        if input is None:
            raise NotImplementedError("_extract_value_info: both input and type_proto arguments cannot be None.")
        elif isinstance(input, list):
            elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input[0].dtype]
            shape = None
            tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type, shape)
            type_proto = onnx.helper.make_sequence_type_proto(tensor_type_proto)
        else:
            elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input.dtype]
            shape = input.shape
            type_proto = onnx.helper.make_tensor_type_proto(elem_type, shape)

    return onnx.helper.make_value_info(name, type_proto)


def _make_test_model_gen_version(graph: GraphProto, **kwargs: Any) -> ModelProto:
    latest_onnx_version, latest_ml_version, latest_training_version = onnx.helper.VERSION_TABLE[-1][2:5]
    if "opset_imports" in kwargs:
        for domain, opset_version in kwargs["opset_imports"]:
            # If the test model uses an unreleased opset version (latest_version+1),
            # directly use make_model to create a model with the latest ir version
            if (
                ((domain == "" or domain == "ai.onnx") and opset_version == latest_onnx_version + 1)
                or (domain == "ai.onnx.ml" and opset_version == latest_ml_version + 1)
                or ((domain == "ai.onnx.training version" or domain == "ai.onnx.preview.training") and opset_version == latest_training_version + 1)
            ):
                return onnx.helper.make_model(graph, **kwargs)
    # Otherwise, find and use the corresponding ir version according to given opset version
    return onnx.helper.make_model_gen_version(graph, **kwargs)


# In the case of ops with optional inputs and outputs, node.input and node.output indicate
# which inputs/outputs are present and which are omitted. However, the parameter inputs
# and outputs of this function include values only for inputs/outputs that are present.
# E.g., for an op with 3 inputs, if the second parameter is optional and we wish to omit it,
# node.inputs would look like ["Param1", "", "Param3"], while inputs would look like
# [input-1-value, input-3-value]
# Instead of creating model with latest version, it now generates models for since_version by default.
# Thus it can make every model uses the same opset version after every opset change.
# Besides, user can specify "use_max_opset_version" to generate models for
# the latest opset vesion that supports before targeted opset version
def expect(node: onnx.NodeProto,
           inputs: Sequence[np.ndarray],
           outputs: Sequence[np.ndarray],
           name: Text,
           **kwargs: Any
           ) -> None:
    # skip if the node's op_type is not same as the given one
    if _TargetOpType and node.op_type != _TargetOpType:
        return
    present_inputs = [x for x in node.input if (x != "")]
    present_outputs = [x for x in node.output if (x != "")]
    input_type_protos = [None] * len(inputs)
    if "input_type_protos" in kwargs:
        input_type_protos = kwargs[str("input_type_protos")]
        del kwargs[str("input_type_protos")]
    output_type_protos = [None] * len(outputs)
    if "output_type_protos" in kwargs:
        output_type_protos = kwargs[str("output_type_protos")]
        del kwargs[str("output_type_protos")]
    inputs_vi = [_extract_value_info(arr, arr_name, input_type)
                 for arr, arr_name, input_type in zip(inputs, present_inputs, input_type_protos)]
    outputs_vi = [_extract_value_info(arr, arr_name, output_type)
                  for arr, arr_name, output_type in zip(outputs, present_outputs, output_type_protos)]
    graph = onnx.helper.make_graph(
        nodes=[node],
        name=name,
        inputs=inputs_vi,
        outputs=outputs_vi)
    kwargs[str("producer_name")] = "backend-test"

    if "opset_imports" not in kwargs:
        # To make sure the model will be produced with the same opset_version after opset changes
        # By default, it uses since_version as opset_version for produced models
        if _TargetOpsetVersion is None:
            produce_opset_version = onnx.defs.get_schema(node.op_type, node.domain).since_version
        # If given targeted opset version
        # it will generate test data for the latest opset vesion that supports before targeted opset version
        else:
            produce_opset_version = onnx.defs.get_schema(node.op_type, int(_TargetOpsetVersion), node.domain).since_version
        kwargs[str("opset_imports")] = [onnx.helper.make_operatorsetid(node.domain, produce_opset_version)]
    else:
        # Also, for the same reason above, convert given opset.version (from opset_imports) to max_inclusive_version
        max_inclusive_versions = []
        for opset in kwargs[str("opset_imports")]:
            produce_opset_version = onnx.defs.get_schema(node.op_type, opset.version, opset.domain).since_version
            max_inclusive_versions.append(onnx.helper.make_operatorsetid(opset.domain, produce_opset_version))
        kwargs[str("opset_imports")] = max_inclusive_versions

    model = _make_test_model_gen_version(graph, **kwargs)

    _NodeTestCases.append(TestCase(
        name=name,
        model_name=name,
        url=None,
        model_dir=None,
        model=model,
        data_sets=[(inputs, outputs)],
        kind="node",
        rtol=1e-3,
        atol=1e-7,
    ))

    # Create list of types for node.input, filling a default TypeProto for missing inputs:
    # E.g. merge(["x", "", "y"], [x-value-info, y-value-info]) will return [x-type, default-type, y-type]
    def merge(node_inputs: List[Text], present_value_info: List[onnx.ValueInfoProto]) -> List[TypeProto]:
        if (node_inputs):
            if (node_inputs[0] != ""):
                return [present_value_info[0].type] + merge(node_inputs[1:], present_value_info[1:])
            else:
                return [TypeProto()] + merge(node_inputs[1:], present_value_info)
        return []
    merged_types = merge(list(node.input), inputs_vi)
    expanded_function_nodes = function_testcase_helper(node, merged_types, name)
    if expanded_function_nodes:
        function_test_name = name + "_expanded"
        graph = onnx.helper.make_graph(
            nodes=expanded_function_nodes,
            name=function_test_name,
            inputs=inputs_vi,
            outputs=outputs_vi)
        kwargs[str("producer_name")] = "backend-test"
        model = _make_test_model_gen_version(graph, **kwargs)
        _NodeTestCases.append(TestCase(
            name=function_test_name,
            model_name=function_test_name,
            url=None,
            model_dir=None,
            model=model,
            data_sets=[(inputs, outputs)],
            kind="node",
            rtol=1e-3,
            atol=1e-7,
        ))


def collect_testcases(op_type: Text, opset_version: int) -> List[TestCase]:
    '''Collect node test cases
    '''
    # only keep those tests related to this operator
    global _TargetOpType
    _TargetOpType = op_type

    global _TargetOpsetVersion
    _TargetOpsetVersion = opset_version
    import_recursive(sys.modules[__name__])
    return _NodeTestCases
