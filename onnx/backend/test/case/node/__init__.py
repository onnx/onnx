from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

from typing import List, Text, Sequence
import numpy as np  # type: ignore

import onnx
import onnx.mapping

from ..utils import import_recursive
from ..test_case import TestCase

_NodeTestCases = []


def _extract_value_info(arr, name):  # type: (np.ndarray, Text) -> onnx.ValueInfoProto
    return onnx.helper.make_tensor_value_info(
        name=name,
        elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype],
        shape=arr.shape)


def expect(node,  # type: onnx.NodeProto
           inputs,  # type: Sequence[np.ndarray]
           outputs,  # type: Sequence[np.ndarray]
           name,  # type: Text
           ):  # type: (...) -> None
    present_inputs = [x for x in node.input if (x != '')]
    present_outputs = [x for x in node.output if (x != '')]
    inputs_vi = [_extract_value_info(arr, arr_name)
                 for arr, arr_name in zip(inputs, present_inputs)]
    outputs_vi = [_extract_value_info(arr, arr_name)
                  for arr, arr_name in zip(outputs, present_outputs)]
    graph = onnx.helper.make_graph(
        nodes=[node],
        name=name,
        inputs=inputs_vi,
        outputs=outputs_vi)
    model = onnx.helper.make_model(graph, producer_name='backend-test')

    _NodeTestCases.append(TestCase(
        name=name,
        model_name=name,
        url=None,
        model_dir=None,
        model=model,
        data_sets=[(inputs, outputs)],
        kind='node',
    ))


def collect_testcases():  # type: () -> List[TestCase]
    '''Collect node test cases defined in python/numpy code.
    '''
    import_recursive(sys.modules[__name__])
    return _NodeTestCases
