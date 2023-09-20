# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
import string
import unittest
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import numpy as np

import onnx
from onnx import TensorProto, ValueInfoProto, helper, shape_inference, version_converter

#####################################################################################
# Every test creates a model containing a single operator from the lowest possible
# opset version, upgrades it to the most recent opset version and then runs checker +
# shape inference on the upgraded model.
####################################################################################

LATEST_OPSET = onnx.defs.onnx_opset_version()
tested_ops = []


class TestAutomaticDowngrade(unittest.TestCase):
    def _test_op_downgrade(
        self,
        op: str,
        to_opset: int,
        input_shapes: Sequence[Union[Sequence[Optional[int]], str]] = ((3, 4, 5),),
        output_shapes: Sequence[Sequence[Optional[int]]] = ((3, 4, 5),),
        input_types: Optional[Sequence[Any]] = None,
        output_types: Optional[Sequence[Any]] = None,
        initializer: Sequence[Any] = (),
        attrs: Optional[Dict[str, Any]] = None,
        seq_inputs: Sequence[int] = (),
        seq_outputs: Sequence[int] = (),
        optional_inputs: Sequence[int] = (),
        optional_outputs: Sequence[int] = (),
    ) -> None:
        if attrs is None:
            attrs = {}

        tested_ops.append(op)

        n_inputs = len(input_shapes)
        letters = list(string.ascii_lowercase)[:n_inputs]
        input_names = [
            letter if shape != "" else ""
            for (letter, shape) in zip(letters, input_shapes)
        ]
        if input_types is None:
            input_types = [TensorProto.FLOAT] * n_inputs
        is_sequence = [0 if id not in seq_inputs else 1 for id in range(n_inputs)]
        is_optional = [0 if id not in optional_inputs else 1 for id in range(n_inputs)]
        # turn empty strings into [0] to ease type analysis, even though those entries
        # will be ignored
        input_shapes_cast = cast(
            List[List[int]],
            [[0] if isinstance(shape, str) else shape for shape in input_shapes],
        )
        inputs: List[ValueInfoProto] = []
        for name, ttype, shape, is_seq, is_opt in zip(
            input_names, input_types, input_shapes_cast, is_sequence, is_optional
        ):
            if name != "":
                if is_seq:
                    inputs += [
                        helper.make_tensor_sequence_value_info(name, ttype, shape)
                    ]
                elif is_opt:
                    type_proto = helper.make_tensor_type_proto(ttype, shape)
                    optional_type_proto = helper.make_optional_type_proto(type_proto)
                    inputs += [helper.make_value_info(name, optional_type_proto)]
                else:
                    inputs += [helper.make_tensor_value_info(name, ttype, shape)]

        n_outputs = len(output_shapes)
        output_names = list(string.ascii_lowercase)[n_inputs : n_inputs + n_outputs]
        if output_types is None:
            output_types = [TensorProto.FLOAT] * n_outputs
        is_sequence = [0 if id not in seq_outputs else 1 for id in range(n_outputs)]
        is_optional = [
            0 if id not in optional_outputs else 1 for id in range(n_outputs)
        ]
        output_shapes_cast = cast(
            List[List[int]],
            [[0] if isinstance(shape, str) else shape for shape in output_shapes],
        )
        outputs: List[ValueInfoProto] = []
        for name, ttype, shape, is_seq, is_opt in zip(
            output_names, output_types, output_shapes_cast, is_sequence, is_optional
        ):
            if is_seq:
                outputs += [helper.make_tensor_sequence_value_info(name, ttype, shape)]
            elif is_opt:
                type_proto = helper.make_tensor_type_proto(ttype, shape)
                optional_type_proto = helper.make_optional_type_proto(type_proto)
                outputs += [helper.make_value_info(name, optional_type_proto)]
            else:
                outputs += [helper.make_tensor_value_info(name, ttype, shape)]

        node = helper.make_node(op, input_names, output_names, **attrs)
        graph = helper.make_graph([node], op, inputs, outputs, initializer)
        original = helper.make_model(
            graph,
            producer_name="test",
            opset_imports=[helper.make_opsetid("", LATEST_OPSET)],
        )
        onnx.checker.check_model(original)
        shape_inference.infer_shapes(original, strict_mode=True)

        converted = version_converter.convert_version(original, to_opset)
        onnx.checker.check_model(converted)
        shape_inference.infer_shapes(converted, strict_mode=True)

    def test_ReduceOps(self) -> None:
        axes = helper.make_tensor(
            "b", TensorProto.INT64, dims=[3], vals=np.array([0, 1, 2])
        )
        reduce_ops = ["ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare"]
        for reduce_op in reduce_ops:
            self._test_op_downgrade(reduce_op, 13, [[3, 4, 5], [3]], [[1, 1, 1]], [TensorProto.FLOAT, TensorProto.INT64], initializer=[axes],)


if __name__ == "__main__":
    unittest.main()
