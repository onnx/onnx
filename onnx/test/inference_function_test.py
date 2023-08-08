# SPDX-License-Identifier: Apache-2.0

# Copyright (c) ONNX Project Contributors

import unittest
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from onnx import TensorProto, TypeProto
from onnx.checker import ValidationError
from onnx.defs import OpSchema, get_all_schemas_with_history, get_schema
from onnx.helper import (
    make_graph,
    make_node,
    make_opsetid,
    make_tensor_type_proto,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array
from onnx.shape_inference import InferenceError, infer_node_outputs

ADD_SCHEMA = max(
    (s for s in get_all_schemas_with_history() if s.name == "Add" and s.domain == ""),
    key=lambda s: s.since_version,
)
RESHAPE_SCHEMA = max(
    (
        s
        for s in get_all_schemas_with_history()
        if s.name == "Reshape" and s.domain == ""
    ),
    key=lambda s: s.since_version,
)


def _to_tensor_types(
    tensor_types: Dict[str, Tuple[int, Tuple[Union[int, str, None], ...]]]
) -> Dict[str, TypeProto]:
    return {key: make_tensor_type_proto(*value) for key, value in tensor_types.items()}


def _run_case(
    schema: OpSchema,
    input_names: List[str],
    output_names: List[str],
    input_types: Dict[str, TypeProto],
    input_data: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, TypeProto]:
    if input_data is None:
        input_data = {}
    return infer_node_outputs(
        schema,
        make_node(schema.name, input_names, output_names, domain=schema.domain),
        input_types,
        {key: from_array(arr) for key, arr in input_data.items()},
    )


class TestInferenceFunctionCall(unittest.TestCase):
    def test_add_inference(self) -> None:
        cases = [
            (
                {"A": (TensorProto.FLOAT, ()), "B": (TensorProto.FLOAT, ())},
                {"C": (TensorProto.FLOAT, ())},
            ),
            (
                {
                    "A": (TensorProto.FLOAT, (None, 2)),
                    "B": (TensorProto.FLOAT, (2,)),
                },
                {"C": (TensorProto.FLOAT, (None, 2))},
            ),
            (
                {
                    "A": (TensorProto.FLOAT, (None, 2)),
                    "B": (TensorProto.FLOAT, (1, 2)),
                },
                {"C": (TensorProto.FLOAT, (None, 2))},
            ),
            (
                {
                    "A": (TensorProto.DOUBLE, ("n", "m")),
                    "B": (TensorProto.DOUBLE, (1, "n", "m")),
                },
                {"C": (TensorProto.DOUBLE, (1, "n", "m"))},
            ),
            (
                {
                    "A": (TensorProto.FLOAT, ("x", 2)),
                    "B": (TensorProto.FLOAT, ("y", 2)),
                },
                {"C": (TensorProto.FLOAT, (None, 2))},
            ),
        ]
        for ins, outs in cases:
            assert _run_case(ADD_SCHEMA, ["A", "B"], ["C"], _to_tensor_types(ins)) == _to_tensor_types(outs)  # type: ignore

    def test_add_inference_raises_errors(self) -> None:
        with self.assertRaises(ValidationError):
            _run_case(
                ADD_SCHEMA,
                ["A"],
                ["C"],
                _to_tensor_types({"A": (TensorProto.FLOAT, (3, 4))}),
            )
        with self.assertRaises(ValidationError):
            _run_case(
                ADD_SCHEMA,
                ["A", "B"],
                ["C"],
                _to_tensor_types({"A": (TensorProto.FLOAT, (3, 4)), "B": (2, (3, 4))}),
            )
        with self.assertRaises(InferenceError):
            _run_case(
                ADD_SCHEMA,
                ["A", "B"],
                ["C"],
                _to_tensor_types(
                    {
                        "A": (TensorProto.FLOAT, (2, 4)),
                        "B": (TensorProto.FLOAT, (3, 4)),
                    }
                ),
            )
        with self.assertRaises(KeyError):
            _run_case(
                ADD_SCHEMA,
                ["A", "B"],
                ["C"],
                _to_tensor_types({"A": (TensorProto.FLOAT, (3, 4))}),
            )

    def test_reshape_inference(self) -> None:
        assert _run_case(
            RESHAPE_SCHEMA,
            ["x", "t"],
            ["y"],
            _to_tensor_types(
                {
                    "x": (TensorProto.FLOAT, (5, 4)),
                    "t": (TensorProto.INT64, (3,)),
                }
            ),
            {"t": np.array([2, 2, 5], dtype=np.int64)},
        ) == _to_tensor_types({"y": (TensorProto.FLOAT, (2, 2, 5))})

    def test_scan_inference_with_subgraph(self) -> None:
        seq_len = "sequence"
        input_size = 2
        loop_state_size = 3

        input_value_infos = [
            make_tensor_value_info("loop_state_in", TensorProto.UNDEFINED, None),
            make_tensor_value_info("input", TensorProto.UNDEFINED, None),
            make_tensor_value_info("outer", TensorProto.UNDEFINED, None),
        ]
        output_value_infos = [
            make_tensor_value_info("loop_state_out", TensorProto.UNDEFINED, None),
            make_tensor_value_info("output", TensorProto.FLOAT, (seq_len, input_size)),
        ]

        subgraph = make_graph(
            [
                make_node("Identity", ["loop_state_in"], ["loop_state_out"]),
                make_node("Add", ["input", "outer"], ["output"]),
            ],
            "subgraph",
            input_value_infos,
            output_value_infos,
        )

        assert infer_node_outputs(
            get_schema("Scan", 9),
            make_node(
                "Scan",
                ["loop_state_orig", "scan_input", "scan_outer"],
                ["loop_state_final", "scan_output"],
                num_scan_inputs=1,
                body=subgraph,
            ),
            _to_tensor_types(
                {
                    "loop_state_orig": (TensorProto.FLOAT, (loop_state_size,)),
                    "scan_input": (TensorProto.FLOAT, (seq_len, input_size)),
                    "scan_outer": (TensorProto.FLOAT, (input_size,)),
                }
            ),
            # Same as default value in Scan-9
            opset_imports=[make_opsetid("", 9)],
            ir_version=4,
        ) == _to_tensor_types(
            {
                "loop_state_final": (TensorProto.FLOAT, (loop_state_size,)),
                "scan_output": (TensorProto.FLOAT, (seq_len, input_size)),
            }
        )


if __name__ == "__main__":
    unittest.main()
