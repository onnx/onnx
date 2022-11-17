import unittest
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import onnx
import onnx.numpy_helper
import onnx.shape_inference
from onnx import TensorProto
from onnx.helper import make_graph, make_node, make_opsetid, make_tensor_value_info

ADD_SCHEMA = max(
    (
        s
        for s in onnx.defs.get_all_schemas_with_history()
        if s.name == "Add" and s.domain == ""
    ),
    key=lambda s: s.since_version,
)
RESHAPE_SCHEMA = max(
    (
        s
        for s in onnx.defs.get_all_schemas_with_history()
        if s.name == "Reshape" and s.domain == ""
    ),
    key=lambda s: s.since_version,
)


def _to_tensor_types(
    tensor_types: Dict[str, Tuple[int, Tuple[Union[int, str, None], ...]]]
) -> Dict[str, onnx.TypeProto]:
    return {
        key: onnx.helper.make_tensor_type_proto(*value)
        for key, value in tensor_types.items()
    }


def _run_case(
    schema: onnx.defs.OpSchema,
    input_names: List[str],
    output_names: List[str],
    input_types: Dict[str, onnx.TypeProto],
    input_data: Optional[Dict[str, np.ndarray]] = None,
    input_sparse_data: Optional[Dict[str, np.ndarray]] = None,
    subgraph: Optional[onnx.GraphProto] = None,
    subgraph_opset_imports: Optional[List[onnx.OperatorSetIdProto]] = None,
    subgraph_ir_version: int = onnx.IR_VERSION,
) -> Dict[str, onnx.TypeProto]:
    if input_data is None:
        input_data = {}
    return onnx.shape_inference.infer_node_outputs(
        schema,
        onnx.helper.make_node(
            schema.name, input_names, output_names, domain=schema.domain, body=subgraph
        ),
        input_types,
        {key: onnx.numpy_helper.from_array(arr) for key, arr in input_data.items()},
        input_sparse_data,
        subgraph_opset_imports,
        subgraph_ir_version,
    )


class TestInferenceFunctionCall(unittest.TestCase):
    def test_add_inference(self) -> None:
        cases = [
            (
                {"A": (onnx.TensorProto.FLOAT, ()), "B": (onnx.TensorProto.FLOAT, ())},
                {"C": (onnx.TensorProto.FLOAT, ())},
            ),
            (
                {
                    "A": (onnx.TensorProto.FLOAT, (None, 2)),
                    "B": (onnx.TensorProto.FLOAT, (2,)),
                },
                {"C": (onnx.TensorProto.FLOAT, (None, 2))},
            ),
            (
                {
                    "A": (onnx.TensorProto.FLOAT, (None, 2)),
                    "B": (onnx.TensorProto.FLOAT, (1, 2)),
                },
                {"C": (onnx.TensorProto.FLOAT, (None, 2))},
            ),
            (
                {
                    "A": (onnx.TensorProto.DOUBLE, ("n", "m")),
                    "B": (onnx.TensorProto.DOUBLE, (1, "n", "m")),
                },
                {"C": (onnx.TensorProto.DOUBLE, (1, "n", "m"))},
            ),
            (
                {
                    "A": (onnx.TensorProto.FLOAT, ("x", 2)),
                    "B": (onnx.TensorProto.FLOAT, ("y", 2)),
                },
                {"C": (onnx.TensorProto.FLOAT, (None, 2))},
            ),
        ]
        for ins, outs in cases:
            assert _run_case(ADD_SCHEMA, ["A", "B"], ["C"], _to_tensor_types(ins)) == _to_tensor_types(outs)  # type: ignore

    def test_add_inference_raises_errors(self) -> None:
        with self.assertRaises(onnx.checker.ValidationError):
            _run_case(
                ADD_SCHEMA,
                ["A"],
                ["C"],
                _to_tensor_types({"A": (onnx.TensorProto.FLOAT, (3, 4))}),
            )
        with self.assertRaises(onnx.checker.ValidationError):
            _run_case(
                ADD_SCHEMA,
                ["A", "B"],
                ["C"],
                _to_tensor_types(
                    {"A": (onnx.TensorProto.FLOAT, (3, 4)), "B": (2, (3, 4))}
                ),
            )
        with self.assertRaises(onnx.shape_inference.InferenceError):
            _run_case(
                ADD_SCHEMA,
                ["A", "B"],
                ["C"],
                _to_tensor_types(
                    {
                        "A": (onnx.TensorProto.FLOAT, (2, 4)),
                        "B": (onnx.TensorProto.FLOAT, (3, 4)),
                    }
                ),
            )
        with self.assertRaises(KeyError):
            _run_case(
                ADD_SCHEMA,
                ["A", "B"],
                ["C"],
                _to_tensor_types({"A": (onnx.TensorProto.FLOAT, (3, 4))}),
            )

    def test_reshape_inference(self) -> None:
        assert _run_case(
            RESHAPE_SCHEMA,
            ["x", "t"],
            ["y"],
            _to_tensor_types(
                {
                    "x": (onnx.TensorProto.FLOAT, (5, 4)),
                    "t": (onnx.TensorProto.INT64, (3,)),
                }
            ),
            {"t": np.array([2, 2, 5], dtype=np.int64)},
        ) == _to_tensor_types({"y": (onnx.TensorProto.FLOAT, (2, 2, 5))})

    def test_scan_inference_with_subgraph(self) -> None:
        seq_len = "sequence"
        input_size = 2
        loop_state_size = 3

        input_value_infos = [
            make_tensor_value_info("loop_state_in", TensorProto.UNDEFINED, None),
            make_tensor_value_info("input", TensorProto.UNDEFINED, None),
        ]
        output_value_infos = [
            make_tensor_value_info("loop_state_out", TensorProto.UNDEFINED, None),
            make_tensor_value_info("output", TensorProto.UNDEFINED, None),
        ]

        subgraph = make_graph(
            [
                make_node("Identity", ["loop_state_in"], ["loop_state_out"]),
                make_node("Identity", ["input"], ["output"]),
            ],
            "subgraph",
            input_value_infos,
            output_value_infos,
        )

        assert onnx.shape_inference.infer_node_outputs(
            onnx.defs.get_schema("Scan", 9),
            make_node(
                "Scan",
                ["loop_state_orig", "scan_input"],
                ["loop_state_final", "scan_output"],
                num_scan_inputs=1,
                body=subgraph,
            ),
            _to_tensor_types(
                {
                    "loop_state_orig": (TensorProto.FLOAT, (loop_state_size,)),
                    "scan_input": (TensorProto.FLOAT, (seq_len, input_size)),
                }
            ),
        ) == _to_tensor_types(
            {
                "loop_state_final": (onnx.TensorProto.FLOAT, (loop_state_size,)),
                "scan_output": (onnx.TensorProto.FLOAT, (seq_len, input_size)),
            }
        )


if __name__ == "__main__":
    unittest.main()
