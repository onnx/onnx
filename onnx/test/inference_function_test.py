import unittest
from typing import List, Dict, Tuple, Union, Optional

import numpy as np

import onnx
import onnx.numpy_helper
import onnx.shape_inference

ADD_SCHEMA = max(
    (s for s in onnx.defs.get_all_schemas_with_history() if s.name == "Add" and s.domain == ""),
    key=lambda s: s.since_version
)
RESHAPE_SCHEMA = max(
    (s for s in onnx.defs.get_all_schemas_with_history() if s.name == "Reshape" and s.domain == ""),
    key=lambda s: s.since_version
)

_tensor = onnx.helper.make_tensor_type_proto


def _to_tensor_types(
        tensor_types: Dict[str, Tuple[int, Tuple[Union[int, str, None], ...]]]
) -> Dict[str, onnx.TypeProto]:
    return {key: onnx.helper.make_tensor_type_proto(*value) for key, value in tensor_types.items()}


def _run_case(
        schema: onnx.defs.OpSchema, input_names: List[str], output_names: List[str],
        input_types: Dict[str, onnx.TypeProto], input_data: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, onnx.TypeProto]:
    if input_data is None:
        input_data = {}
    return onnx.shape_inference.infer_node_outputs(
        schema,
        onnx.helper.make_node(schema.name, input_names, output_names, domain=schema.domain),
        input_types,
        {key: onnx.numpy_helper.from_array(arr) for key, arr in input_data.items()}
    )


class TestInferenceFunctionCall(unittest.TestCase):
    def test_add_inference(self) -> None:
        cases = [
            ({'A': (1, ()), 'B': (1, ())}, {'C': (1, ())}),
            ({'A': (1, (None, 2)), 'B': (1, (2,))}, {'C': (1, (None, 2))}),
            ({'A': (1, (None, 2)), 'B': (1, (1, 2))}, {'C': (1, (None, 2))}),
            ({'A': (2, ('n', 'm')), 'B': (2, (1, 'n', 'm'))}, {'C': (2, (1, 'n', 'm'))}),
            ({'A': (1, ('x', 2)), 'B': (1, ('y', 2))}, {'C': (1, (None, 2))})
        ]
        for ins, outs in cases:
            assert _run_case(ADD_SCHEMA, ['A', 'B'], ['C'], _to_tensor_types(ins)) == _to_tensor_types(outs)

    def test_add_inference_raises_errors(self) -> None:
        with self.assertRaises(onnx.checker.ValidationError):
            _run_case(ADD_SCHEMA, ['A'], ['C'], _to_tensor_types({'A': (1, (3, 4))}))
        with self.assertRaises(onnx.checker.ValidationError):
            _run_case(ADD_SCHEMA, ['A', 'B'], ['C'], _to_tensor_types({'A': (1, (3, 4)), 'B': (2, (3, 4))}))
        with self.assertRaises(onnx.shape_inference.InferenceError):
            _run_case(ADD_SCHEMA, ['A', 'B'], ['C'], _to_tensor_types({'A': (1, (2, 4)), 'B': (1, (3, 4))}))
        with self.assertRaises(KeyError):
            _run_case(ADD_SCHEMA, ['A', 'B'], ['C'], _to_tensor_types({'A': (1, (3, 4))}))

    def test_reshape_inference(self) -> None:
        assert _run_case(
            RESHAPE_SCHEMA,
            ['x', 't'], ['y'],
            _to_tensor_types({'x': (1, (5, 4)), 't': (onnx.TensorProto.INT64, (3,))}),
            {'t': np.array([2, 2, 5])}
        ) == _to_tensor_types({'y': (1, (2, 2, 5))})


if __name__ == '__main__':
    unittest.main()
