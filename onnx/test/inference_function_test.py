from typing import Iterable, Optional, List

import pytest

import onnx
import onnx.shape_inference


def call_infer_types(
    schema: onnx.defs.OpSchema, inputs: Iterable[onnx.TypeProto], num_outputs: int, node: Optional[onnx.NodeProto] = None
) -> Optional[List[onnx.TypeProto]]:
    inputs = list(inputs)

    if not schema.has_type_and_shape_inference_function:  # type: ignore
        return None

    if node is None:
        input_names = [f"in{i}" for i in range(len(inputs))]
        output_names = [f"out{i}" for i in range(num_outputs)]
        node = onnx.helper.make_node(schema.name, input_names, output_names)

    outputs = schema.infer_types(  # type: ignore
        node.SerializeToString(),
        {f"in{i}": in_type.SerializeToString() for i, in_type in enumerate(inputs)},
        {},
        {},
    )

    return [onnx.TypeProto.FromString(output) for output in outputs]


@pytest.fixture
def add_schema():
    return max(
        (s for s in onnx.defs.get_all_schemas_with_history() if s.name == "Add"),
        key=lambda s: s.since_version
    )


def test_add_inference(add_schema):
    assert call_infer_types(add_schema, [
        onnx.helper.make_tensor_type_proto(1, ()),
        onnx.helper.make_tensor_type_proto(1, ())
    ], 1) == [onnx.helper.make_tensor_type_proto(1, ())]
    assert call_infer_types(add_schema, [
        onnx.helper.make_tensor_type_proto(1, (None, 2)),
        onnx.helper.make_tensor_type_proto(1, (2,))
    ], 1) == [onnx.helper.make_tensor_type_proto(1, (None, 2))]
    assert call_infer_types(add_schema, [
        onnx.helper.make_tensor_type_proto(1, (None, 2)),
        onnx.helper.make_tensor_type_proto(1, (1, 2))
    ], 1) == [onnx.helper.make_tensor_type_proto(1, (None, 2))]


def test_add_inference_raises_errors(add_schema):
    with pytest.raises(onnx.shape_inference.InferenceError):
        call_infer_types(add_schema, [
            onnx.helper.make_tensor_type_proto(1, (2,)),
            onnx.helper.make_tensor_type_proto(1, (3,))
        ], 1)
