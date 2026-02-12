# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
"""Test ONNX shape inference against backend node test data.

For every node-level backend test, this test:
1. Loads ``model.onnx`` and records the expected output type info.
2. Strips output shape/type information.
3. Runs ``onnx.shape_inference.infer_shapes``.
4. Compares the inferred output shapes/types against the expected ones.

Expected outputs are loaded from ``output_*.pb`` as a fallback when the model
output type info is insufficient (e.g. sequence types).
"""

from __future__ import annotations

import pathlib
import unittest

import parameterized

import onnx
import onnx.backend.test
import onnx.shape_inference
from onnx import TensorProto, TypeProto

_NODE_TEST_DIR = pathlib.Path(onnx.backend.test.__file__).parent / "data" / "node"


def _discover_tests() -> list[str]:
    """Return sorted list of test directory names that have model.onnx."""
    return sorted(
        d.name for d in _NODE_TEST_DIR.iterdir() if (d / "model.onnx").exists()
    )


def _get_tensor_shape(type_proto: TypeProto) -> list[int | str] | None:
    """Extract shape from a TypeProto as a list of ints/dim_params, or None."""
    if not type_proto.HasField("tensor_type"):
        return None
    tt = type_proto.tensor_type
    if not tt.HasField("shape"):
        return None
    dims: list[int | str] = []
    for dim in tt.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(-1)  # unknown
    return dims


def _get_tensor_dtype(type_proto: TypeProto) -> int:
    """Extract element type from a TypeProto (0 if not a tensor)."""
    if not type_proto.HasField("tensor_type"):
        return 0
    return type_proto.tensor_type.elem_type


def _inject_inputs_as_initializers(
    model: onnx.ModelProto, test_dir: pathlib.Path
) -> None:
    """Load input_*.pb and add them as graph initializers.

    This makes constant inputs (e.g. axes for Unsqueeze, split sizes)
    visible to shape inference, which only reads initializers and attributes.
    """
    data_dir = test_dir / "test_data_set_0"
    if not data_dir.exists():
        return
    existing = {init.name for init in model.graph.initializer}
    input_names = [inp.name for inp in model.graph.input]
    for pb_file in sorted(data_dir.glob("input_*.pb")):
        tensor = TensorProto()
        tensor.ParseFromString(pb_file.read_bytes())
        if tensor.name and tensor.name not in existing and tensor.name in input_names:
            model.graph.initializer.append(tensor)


def _clear_output_type_info(model: onnx.ModelProto) -> None:
    """Strip shape and dtype from graph outputs so inference must rediscover them."""
    for output in model.graph.output:
        if output.type.HasField("tensor_type"):
            output.type.tensor_type.elem_type = 0
            output.type.tensor_type.ClearField("shape")
        elif output.type.HasField("sequence_type"):
            output.type.ClearField("sequence_type")
        elif output.type.HasField("optional_type"):
            output.type.ClearField("optional_type")
    # Also clear intermediate value_info
    del model.graph.value_info[:]


def _load_expected_output_shapes(
    model: onnx.ModelProto,
) -> dict[str, tuple[int, list[int | str] | None]]:
    """Collect expected (dtype, shape) for each output from the model proto.

    Returns:
        Mapping from output name to (elem_type, shape_list_or_None).
    """
    expected: dict[str, tuple[int, list[int | str] | None]] = {}
    for output in model.graph.output:
        dtype = _get_tensor_dtype(output.type)
        shape = _get_tensor_shape(output.type)
        expected[output.name] = (dtype, shape)
    return expected


# Tests with expanded (multi-node) function bodies where shape inference
# cannot propagate shapes through the expanded graph.
_SKIP_EXPANDED_MODELS: set[str] = {
    # AffineGrid expanded models
    "test_affine_grid_2d_align_corners_expanded",
    "test_affine_grid_2d_expanded",
    "test_affine_grid_3d_align_corners_expanded",
    "test_affine_grid_3d_expanded",
    # LayerNormalization expanded models
    "test_layer_normalization_2d_axis0_expanded",
    "test_layer_normalization_2d_axis0_expanded_ver18",
    "test_layer_normalization_2d_axis1_expanded",
    "test_layer_normalization_2d_axis1_expanded_ver18",
    "test_layer_normalization_2d_axis_negative_1_expanded",
    "test_layer_normalization_2d_axis_negative_1_expanded_ver18",
    "test_layer_normalization_2d_axis_negative_2_expanded",
    "test_layer_normalization_2d_axis_negative_2_expanded_ver18",
    "test_layer_normalization_3d_axis0_epsilon_expanded",
    "test_layer_normalization_3d_axis0_epsilon_expanded_ver18",
    "test_layer_normalization_3d_axis1_epsilon_expanded",
    "test_layer_normalization_3d_axis1_epsilon_expanded_ver18",
    "test_layer_normalization_3d_axis2_epsilon_expanded",
    "test_layer_normalization_3d_axis2_epsilon_expanded_ver18",
    "test_layer_normalization_3d_axis_negative_1_epsilon_expanded",
    "test_layer_normalization_3d_axis_negative_1_epsilon_expanded_ver18",
    "test_layer_normalization_3d_axis_negative_2_epsilon_expanded",
    "test_layer_normalization_3d_axis_negative_2_epsilon_expanded_ver18",
    "test_layer_normalization_3d_axis_negative_3_epsilon_expanded",
    "test_layer_normalization_3d_axis_negative_3_epsilon_expanded_ver18",
    "test_layer_normalization_4d_axis0_expanded",
    "test_layer_normalization_4d_axis0_expanded_ver18",
    "test_layer_normalization_4d_axis1_expanded",
    "test_layer_normalization_4d_axis1_expanded_ver18",
    "test_layer_normalization_4d_axis2_expanded",
    "test_layer_normalization_4d_axis2_expanded_ver18",
    "test_layer_normalization_4d_axis3_expanded",
    "test_layer_normalization_4d_axis3_expanded_ver18",
    "test_layer_normalization_4d_axis_negative_1_expanded",
    "test_layer_normalization_4d_axis_negative_1_expanded_ver18",
    "test_layer_normalization_4d_axis_negative_2_expanded",
    "test_layer_normalization_4d_axis_negative_2_expanded_ver18",
    "test_layer_normalization_4d_axis_negative_3_expanded",
    "test_layer_normalization_4d_axis_negative_3_expanded_ver18",
    "test_layer_normalization_4d_axis_negative_4_expanded",
    "test_layer_normalization_4d_axis_negative_4_expanded_ver18",
    "test_layer_normalization_default_axis_expanded",
    "test_layer_normalization_default_axis_expanded_ver18",
    # RMSNormalization expanded models
    "test_rms_normalization_2d_axis0_expanded",
    "test_rms_normalization_2d_axis1_expanded",
    "test_rms_normalization_2d_axis_negative_1_expanded",
    "test_rms_normalization_2d_axis_negative_2_expanded",
    "test_rms_normalization_3d_axis0_epsilon_expanded",
    "test_rms_normalization_3d_axis1_epsilon_expanded",
    "test_rms_normalization_3d_axis2_epsilon_expanded",
    "test_rms_normalization_3d_axis_negative_1_epsilon_expanded",
    "test_rms_normalization_3d_axis_negative_2_epsilon_expanded",
    "test_rms_normalization_3d_axis_negative_3_epsilon_expanded",
    "test_rms_normalization_4d_axis0_expanded",
    "test_rms_normalization_4d_axis1_expanded",
    "test_rms_normalization_4d_axis2_expanded",
    "test_rms_normalization_4d_axis3_expanded",
    "test_rms_normalization_4d_axis_negative_1_expanded",
    "test_rms_normalization_4d_axis_negative_2_expanded",
    "test_rms_normalization_4d_axis_negative_3_expanded",
    "test_rms_normalization_4d_axis_negative_4_expanded",
    "test_rms_normalization_default_axis_expanded",
    # SoftmaxCrossEntropyLoss expanded models
    "test_sce_NCd1_mean_weight_negative_ii_expanded",
    "test_sce_NCd1_mean_weight_negative_ii_log_prob_expanded",
    "test_sce_NCd1d2d3_none_no_weight_negative_ii_expanded",
    "test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded",
    "test_sce_NCd1d2d3_sum_weight_high_ii_expanded",
    "test_sce_NCd1d2d3_sum_weight_high_ii_log_prob_expanded",
    "test_sce_NCd1d2d3d4d5_mean_weight_expanded",
    "test_sce_NCd1d2d3d4d5_mean_weight_log_prob_expanded",
    "test_sce_NCd1d2d3d4d5_none_no_weight_expanded",
    "test_sce_NCd1d2d3d4d5_none_no_weight_log_prob_expanded",
    "test_sce_mean_3d_expanded",
    "test_sce_mean_3d_log_prob_expanded",
    "test_sce_mean_expanded",
    "test_sce_mean_log_prob_expanded",
    "test_sce_mean_no_weight_ii_3d_expanded",
    "test_sce_mean_no_weight_ii_3d_log_prob_expanded",
    "test_sce_mean_no_weight_ii_4d_expanded",
    "test_sce_mean_no_weight_ii_4d_log_prob_expanded",
    "test_sce_mean_no_weight_ii_expanded",
    "test_sce_mean_no_weight_ii_log_prob_expanded",
    "test_sce_mean_weight_expanded",
    "test_sce_mean_weight_ii_3d_expanded",
    "test_sce_mean_weight_ii_3d_log_prob_expanded",
    "test_sce_mean_weight_ii_4d_expanded",
    "test_sce_mean_weight_ii_4d_log_prob_expanded",
    "test_sce_mean_weight_ii_expanded",
    "test_sce_mean_weight_ii_log_prob_expanded",
    "test_sce_mean_weight_log_prob_expanded",
    "test_sce_none_expanded",
    "test_sce_none_log_prob_expanded",
    "test_sce_none_weights_expanded",
    "test_sce_none_weights_log_prob_expanded",
    "test_sce_sum_expanded",
    "test_sce_sum_log_prob_expanded",
    # Range expanded models
    "test_range_float_type_positive_delta_expanded",
    "test_range_int32_type_negative_delta_expanded",
}

# Tests requiring strict_mode=False or with known C++ inference limitations.
_SKIP_KNOWN_LIMITATIONS: set[str] = {
    # MeanVarianceNormalization fails with strict_mode=True
    # (see test_backend_test.py test_kwargs)
    "test_mvn",
    # Loop with subgraph — shape inference across subgraphs is limited
    "test_loop11",
    # MaxUnpool with output_shape input — C++ inference doesn't read initializer value
    "test_maxunpool_export_with_output_shape",
}

_SKIP_TESTS: set[str] = _SKIP_EXPANDED_MODELS | _SKIP_KNOWN_LIMITATIONS

_ALL_TESTS = _discover_tests()


class ShapeInferenceBackendNodeTest(unittest.TestCase):
    """Compare ONNX shape inference results against backend test expected outputs."""

    @parameterized.parameterized.expand(
        [(name,) for name in _ALL_TESTS],
    )
    def test_shape_inference(self, test_name: str) -> None:
        if test_name in _SKIP_TESTS:
            self.skipTest(f"In skip list: {test_name}")

        test_dir = _NODE_TEST_DIR / test_name
        model = onnx.load(str(test_dir / "model.onnx"))

        # Record expected output types before stripping
        expected = _load_expected_output_shapes(model)

        # Inject test inputs as initializers for constant folding
        _inject_inputs_as_initializers(model, test_dir)

        # Strip output type info
        _clear_output_type_info(model)

        # Run shape inference
        inferred_model = onnx.shape_inference.infer_shapes(
            model, check_type=True, strict_mode=True
        )

        # Build lookup from inferred model
        inferred_outputs = {o.name: o for o in inferred_model.graph.output}

        # Compare each output
        errors: list[str] = []
        for name, (exp_dtype, exp_shape) in expected.items():
            if name not in inferred_outputs:
                errors.append(f"Output '{name}': missing from inferred model")
                continue

            inf_output = inferred_outputs[name]
            inf_dtype = _get_tensor_dtype(inf_output.type)
            inf_shape = _get_tensor_shape(inf_output.type)

            # Skip non-tensor outputs (sequence, optional)
            if exp_dtype == 0 and exp_shape is None:
                continue

            if exp_dtype not in {0, inf_dtype}:
                errors.append(
                    f"Output '{name}': dtype mismatch: "
                    f"expected {TensorProto.DataType.Name(exp_dtype)}, "
                    f"got {TensorProto.DataType.Name(inf_dtype)}"
                )

            if exp_shape is not None and inf_shape is not None:
                if len(exp_shape) != len(inf_shape):
                    errors.append(
                        f"Output '{name}': rank mismatch: "
                        f"expected {exp_shape}, got {inf_shape}"
                    )
                else:
                    for i, (e, g) in enumerate(zip(exp_shape, inf_shape, strict=False)):
                        # Allow symbolic dims in inferred (dim_param)
                        if isinstance(g, str):
                            continue
                        if e != g:
                            errors.append(
                                f"Output '{name}': dim {i} mismatch: "
                                f"expected {e}, got {g}"
                            )
                            break
            elif exp_shape is not None and inf_shape is None:
                errors.append(
                    f"Output '{name}': expected shape {exp_shape} but got None"
                )

        if errors:
            self.fail("\n".join(errors))


if __name__ == "__main__":
    unittest.main()
