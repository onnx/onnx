# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
"""Test ONNX shape inference against in-memory backend node test cases.

For every node-level backend test, this test:
1. Copies the model and records the expected output type info.
2. Strips output shape/type information.
3. Runs ``onnx.shape_inference.infer_shapes``.
4. Compares the inferred output shapes/types against the expected ones.

Unknown shapes and dimensions are valid inference results. This checks soundness
of known results; operator-specific shape inference tests check completeness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import onnx
from onnx.backend.test.loader import load_node_model_tests

if TYPE_CHECKING:
    from onnx.backend.test.case.test_case import TestCase


def _get_tensor_shape(type_proto: onnx.TypeProto) -> list[int | str | None] | None:
    """Extract shape from a TypeProto as a list of ints/dim_params, or None."""
    if not type_proto.HasField("tensor_type"):
        return None
    tt = type_proto.tensor_type
    if not tt.HasField("shape"):
        return None
    dims: list[int | str | None] = []
    for dim in tt.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def _get_tensor_dtype(type_proto: onnx.TypeProto) -> int:
    """Extract element type from a TypeProto (0 if not a tensor)."""
    if not type_proto.HasField("tensor_type"):
        return 0
    return type_proto.tensor_type.elem_type


def _inject_inputs_as_initializers(model: onnx.ModelProto, test_case: TestCase) -> None:
    """Add tensor inputs from the first data set as graph initializers.

    This makes constant inputs (e.g. axes for Unsqueeze, split sizes)
    visible to shape inference, which only reads initializers and attributes.
    """
    assert test_case.data_sets
    existing = {init.name for init in model.graph.initializer}
    inputs, _ = test_case.data_sets[0]
    for inp, value in zip(model.graph.input, inputs, strict=True):
        if inp.name not in existing and inp.type.HasField("tensor_type"):
            if isinstance(value, onnx.TensorProto):
                tensor = onnx.TensorProto()
                tensor.CopyFrom(value)
                tensor.name = inp.name
            else:
                tensor = onnx.numpy_helper.from_array(np.asarray(value), inp.name)
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
) -> dict[str, tuple[int, list[int | str | None] | None]]:
    """Collect expected (dtype, shape) for each output from the model proto.

    Returns:
        Mapping from output name to (elem_type, shape_list_or_None).
    """
    expected: dict[str, tuple[int, list[int | str | None] | None]] = {}
    for output in model.graph.output:
        dtype = _get_tensor_dtype(output.type)
        shape = _get_tensor_shape(output.type)
        expected[output.name] = (dtype, shape)
    return expected


# Tests with expanded (multi-node) function bodies where shape inference
# cannot propagate shapes through the expanded graph.
_SKIP_EXPANDED_MODELS: set[str] = {
    # AffineGrid expanded — float arithmetic in grid construction can't propagate
    # through data propagation (only supports integer dim_values)
    "test_affine_grid_2d_align_corners_expanded",
    "test_affine_grid_2d_expanded",
    "test_affine_grid_3d_align_corners_expanded",
    "test_affine_grid_3d_expanded",
    # Range expanded — Loop trip count depends on float arithmetic
    # (data propagation only supports integer values)
    "test_range_float_type_positive_delta_expanded",
    "test_range_int32_type_negative_delta_expanded",
}

# Tests requiring strict_mode=False or with known C++ inference limitations.
_SKIP_KNOWN_LIMITATIONS: set[str] = set()

# Tests that need strict_mode=False to pass.
_STRICT_MODE_FALSE: set[str] = {
    # MeanVarianceNormalization fails with strict_mode=True
    "test_mvn",
}

_SKIP_TESTS: set[str] = _SKIP_EXPANDED_MODELS | _SKIP_KNOWN_LIMITATIONS

_ALL_TESTS = sorted(load_node_model_tests(), key=lambda case: case.name)


class TestShapeInferenceBackendNode:
    """Compare ONNX shape inference results against backend test expected outputs."""

    @pytest.mark.parametrize(
        "shape_kind", ["unknown_rank", "unknown_dim", "wrong_dim", "wrong_rank"]
    )
    def test_comparison_contract(
        self, monkeypatch: pytest.MonkeyPatch, shape_kind: str
    ) -> None:
        test_case = next(case for case in _ALL_TESTS if case.name == "test_abs")
        assert test_case.model is not None
        inferred = onnx.ModelProto()
        inferred.CopyFrom(test_case.model)
        tensor_type = inferred.graph.output[0].type.tensor_type
        if shape_kind == "unknown_rank":
            tensor_type.ClearField("shape")
        elif shape_kind == "unknown_dim":
            tensor_type.shape.dim[0].Clear()
        elif shape_kind == "wrong_dim":
            tensor_type.shape.dim[0].dim_value += 1
        else:
            tensor_type.shape.dim.add().dim_value = 1

        def infer(_model: onnx.ModelProto, **_kwargs: bool) -> onnx.ModelProto:
            return inferred

        monkeypatch.setattr(onnx.shape_inference, "infer_shapes", infer)
        if shape_kind.startswith("wrong"):
            with pytest.raises(pytest.fail.Exception, match="mismatch"):
                self._check_shape_inference(test_case)
        else:
            self._check_shape_inference(test_case)

    @pytest.mark.parametrize("test_case", _ALL_TESTS, ids=lambda case: case.name)
    def test_shape_inference(self, test_case: TestCase) -> None:
        test_name = test_case.name
        if test_name in _SKIP_TESTS:
            pytest.skip(f"In skip list: {test_name}")
        if test_name == "test_negative_pad_axes":
            # Also fails on main 04e9ef9f7: Pad parses int32 axes as int64.
            with pytest.raises(
                onnx.shape_inference.InferenceError,
                match="ParseData type mismatch for tensor: axes",
            ):
                self._check_shape_inference(test_case)
            pytest.xfail("Pad shape inference does not yet support int32 axes")

        self._check_shape_inference(test_case)

    def _check_shape_inference(self, test_case: TestCase) -> None:
        test_name = test_case.name

        assert test_case.model is not None
        model = onnx.ModelProto()
        model.CopyFrom(test_case.model)

        # Record expected output types before stripping
        expected = _load_expected_output_shapes(model)

        # Inject test inputs as initializers for constant folding
        _inject_inputs_as_initializers(model, test_case)

        # Strip output type info
        _clear_output_type_info(model)

        # Run shape inference
        strict = test_name not in _STRICT_MODE_FALSE
        inferred_model = onnx.shape_inference.infer_shapes(
            model, check_type=True, strict_mode=strict, data_prop=True
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
                    f"expected {onnx.TensorProto.DataType.Name(exp_dtype)}, "
                    f"got {onnx.TensorProto.DataType.Name(inf_dtype)}"
                )

            if exp_shape is not None and inf_shape is not None:
                if len(exp_shape) != len(inf_shape):
                    errors.append(
                        f"Output '{name}': rank mismatch: "
                        f"expected {exp_shape}, got {inf_shape}"
                    )
                else:
                    for i, (e, g) in enumerate(zip(exp_shape, inf_shape, strict=False)):
                        # Symbolic and unknown dimensions need not be resolved.
                        if isinstance(g, str) or g is None or e is None:
                            continue
                        if e != g:
                            errors.append(
                                f"Output '{name}': dim {i} mismatch: "
                                f"expected {e}, got {g}"
                            )
                            break

        if errors:
            pytest.fail("\n".join(errors))
