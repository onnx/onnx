# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

import onnx
from onnx import TensorProto, checker, defs, helper, numpy_helper, shape_inference
from onnx.reference import ReferenceEvaluator
from onnx.version_converter import convert_version


def _model(
    signal_shape: list[int | str | None] | None,
    *,
    opset: int = 29,
    onesided: int = 1,
    frame_length: int | None = 4,
    window: np.ndarray | None = None,
    elem_type: int = TensorProto.FLOAT,
) -> onnx.ModelProto:
    names = ["signal", "frame_step"]
    initializers = [
        numpy_helper.from_array(np.array(2, dtype=np.int64), name="frame_step")
    ]
    if window is not None or frame_length is not None:
        names.append("window" if window is not None else "")
    if window is not None:
        initializers.append(numpy_helper.from_array(window, name="window"))
    if frame_length is not None:
        names.append("frame_length")
        initializers.append(
            numpy_helper.from_array(
                np.array(frame_length, dtype=np.int64), name="frame_length"
            )
        )
    graph = helper.make_graph(
        [helper.make_node("STFT", names, ["output"], onesided=onesided)],
        "stft_rank2",
        [helper.make_tensor_value_info("signal", elem_type, signal_shape)],
        [helper.make_tensor_value_info("output", elem_type, [None, None, None, 2])],
        initializers,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10
    )


def _output_shape(model: onnx.ModelProto) -> list[int | str | None]:
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    return [
        dim.dim_value
        if dim.HasField("dim_value")
        else dim.dim_param
        if dim.HasField("dim_param")
        else None
        for dim in inferred.graph.output[0].type.tensor_type.shape.dim
    ]


def _expected(
    signal: np.ndarray, length: int, window: np.ndarray, onesided: int
) -> np.ndarray:
    frames = np.stack(
        [
            signal[:, start : start + length] * window
            for start in range(0, signal.shape[1] - length + 1, 2)
        ],
        axis=1,
    )
    transformed = np.fft.fft(frames, axis=2)
    if onesided:
        transformed = transformed[:, :, : length // 2 + 1]
    return np.stack((transformed.real, transformed.imag), axis=-1)


@pytest.mark.parametrize("opset", [17, 28])
def test_legacy_schema_still_rejects_rank2(opset: int) -> None:
    assert defs.get_schema("STFT", opset).since_version == 17
    with pytest.raises(shape_inference.InferenceError, match="must have rank 3"):
        shape_inference.infer_shapes(_model([2, 16], opset=opset), strict_mode=True)


def test_rank2_schema_is_versioned() -> None:
    schema = defs.get_schema("STFT", 29)
    assert schema.since_version == 29
    assert "[batch_size][signal_length] or" in schema.inputs[0].description


@pytest.mark.parametrize("onesided", [0, 1])
@pytest.mark.parametrize("mode", ["length", "window", "default"])
@pytest.mark.parametrize("elem_type", [TensorProto.FLOAT, TensorProto.DOUBLE])
def test_rank2_matches_numpy_and_rank3(
    onesided: int, mode: str, elem_type: int
) -> None:
    dtype = np.float32 if elem_type == TensorProto.FLOAT else np.float64
    signal = np.arange(32, dtype=dtype).reshape(2, 16)
    length = 16 if mode == "default" else 4
    window = np.hanning(length).astype(dtype) if mode == "window" else None
    frame_length = length if mode == "length" else None
    model = _model(
        [2, 16],
        onesided=onesided,
        frame_length=frame_length,
        window=window,
        elem_type=elem_type,
    )
    checker.check_model(model, full_check=True)
    expected = _expected(
        signal,
        length,
        np.ones(length, dtype=dtype) if window is None else window,
        onesided,
    )
    assert _output_shape(model) == list(expected.shape)
    output = ReferenceEvaluator(model).run(None, {"signal": signal})[0]
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)
    assert output.dtype == signal.dtype
    rank3 = _model(
        [2, 16, 1],
        onesided=onesided,
        frame_length=frame_length,
        window=window,
        elem_type=elem_type,
    )
    control = ReferenceEvaluator(rank3).run(None, {"signal": signal[..., np.newaxis]})[
        0
    ]
    np.testing.assert_array_equal(output, control)


@pytest.mark.parametrize("length", [1, 2])
def test_rank2_last_dimension_is_signal_length(length: int) -> None:
    signal = np.arange(2 * length, dtype=np.float32).reshape(2, length)
    model = _model([2, length], frame_length=None)
    checker.check_model(model, full_check=True)
    output = ReferenceEvaluator(model).run(None, {"signal": signal})[0]
    assert output.shape == (2, 1, length // 2 + 1, 2)


@pytest.mark.parametrize("onesided", [0, 1])
def test_symbolic_rank2_inference(onesided: int) -> None:
    model = _model(["batch", "length"], onesided=onesided)
    shape = _output_shape(model)
    assert shape[0] == "batch"
    assert shape[1] is None or isinstance(shape[1], str)
    assert shape[2:] == [3 if onesided else 4, 2]


@pytest.mark.parametrize("shape", [[], [16], [2, 16, 3], [2, 4, 4, 1]])
def test_invalid_signal_shapes_are_rejected(shape: list[int]) -> None:
    with pytest.raises(shape_inference.InferenceError):
        shape_inference.infer_shapes(_model(shape), strict_mode=True)


def test_complex_onesided_signal_is_rejected() -> None:
    with pytest.raises(shape_inference.InferenceError, match="requires real input"):
        shape_inference.infer_shapes(_model([2, 16, 2]), strict_mode=True)


@pytest.mark.parametrize("target", [17, 28])
@pytest.mark.parametrize("onesided", [0, 1])
def test_downgrade_normalizes_rank2(target: int, onesided: int) -> None:
    signal = np.arange(32, dtype=np.float32).reshape(2, 16)
    model = _model([2, 16], onesided=onesided)
    converted = convert_version(model, target)
    assert converted.opset_import[0].version == target
    assert [node.op_type for node in converted.graph.node] == [
        "Constant",
        "Unsqueeze",
        "STFT",
    ]
    checker.check_model(converted, full_check=True)
    expected = ReferenceEvaluator(model).run(None, {"signal": signal})[0]
    output = ReferenceEvaluator(converted).run(None, {"signal": signal})[0]
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("components", [1, 2])
def test_rank3_upgrade_and_downgrade_preserve_graph(components: int) -> None:
    signal = np.arange(32 * components, dtype=np.float32).reshape(2, 16, components)
    old = _model([2, 16, components], opset=28, onesided=0)
    upgraded = convert_version(old, 29)
    downgraded = convert_version(upgraded, 28)
    expected = ReferenceEvaluator(old).run(None, {"signal": signal})[0]
    for model in (upgraded, downgraded):
        assert [node.op_type for node in model.graph.node] == ["STFT"]
        checker.check_model(model, full_check=True)
        output = ReferenceEvaluator(model).run(None, {"signal": signal})[0]
        np.testing.assert_array_equal(output, expected)


def test_symbolic_rank2_downgrade() -> None:
    converted = convert_version(_model(["batch", "length"]), 28)
    shape = _output_shape(converted)
    assert shape[0] == "batch"
    assert shape[1] is None or isinstance(shape[1], str)
    assert shape[2:] == [3, 2]


def test_downgrade_unknown_rank_is_not_guessed() -> None:
    with pytest.raises(RuntimeError, match="known signal rank"):
        convert_version(_model(None), 28)
