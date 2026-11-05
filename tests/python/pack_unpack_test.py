# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from onnx import (
    FunctionProto,
    TensorProto,
    checker,
    defs,
    helper,
    parser,
    shape_inference,
)
from onnx.reference import ReferenceEvaluator


def evaluator(
    op: str, bits: int, function: bool = False, shape: tuple[int, ...] = (1,)
) -> ReferenceEvaluator:
    dimensions = ", ".join(map(str, shape))
    output_dimensions = ", ".join([*map(str, shape[:-1]), "N"])
    signature = f"uint8[{dimensions}] X"
    inputs = "X"
    if op == "Unpack":
        signature += ", int64 count"
        inputs += ", count"
    model = parser.parse_model(
        f"""
        <ir_version: 14, opset_import: ["" : 29]>
        g ({signature}) => (uint8[{output_dimensions}] Y) {{
            Y = {op}<bits = {bits}>({inputs})
        }}
        """
    )
    if function:
        body = FunctionProto()
        body.ParseFromString(
            defs.get_schema(op).get_context_dependent_function(
                model.graph.node[0].SerializeToString(),
                [value.type.SerializeToString() for value in model.graph.input],
            )
        )
        body.domain = "test.packing"
        model.functions.append(body)
        model.graph.node[0].domain = body.domain
        model.opset_import.append(helper.make_opsetid(body.domain, 1))
    checker.check_model(model, full_check=function)
    return ReferenceEvaluator(model)


def oracle(x: np.ndarray, bits: int) -> np.ndarray:
    n = x.shape[-1]
    packed = np.empty((*x.shape[:-1], (n * bits + 7) // 8), dtype=np.uint8)
    for index in np.ndindex(x.shape[:-1]):
        word = sum(int(code) << (i * bits) for i, code in enumerate(x[index]))
        packed[index] = np.frombuffer(
            word.to_bytes(packed.shape[-1], "little"), dtype=np.uint8
        )
    return packed


def grouped_evaluator(
    op: str,
    attrs: dict[str, int],
    shape: tuple[int, ...],
    function: bool = False,
    storage_type: int = TensorProto.UINT8,
) -> ReferenceEvaluator:
    inputs = [
        helper.make_tensor_value_info(
            "X", storage_type if op == "Unpack" else TensorProto.UINT8, shape
        )
    ]
    names = ["X"]
    if op == "Unpack":
        inputs.append(helper.make_tensor_value_info("count", TensorProto.INT64, []))
        names.append("count")
    node = helper.make_node(op, names, ["Y"], **attrs)
    output_shape = list(shape)
    if output_shape:
        output_shape[attrs.get("axis", -1)] = None
    graph = helper.make_graph(
        [node],
        "grouped",
        inputs,
        [
            helper.make_tensor_value_info(
                "Y",
                attrs.get("output_type", TensorProto.UINT8)
                if op == "Pack"
                else TensorProto.UINT8,
                output_shape,
            )
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 29)], ir_version=14
    )
    if function:
        body = FunctionProto()
        body.ParseFromString(
            defs.get_schema(op).get_context_dependent_function(
                node.SerializeToString(),
                [value.type.SerializeToString() for value in inputs],
            )
        )
        body.domain = "test.grouped"
        model.functions.append(body)
        node = model.graph.node[0]
        node.domain = body.domain
        model.opset_import.append(helper.make_opsetid(body.domain, 1))
    checker.check_model(model, full_check=function)
    return ReferenceEvaluator(model)


@pytest.mark.parametrize("storage_type", [TensorProto.UINT8, TensorProto.UINT32])
@pytest.mark.parametrize("axis", [-3, -2, -1, 0, 1, 2])
@pytest.mark.parametrize(
    "attrs",
    [
        {"bits": 5},
        {"bits": 8},
        {"radix": 3, "group_size": 5},
        {"radix": 3, "group_size": 39},
    ],
)
@pytest.mark.parametrize("shape", [(2, 11, 17), (0, 3, 9), (2, 0, 9), (2, 3, 0)])
def test_storage_words_and_axis(
    storage_type: int, axis: int, attrs: dict[str, int], shape: tuple[int, ...]
) -> None:
    radix = attrs.get("radix", 1 << attrs.get("bits", 1))
    group_size = attrs.get("group_size", 1)
    x = np.random.default_rng(2).integers(0, radix, shape, dtype=np.uint8)
    byte_rows = grouped_oracle(np.moveaxis(x, axis, -1), radix, group_size)
    word_bits = 32 if storage_type == TensorProto.UINT32 else 8
    words = (byte_rows.shape[-1] * 8 + word_bits - 1) // word_bits
    dtype = np.uint32 if storage_type == TensorProto.UINT32 else np.uint8
    expected_rows = np.zeros((*byte_rows.shape[:-1], words), dtype=dtype)
    for row in np.ndindex(byte_rows.shape[:-1]):
        stream = int.from_bytes(byte_rows[row].tobytes(), "little")
        for i in range(words):
            expected_rows[row][i] = (stream >> (word_bits * i)) & ((1 << word_bits) - 1)
    expected = np.moveaxis(expected_rows, -1, axis)
    pack_attrs = {**attrs, "axis": axis, "output_type": storage_type}
    unpack_attrs = {**attrs, "axis": axis}
    for function in (False, True):
        pack = grouped_evaluator("Pack", pack_attrs, x.shape, function)
        packed = pack.run(None, {"X": x})[0]
        assert packed.dtype == dtype
        np.testing.assert_array_equal(packed, expected)
        unpack = grouped_evaluator(
            "Unpack", unpack_attrs, packed.shape, function, storage_type
        )
        decoded = unpack.run(
            None, {"X": packed, "count": np.array(x.shape[axis], dtype=np.int64)}
        )[0]
        np.testing.assert_array_equal(decoded, x)
        unpack.proto_.graph.initializer.append(
            helper.make_tensor("count", TensorProto.INT64, [], [x.shape[axis]])
        )
        for model, output in ((pack.proto_, packed), (unpack.proto_, decoded)):
            inferred = shape_inference.infer_shapes(model, strict_mode=True)
            tensor = inferred.graph.output[0].type.tensor_type
            if not function:
                assert tuple(d.dim_value for d in tensor.shape.dim) == output.shape
            else:
                assert len(tensor.shape.dim) == output.ndim
                assert all(
                    not d.HasField("dim_value") or d.dim_value == size
                    for d, size in zip(tensor.shape.dim, output.shape, strict=True)
                )
            assert tensor.elem_type == helper.np_dtype_to_tensor_dtype(output.dtype)


@pytest.mark.parametrize("function", [False, True])
def test_uint32_bit31_and_padding(function: bool) -> None:
    packed = np.array([0xFFFFFFFF, 0xFFFFFFFF], dtype=np.uint32)
    decoded = grouped_evaluator(
        "Unpack", {"bits": 5}, packed.shape, function, TensorProto.UINT32
    ).run(None, {"X": packed, "count": np.array(7, dtype=np.int64)})[0]
    np.testing.assert_array_equal(decoded, np.full(7, 31, dtype=np.uint8))
    encoded = grouped_evaluator(
        "Pack", {"bits": 5, "output_type": TensorProto.UINT32}, decoded.shape, function
    ).run(None, {"X": decoded})[0]
    np.testing.assert_array_equal(encoded, [0xFFFFFFFF, 7])


@pytest.mark.parametrize("op", ["Pack", "Unpack"])
@pytest.mark.parametrize("axis", [-3, 2])
def test_invalid_axis(op: str, axis: int) -> None:
    model = grouped_evaluator(op, {"bits": 3}, (2, 1)).proto_
    model.graph.node[0].attribute.append(helper.make_attribute("axis", axis))
    feeds = {"X": np.zeros((2, 1), dtype=np.uint8)}
    if op == "Unpack":
        feeds["count"] = np.array(2, dtype=np.int64)
    with pytest.raises(ValueError, match="axis"):
        ReferenceEvaluator(model).run(None, feeds)
    with pytest.raises(shape_inference.InferenceError, match="axis"):
        shape_inference.infer_shapes(model, strict_mode=True)


@pytest.mark.parametrize(
    "storage_type", [TensorProto.INT32, TensorProto.UINT16, TensorProto.FLOAT]
)
def test_invalid_output_storage_type(storage_type: int) -> None:
    model = grouped_evaluator("Pack", {"bits": 3}, (1,)).proto_
    model.graph.node[0].attribute.append(
        helper.make_attribute("output_type", storage_type)
    )
    with pytest.raises(ValueError, match="output_type"):
        ReferenceEvaluator(model).run(None, {"X": np.zeros(1, dtype=np.uint8)})
    with pytest.raises(shape_inference.InferenceError, match="storage type"):
        shape_inference.infer_shapes(model, strict_mode=True)


def test_uint32_symbolic_axis_shape() -> None:
    model = grouped_evaluator(
        "Pack", {"bits": 5, "axis": 0, "output_type": TensorProto.UINT32}, (2, 3)
    ).proto_
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "N"
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    shape = inferred.graph.output[0].type.tensor_type.shape
    assert not shape.dim[0].HasField("dim_value")
    assert shape.dim[1] == model.graph.input[0].type.tensor_type.shape.dim[1]


def grouped_oracle(x: np.ndarray, radix: int, group_size: int) -> np.ndarray:
    groups = (x.shape[-1] + group_size - 1) // group_size
    width = (radix**group_size - 1).bit_length()
    output = np.zeros((*x.shape[:-1], (groups * width + 7) // 8), dtype=np.uint8)
    for row in np.ndindex(x.shape[:-1]):
        bits = []
        for start in range(0, x.shape[-1], group_size):
            group = x[row][start : start + group_size]
            value = sum(int(digit) * radix**j for j, digit in enumerate(group))
            bits.extend((value >> j) & 1 for j in range(width))
        for i, bit in enumerate(bits):
            output[row][i // 8] |= bit << (i % 8)
    return output


@pytest.mark.parametrize(
    "radix,group_size",
    [(3, 5), (3, 3), (5, 3), (7, 2), (16, 2), (256, 1), (2, 62), (3, 39)],
)
@pytest.mark.parametrize("n", [0, 1, 2, 5, 9, 17, 65])
@pytest.mark.parametrize("leading", [(), (2,), (2, 3), (0,), (2, 0)])
def test_grouped_pack_unpack(
    radix: int, group_size: int, n: int, leading: tuple[int, ...]
) -> None:
    rng = np.random.default_rng(1)
    x = rng.integers(0, radix, (*leading, n), dtype=np.uint8)
    expected = grouped_oracle(x, radix, group_size)
    attrs = {"radix": radix, "group_size": group_size}
    for function in (False, True):
        packed = grouped_evaluator("Pack", attrs, x.shape, function).run(
            None, {"X": x}
        )[0]
        np.testing.assert_array_equal(packed, expected)
        decoded = grouped_evaluator("Unpack", attrs, packed.shape, function).run(
            None, {"X": packed, "count": np.array(n, dtype=np.int64)}
        )[0]
        np.testing.assert_array_equal(decoded, x)


@pytest.mark.parametrize("bits", range(1, 9))
def test_grouped_bits_equivalence(bits: int) -> None:
    x = (np.arange(37) % (1 << bits)).astype(np.uint8)
    expected = oracle(x, bits)
    for function in (False, True):
        actual = grouped_evaluator(
            "Pack", {"radix": 1 << bits, "group_size": 1}, x.shape, function
        ).run(None, {"X": x})[0]
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("attrs", [{"bits": 8}, {"radix": 256, "group_size": 1}])
def test_pack_full_uint8_range_strided(attrs: dict[str, int]) -> None:
    # NumPy 2.0 cannot safely compare a strided UINT8 array against 256.
    x = np.arange(256, dtype=np.uint8)[::-2]
    packed = grouped_evaluator("Pack", attrs, x.shape).run(None, {"X": x})[0]
    np.testing.assert_array_equal(packed, x)


@pytest.mark.parametrize(
    "attrs",
    [
        {},
        {"radix": 3},
        {"group_size": 5},
        {"bits": 3, "radix": 3, "group_size": 5},
        {"bits": 3, "radix": 3},
        {"bits": 3, "group_size": 5},
        {"radix": 1, "group_size": 5},
        {"radix": 257, "group_size": 1},
        {"radix": 3, "group_size": 0},
        {"radix": 3, "group_size": -1},
        {"radix": 256, "group_size": 8},
        {"radix": 2, "group_size": 63},
        {"radix": 3, "group_size": 40},
    ],
)
@pytest.mark.parametrize("op", ["Pack", "Unpack"])
def test_grouped_invalid_attributes(op: str, attrs: dict[str, int]) -> None:
    feeds = {"X": np.array([0], dtype=np.uint8)}
    if op == "Unpack":
        feeds["count"] = np.array(1, dtype=np.int64)
    with pytest.raises(ValueError):
        grouped_evaluator(op, attrs, (1,)).run(None, feeds)
    node = helper.make_node(op, list(feeds), ["Y"], **attrs)
    model = helper.make_model(
        helper.make_graph(
            [node],
            "invalid",
            [
                helper.make_tensor_value_info(
                    k, TensorProto.UINT8 if k == "X" else TensorProto.INT64, v.shape
                )
                for k, v in feeds.items()
            ],
            [helper.make_tensor_value_info("Y", TensorProto.UINT8, [None])],
        ),
        opset_imports=[helper.make_opsetid("", 29)],
    )
    with pytest.raises(shape_inference.InferenceError):
        shape_inference.infer_shapes(model, strict_mode=True)


def test_grouped_invalid_indices_and_codes() -> None:
    attrs = {"radix": 3, "group_size": 5}
    with pytest.raises(ValueError, match="representable"):
        grouped_evaluator("Pack", attrs, (1,)).run(
            None, {"X": np.array([3], dtype=np.uint8)}
        )
    with pytest.raises(ValueError, match="group index"):
        grouped_evaluator("Unpack", attrs, (1,)).run(
            None,
            {
                "X": np.array([243], dtype=np.uint8),
                "count": np.array(5, dtype=np.int64),
            },
        )


@pytest.mark.parametrize("function", [False, True])
def test_grouped_ignored_digits_and_byte_padding(function: bool) -> None:
    for attrs, payload, count, expected in [
        ({"radix": 3, "group_size": 5}, [242], 2, [2, 2]),
        ({"radix": 3, "group_size": 2}, [0xF8], 2, [2, 2]),
    ]:
        decoded = grouped_evaluator("Unpack", attrs, (1,), function).run(
            None,
            {
                "X": np.array(payload, dtype=np.uint8),
                "count": np.array(count, dtype=np.int64),
            },
        )[0]
        np.testing.assert_array_equal(decoded, expected)


@pytest.mark.parametrize("n,expected", [(0, 0), (1, 1), (5, 1), (6, 2), (11, 3)])
def test_grouped_shape_inference(n: int, expected: int) -> None:
    graph = helper.make_graph(
        [helper.make_node("Pack", ["X"], ["Y"], radix=3, group_size=5)],
        "shape",
        [helper.make_tensor_value_info("X", TensorProto.UINT8, ["rows", n])],
        [helper.make_tensor_value_info("Y", TensorProto.UINT8, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 29)])
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    shape = inferred.graph.output[0].type.tensor_type.shape
    assert shape.dim[0].dim_param == "rows"
    assert shape.dim[1].dim_value == expected


def test_grouped_unpack_shape_and_invalid_length() -> None:
    expected_count = 6
    for length in (1, 2):
        count = helper.make_tensor("count", TensorProto.INT64, [], [expected_count])
        graph = helper.make_graph(
            [helper.make_node("Unpack", ["X", "count"], ["Y"], radix=3, group_size=5)],
            "shape",
            [helper.make_tensor_value_info("X", TensorProto.UINT8, [2, length])],
            [helper.make_tensor_value_info("Y", TensorProto.UINT8, None)],
            [count],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 29)])
        if length == 1:
            with pytest.raises(shape_inference.InferenceError, match="word count"):
                shape_inference.infer_shapes(model, strict_mode=True)
        else:
            inferred = shape_inference.infer_shapes(model, strict_mode=True)
            assert (
                inferred.graph.output[0].type.tensor_type.shape.dim[1].dim_value
                == expected_count
            )


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("n", [0, 1, 2, 7, 8, 9, 17])
@pytest.mark.parametrize("leading", [(), (2,), (2, 3), (0,), (2, 0)])
def test_pack_unpack(bits: int, n: int, leading: tuple[int, ...]) -> None:
    rng = np.random.default_rng(0)
    x = rng.integers(0, 1 << bits, size=(*leading, n), dtype=np.uint8)
    expected = oracle(x, bits)
    for function in (False, True):
        packed = evaluator("Pack", bits, function, x.shape).run(None, {"X": x})[0]
        np.testing.assert_array_equal(packed, expected)
        unpacked = evaluator("Unpack", bits, function, packed.shape).run(
            None, {"X": packed, "count": np.array(n, dtype=np.int64)}
        )[0]
        np.testing.assert_array_equal(unpacked, x)


@pytest.mark.parametrize("function", [False, True])
def test_unpack_ignores_padding(function: bool) -> None:
    x = np.array([[0xFF]], dtype=np.uint8)
    y = evaluator("Unpack", 3, function, x.shape).run(
        None, {"X": x, "count": np.array(2, dtype=np.int64)}
    )[0]
    np.testing.assert_array_equal(y, [[7, 7]])


@pytest.mark.parametrize("bits", [0, 9, -1])
@pytest.mark.parametrize("op", ["Pack", "Unpack"])
def test_invalid_bits(op: str, bits: int) -> None:
    feeds = {"X": np.array([0], dtype=np.uint8)}
    if op == "Unpack":
        feeds["count"] = np.array(1, dtype=np.int64)
    with pytest.raises(ValueError, match="bits must be"):
        evaluator(op, bits).run(None, feeds)


@pytest.mark.parametrize("op", ["Pack", "Unpack"])
def test_scalar_input(op: str) -> None:
    feeds = {"X": np.array(0, dtype=np.uint8)}
    if op == "Unpack":
        feeds["count"] = np.array(1, dtype=np.int64)
    with pytest.raises(ValueError, match="rank at least 1"):
        evaluator(op, 3).run(None, feeds)


def test_pack_code_range() -> None:
    with pytest.raises(ValueError, match="representable"):
        evaluator("Pack", 3).run(None, {"X": np.array([8], dtype=np.uint8)})


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("function", [False, True])
def test_noncontiguous_input(bits: int, function: bool) -> None:
    x = (np.arange(48) % (1 << bits)).astype(np.uint8).reshape(4, 12)[:, ::2]
    packed = evaluator("Pack", bits, function, x.shape).run(None, {"X": x})[0]
    np.testing.assert_array_equal(packed, oracle(x, bits))
    padded = np.zeros((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.uint8)
    padded[:, ::2] = packed
    sliced = padded[:, ::2]
    unpacked = evaluator("Unpack", bits, function, sliced.shape).run(
        None, {"X": sliced, "count": np.array(x.shape[-1], dtype=np.int64)}
    )[0]
    np.testing.assert_array_equal(unpacked, x)


@pytest.mark.parametrize(
    "count",
    [
        np.array(-1, dtype=np.int64),
        np.array([2], dtype=np.int64),
        np.array([[2]], dtype=np.int64),
        np.array(2, dtype=np.int32),
        np.array(2, dtype=np.uint64),
        np.array(2, dtype=np.float32),
    ],
)
def test_unpack_invalid_count(count: np.ndarray) -> None:
    with pytest.raises(ValueError, match="nonnegative scalar"):
        evaluator("Unpack", 3).run(
            None, {"X": np.array([0], dtype=np.uint8), "count": count}
        )


@pytest.mark.parametrize("count", [0, 3])
def test_unpack_invalid_length(count: int) -> None:
    with pytest.raises(ValueError, match="Packed axis dimension"):
        evaluator("Unpack", 3).run(
            None,
            {
                "X": np.array([0], dtype=np.uint8),
                "count": np.array(count, dtype=np.int64),
            },
        )


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("function", [False, True])
def test_all_codes(bits: int, function: bool) -> None:
    x = np.arange(1 << bits, dtype=np.uint8)[::-1]
    packed = evaluator("Pack", bits, function, x.shape).run(None, {"X": x})[0]
    np.testing.assert_array_equal(packed, oracle(x, bits))
    unpacked = evaluator("Unpack", bits, function, packed.shape).run(
        None, {"X": packed, "count": np.array(x.size, dtype=np.int64)}
    )[0]
    np.testing.assert_array_equal(unpacked, x)
    assert packed.dtype == unpacked.dtype == np.uint8


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("n", [1, 2, 7, 9, 17])
@pytest.mark.parametrize("function", [False, True])
def test_unpack_arbitrary_bytes(bits: int, n: int, function: bool) -> None:
    rng = np.random.default_rng(42)
    x = rng.integers(0, 256, size=(3, (n * bits + 7) // 8), dtype=np.uint8)
    expected = np.array(
        [
            [
                (int.from_bytes(row.tobytes(), "little") >> (i * bits))
                & ((1 << bits) - 1)
                for i in range(n)
            ]
            for row in x
        ],
        dtype=np.uint8,
    )
    unpacked = evaluator("Unpack", bits, function, x.shape).run(
        None, {"X": x, "count": np.array(n, dtype=np.int64)}
    )[0]
    np.testing.assert_array_equal(unpacked, expected)


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("n", [0, 9, 2**61, 2**61 + 1])
@pytest.mark.parametrize("function", [False, True])
def test_empty_rows_do_not_iterate(
    bits: int, n: int, function: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_range(*_args, **_kwargs):
        pytest.fail("Empty tensors must not construct or iterate over code indices.")

    # Guard the regression against both unbounded Python loops and huge allocations.
    monkeypatch.setattr(
        "onnx.reference.ops.op_pack.range", unexpected_range, raising=False
    )
    monkeypatch.setattr(
        "onnx.reference.ops.op_unpack.range", unexpected_range, raising=False
    )
    monkeypatch.setattr("onnx.reference.ops.op_range.Range._run", unexpected_range)
    x = np.empty((2, 0, n), dtype=np.uint8)
    packed = evaluator("Pack", bits, function, x.shape).run(None, {"X": x})[0]
    assert packed.shape == (2, 0, (n * bits + 7) // 8)
    unpacked = evaluator("Unpack", bits, function, packed.shape).run(
        None, {"X": packed, "count": np.array(n, dtype=np.int64)}
    )[0]
    assert unpacked.shape == x.shape
    assert packed.dtype == unpacked.dtype == np.uint8


@pytest.mark.parametrize("op", ["Pack", "Unpack"])
@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.float32, np.bool_])
def test_invalid_input_dtype(op: str, dtype: type) -> None:
    feeds = {"X": np.zeros((1,), dtype=dtype)}
    if op == "Unpack":
        feeds["count"] = np.array(2, dtype=np.int64)
    with pytest.raises(ValueError, match="UINT8"):
        evaluator(op, 3).run(None, feeds)


@pytest.mark.parametrize("op", ["Pack", "Unpack"])
def test_packing_attributes_are_required(op: str) -> None:
    inputs = [helper.make_tensor_value_info("X", TensorProto.UINT8, [1])]
    if op == "Unpack":
        inputs.append(helper.make_tensor_value_info("count", TensorProto.INT64, []))
    graph = helper.make_graph(
        [helper.make_node(op, [value.name for value in inputs], ["Y"])],
        "packing",
        inputs,
        [helper.make_tensor_value_info("Y", TensorProto.UINT8, [None])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 29)])
    with pytest.raises(shape_inference.InferenceError, match="bits"):
        shape_inference.infer_shapes(model, strict_mode=True)
