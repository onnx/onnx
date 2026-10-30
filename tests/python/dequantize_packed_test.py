# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path

import numpy as np
import pytest

from onnx import (
    AttributeProto,
    TensorProto,
    checker,
    defs,
    helper,
    parser,
    shape_inference,
)
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops.op_dequantize_packed import (
    GGUF_BLOCK_SIZES,
    GGUF_FORMATS,
    _decode_block,
    _GGMLType,
    _grid,
)

_SPECIFICATION_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "DequantizePacked.md"
)

# SHA256 of little-endian FLOAT output from 64 blocks per format, using the
# unmodified dequantize_row_* functions at GGML da8e2a54b919557b816b1c2beb432ccb766f7ce0.
# Q8_1 has no upstream row decoder: its oracle is d * signed qs, ignoring s.
_ORACLE_DIGESTS = {
    2: "09f0f0b2d729e51f2f646e693cc3c6e18b51d795f4b6cbe73097e9be7d87b6e0",
    3: "1b44f52d3d6eb9725387950ae0bd486124a8fd1940edca23eab00bd311132662",
    6: "31398b6c3cac68a02d7507c3b18ca0793edd16b3110210a0249f88c56e5abc31",
    7: "e8328a8f385ad8b9e37ef5b0b97e666bb0cd06ff61be6b1efb971d2098fab9e5",
    8: "2b4f748be8ea71681a7dcf29556f393a9d20d931e0f34eec5116db312ab1c6bf",
    9: "12c88d0aa1a44a5a6dde0d51fad26d7e021f3ac6af8e86b5f24be6fcee211664",
    10: "66915016ab9716dbb720328e90de144109f7ffc1b94d1a071946a3ef559a9143",
    11: "9e556592671f4e83889fd79ee42509cd4d93353727950ecc2712f989b1e989fe",
    12: "a7a2145b24199ff10b9198c38cac8bc558bf1cdc205f3edaa8ac00ec3cae01ac",
    13: "075f87db3496c7a5742fe2e7225bcecd08dc2b5ec16bdfee9fb80562bc55bedf",
    14: "8ee3fb3409c3accaf8db9fdb1aeed9caaf0a195d1cc4c93ebf848b61e8047ca9",
    15: "96100a02d884e76c76fe40589cc81e6a8564670002541cc73c96b4dfd0df8006",
    16: "681174e4bd8c22e1000d5f7f30460173742b7daab8f2277e364c3f9891c332ac",
    17: "0f6f01226d6e1f01a24ba36d11b4854af31e0da8149c2a039d29b231ba8e27fa",
    18: "5d823365863f15bae4877cd80f09f6472c2c06ab0bec8b329eac9c15ad3a7dab",
    19: "a56cb261436667ddb8fe7d80e845ee4f7234b9fc4faf4b9df653868554264611",
    20: "07db142dcc9a02c0e22aa1137f673e791f7eaf8e6ef925cebb7d2be661b7e9c5",
    21: "1f5ae32b37d4272f52cda0a718982e287368c5b8970f9c3faee7db518c10ba00",
    22: "b547a636de8076d309194c7a88234dae1ee8e5452d7f24a573254139447ee544",
    23: "691fb3a9f7266598a48b64d783702c012127a9a983c847dacdecf7bef2856100",
    29: "96889e22b60e45700add104bc47d5280cf608af7f9104a5fd6212de7abefdc3e",
    34: "a191edf81615eb82787f56545027c6680c287510293e35ec4f602ddb84f30e59",
    35: "7c9ca5cf4a7877d79ee1112a48c216a10a2b52f751c66d5a4f6ce036c236b732",
    39: "a259fc1ca83ab1225afd5daeb12da0ca2f1115ef182b48117aedfd254f37da82",
    40: "50f826780120a76023dac805e84bed3688aff1a165cf8a2499ced857cbc8949e",
    41: "9fb4de79ccb7e86f0803f11b83e6c1d8d54d3d20c1b44a65573074f87c55d0e5",
    42: "057c75b6666c8cc1d2de74644fdf630872e0baf042b50024312feb3facd3419b",
}
_CODEBOOK_DIGESTS = {
    "iq2xxs_grid": "05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54",
    "iq2xs_grid": "06e47aaca60b4dc1d9b5a3f34540437058a6b142b4d7a59d5ded769b4d1bf1de",
    "iq2s_grid": "e1aa1473412b0552c2174c30ef22ab4073f6a181b85a17056e8249bd2932fd88",
    "iq3xxs_grid": "46e35f5a997efdee6c99ce57854c8a0d4f0ff8ca57e5e8a60c0793ea580acf5d",
    "iq3s_grid": "bd1af4945a1717c65610b0284e4628b9a1ba3b306fae3a06f6e5f597356e349f",
    "iq1s_grid": "07540ffc1aeaf6ad4d97e96b0fcc765aae39671d4ae4a27bbd0e796fde167c6a",
}


def _corpus(qtype: int, count: int = 64) -> np.ndarray:
    """Deterministic nonzero fields, varied indices/signs, and finite scales."""
    _, size = GGUF_BLOCK_SIZES[qtype]
    blocks = np.empty((count, size), dtype=np.uint8)
    scale_offsets = {
        2: (0,),
        3: (0, 2),
        6: (0,),
        7: (0, 2),
        8: (0,),
        9: (0,),
        10: (80, 82),
        11: (108,),
        12: (0, 2),
        13: (0, 2),
        14: (208,),
        16: (0,),
        17: (0,),
        18: (0,),
        19: (0,),
        20: (0,),
        21: (0,),
        22: (0,),
        23: (0,),
        34: (52,),
        35: (64,),
        41: (0,),
        42: (0,),
    }
    scales = [0.3125, -0.625, 0, 1.125, 0.1, -3.25, 2**-24, 65504]
    for i in range(count):
        block = bytearray(
            (j * 73 + i * 29 + (j // 7) * 17 + (i // 3) * 11) & 255 for j in range(size)
        )
        for k, offset in enumerate(scale_offsets.get(qtype, ())):
            struct.pack_into("<e", block, offset, scales[(i + k) % len(scales)])
        if qtype == _GGMLType.Q8_K:
            struct.pack_into("<f", block, 0, scales[i % len(scales)] / 3)
        if qtype == _GGMLType.IQ1_M:
            bits = int.from_bytes(struct.pack("<e", scales[i % len(scales)]), "little")
            for k in range(4):
                offset = 48 + 2 * k
                word = int.from_bytes(block[offset : offset + 2], "little") & 0x0FFF
                struct.pack_into(
                    "<H", block, offset, word | (((bits >> (4 * k)) & 15) << 12)
                )
        if qtype == _GGMLType.MXFP4:
            block[0] = (i * 17) & 255
        blocks[i] = np.frombuffer(block, dtype=np.uint8)
    return blocks


def _run(
    x: np.ndarray,
    qtype: int | str = 2,
    shape=None,
    output_dtype: int = TensorProto.FLOAT,
) -> np.ndarray:
    node = helper.make_node(
        "DequantizePacked",
        ["x"],
        ["y"],
        format=f"gguf:{_GGMLType(qtype).name.lower()}"
        if isinstance(qtype, int)
        else qtype,
        output_dtype=output_dtype,
    )
    node.attribute.append(
        helper.make_attribute(
            "shape",
            [32] if shape is None else shape,
            attr_type=AttributeProto.INTS,
        )
    )
    return ReferenceEvaluator(node).run(None, {"x": x})[0]


def test_complete_format_set() -> None:
    assert set(GGUF_BLOCK_SIZES) == set(_ORACLE_DIGESTS)
    assert set(GGUF_FORMATS.values()) == set(_ORACLE_DIGESTS)


def test_format_attribute_schema() -> None:
    schema = defs.get_schema("DequantizePacked", 29)
    assert schema.attributes["format"].type == defs.OpSchema.AttrType.STRING
    assert schema.attributes["format"].required
    assert "ggml_type" not in schema.attributes


def test_documented_format_identifiers() -> None:
    expected = {
        name: (kind.name, *GGUF_BLOCK_SIZES[kind])
        for name, kind in GGUF_FORMATS.items()
    }
    for text in (
        _SPECIFICATION_PATH.read_text(encoding="utf-8"),
        defs.get_schema("DequantizePacked", 29).doc,
    ):
        rows = re.findall(
            r"^\|\s*(gguf:[a-z0-9_]+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|$",
            text,
            flags=re.MULTILINE,
        )
        assert len(rows) == len(expected)
        assert {
            name: (kind, int(width), int(size)) for name, kind, width, size in rows
        } == expected


@pytest.mark.parametrize("attrs", [{"format": 2}, {"ggml_type": 2}, {}])
def test_format_attribute_required_string(attrs) -> None:
    model = parser.parse_model("""
        <ir_version: 13, opset_import: ["" : 29]>
        gguf (uint8[18] x) => (float[32] y) {
            y = DequantizePacked<shape=[32]>(x)
        }
    """)
    model.graph.node[0].attribute.extend(
        helper.make_attribute(name, value) for name, value in attrs.items()
    )
    with pytest.raises(checker.ValidationError):
        checker.check_model(model)


def test_reference_rejects_numeric_format() -> None:
    node = helper.make_node("DequantizePacked", ["x"], ["y"], format=2, shape=[32])
    with pytest.raises(ValueError, match="Unsupported quantization format"):
        ReferenceEvaluator(node).run(None, {"x": np.zeros(18, dtype=np.uint8)})


def _load_specification():
    blocks = re.findall(
        r"^```python\n(.*?)^```$",
        _SPECIFICATION_PATH.read_text(encoding="utf-8"),
        flags=re.MULTILINE | re.DOTALL,
    )
    namespace = {}
    exec("\n".join(blocks), namespace)  # noqa: S102
    return namespace


@pytest.fixture(scope="module")
def specification():
    return _load_specification()


@pytest.mark.parametrize("qtype", sorted(_ORACLE_DIGESTS))
def test_specification_decoding(specification, qtype: int) -> None:
    actual = np.array(
        [
            specification["decode_block"](block.tobytes(), _GGMLType(qtype).name)
            for block in _corpus(qtype)
        ],
        dtype=np.float32,
    )
    digest = hashlib.sha256(actual.astype("<f4").tobytes()).hexdigest()
    assert digest == _ORACLE_DIGESTS[qtype]


@pytest.mark.parametrize(
    "name,rows,width",
    [
        ("iq2xxs_grid", 256, 8),
        ("iq2xs_grid", 512, 8),
        ("iq2s_grid", 1024, 8),
        ("iq3xxs_grid", 256, 4),
        ("iq3s_grid", 512, 4),
        ("iq1s_grid", 2048, 8),
    ],
)
def test_specification_codebooks(specification, name, rows, width) -> None:
    width_in_spec, _, bits, text = specification["CODEBOOKS"][name]
    assert width_in_spec == width
    assert len(bytes.fromhex(text)) * 8 == rows * width * bits
    data = np.array(
        [specification["grid"](name, i) for i in range(rows)], dtype=np.int8
    )
    assert hashlib.sha256(data.tobytes()).hexdigest() == _CODEBOOK_DIGESTS[name]


@pytest.mark.parametrize("qtype", [39, 40])
def test_specification_block_float_scales(specification, qtype: int) -> None:
    quants = bytes((j % 16) | ((15 - j % 16) << 4) for j in range(32))
    for code in range(256):
        block = (
            bytes([code]) + quants[:16]
            if qtype == _GGMLType.MXFP4
            else bytes([code] * 4) + quants
        )
        actual = np.array(
            specification["decode_block"](block, _GGMLType(qtype).name),
            dtype=np.float32,
        )
        with np.errstate(over="ignore"):
            expected = _decode_block(block, qtype)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("qtype", sorted(set(GGUF_BLOCK_SIZES) - {39, 40}))
def test_specification_ieee_scales(specification, qtype: int) -> None:
    scale_bits = [0, 0x8000, 1, 0x3C00, 0x7BFF, 0x7C00, 0xFC00, 0x7E00]
    for block, bits in zip(_corpus(qtype, len(scale_bits)), scale_bits, strict=True):
        data = bytearray(block.tobytes())
        if qtype == _GGMLType.Q8_K:
            struct.pack_into(
                "<f", data, 0, struct.unpack("<e", bits.to_bytes(2, "little"))[0]
            )
        elif qtype == _GGMLType.IQ1_M:
            for j in range(4):
                offset = 48 + 2 * j
                word = int.from_bytes(data[offset : offset + 2], "little") & 0x0FFF
                struct.pack_into(
                    "<H", data, offset, word | (((bits >> (4 * j)) & 15) << 12)
                )
        else:
            offset = {10: 80, 11: 108, 14: 208, 34: 52, 35: 64}.get(qtype, 0)
            struct.pack_into("<H", data, offset, bits)
        actual = np.array(
            specification["decode_block"](data, _GGMLType(qtype).name), dtype=np.float32
        )
        with np.errstate(over="ignore", invalid="ignore"):
            expected = _decode_block(bytes(data), qtype)
        np.testing.assert_array_equal(actual, expected)
        non_nan = ~np.isnan(expected)
        np.testing.assert_array_equal(
            actual[non_nan].view(np.uint32), expected[non_nan].view(np.uint32)
        )


@pytest.mark.parametrize("qtype", sorted(_ORACLE_DIGESTS))
def test_upstream_oracle(qtype: int) -> None:
    blocks = _corpus(qtype)
    width, _ = GGUF_BLOCK_SIZES[qtype]
    with np.errstate(over="ignore", invalid="ignore"):
        actual = _run(blocks.ravel(), qtype, [2, 32, width])
    assert actual.shape == (2, 32, width)
    assert actual.dtype == np.float32
    digest = hashlib.sha256(actual.astype("<f4").tobytes()).hexdigest()
    assert digest == _ORACLE_DIGESTS[qtype], (
        f"GGML type {qtype} disagrees with upstream C"
    )


@pytest.mark.parametrize(
    "name",
    ["iq2xxs_grid", "iq2xs_grid", "iq2s_grid", "iq3xxs_grid", "iq3s_grid", "iq1s_grid"],
)
def test_complete_codebooks(name: str) -> None:
    digest = hashlib.sha256(_grid(name).astype(np.int8).tobytes()).hexdigest()
    assert digest == _CODEBOOK_DIGESTS[name]


@pytest.mark.parametrize(
    "dtype",
    [TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.DOUBLE],
)
def test_output_dtype_order_and_storage(
    dtype: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks = _corpus(2, 6)
    expected = np.stack(
        [_decode_block(block.tobytes(), 2) for block in blocks]
    ).reshape(2, 3, 32)
    decoded_buffers: list[np.ndarray] = []
    original_empty = np.empty

    def track_empty(*args, **kwargs) -> np.ndarray:
        allocated = original_empty(*args, **kwargs)
        if allocated.shape == (expected.size,) and allocated.dtype == np.float32:
            decoded_buffers.append(allocated)
        return allocated

    monkeypatch.setattr(np, "empty", track_empty)
    actual = _run(blocks.ravel(), shape=[2, 3, 32], output_dtype=dtype)
    np.testing.assert_array_equal(
        actual, expected.astype(helper.tensor_dtype_to_np_dtype(dtype))
    )
    assert actual.dtype == helper.tensor_dtype_to_np_dtype(dtype)
    assert actual.flags.c_contiguous
    assert len(decoded_buffers) == 1
    assert np.shares_memory(actual, decoded_buffers[0]) == (dtype == TensorProto.FLOAT)
    assert not np.shares_memory(actual, blocks)


def test_default_output_dtype() -> None:
    model = parser.parse_model("""
        <ir_version: 13, opset_import: ["" : 29]>
        gguf (uint8[18] x) => (float[32] y) {
            y = DequantizePacked<format="gguf:q4_0", shape=[32]>(x)
        }
    """)
    checker.check_model(model, full_check=True)
    block = np.frombuffer(struct.pack("<e", 0.5) + bytes(range(16)), dtype=np.uint8)
    actual = ReferenceEvaluator(model).run(None, {"x": block})[0]
    expected = np.array(list(np.arange(-8, 8) * 0.5) + [-4] * 16, dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("qtype", sorted(GGUF_BLOCK_SIZES))
def test_empty(qtype: int) -> None:
    width, _ = GGUF_BLOCK_SIZES[qtype]
    actual = _run(np.empty(0, dtype=np.uint8), qtype, [0, width])
    assert actual.shape == (0, width)
    assert actual.dtype == np.float32


@pytest.mark.parametrize(
    "name",
    [
        "",
        "q4_0",
        "GGUF:q4_0",
        "gguf:Q4_0",
        "gguf:q2_K",
        "onnx:q4_0",
        "gguf:f32",
        "gguf:f16",
        "gguf:bf16",
        "gguf:i8",
        "gguf:q4_2",
        "gguf:q4_3",
        "gguf:q4_0_4_4",
        "gguf:unknown",
        "gguf:2",
        "2",
        "gguf:q4_0 ",
    ],
)
def test_rejected_formats(name: str) -> None:
    with pytest.raises(ValueError, match="Unsupported quantization format"):
        _run(np.zeros(18, dtype=np.uint8), name)
    with pytest.raises(
        shape_inference.InferenceError, match="Unsupported quantization format"
    ):
        _infer(format=name)


@pytest.mark.parametrize("shape", [[], [-1, 32], [31], [0, 31]])
def test_invalid_shape(shape) -> None:
    with pytest.raises(ValueError, match=r"shape|dimension"):
        _run(np.zeros(18, dtype=np.uint8), shape=shape)


@pytest.mark.parametrize(
    "x", [np.zeros(17, dtype=np.uint8), np.zeros(19, dtype=np.uint8)]
)
def test_invalid_length(x) -> None:
    with pytest.raises(ValueError, match="exactly 18 bytes"):
        _run(x)


@pytest.mark.parametrize(
    "x",
    [
        np.zeros((1, 18), dtype=np.uint8),
        np.zeros(18, dtype=np.int8),
        np.zeros((), dtype=np.uint8),
    ],
)
def test_invalid_input(x) -> None:
    with pytest.raises(ValueError, match="one-dimensional UINT8"):
        _run(x)


@pytest.mark.parametrize(
    "dtype", [TensorProto.UNDEFINED, TensorProto.INT32, TensorProto.FLOAT8E4M3FN]
)
def test_invalid_output_dtype(dtype: int) -> None:
    with pytest.raises(ValueError, match="output_dtype"):
        _run(np.zeros(18, dtype=np.uint8), output_dtype=dtype)


def test_cached_sums_are_ignored() -> None:
    for qtype, start, end in [(9, 2, 4), (15, 260, 292)]:
        blocks = _corpus(qtype, 2)
        changed = blocks.copy()
        changed[:, start:end] ^= np.uint8(255)
        np.testing.assert_array_equal(
            _run(blocks.ravel(), qtype, [2, GGUF_BLOCK_SIZES[qtype][0]]),
            _run(changed.ravel(), qtype, [2, GGUF_BLOCK_SIZES[qtype][0]]),
        )


@pytest.mark.parametrize("qtype", [39, 40])
def test_all_block_float_scale_bytes(qtype: int) -> None:
    blocks = []
    expected = []
    for code in range(256):
        if qtype == _GGMLType.MXFP4:
            blocks.append(bytes([code]) + bytes([0x21] * 16))
            scale = 2.0 ** (code - 127)
            with np.errstate(over="ignore"):
                expected.extend(
                    [np.float32(scale * 0.5)] * 16 + [np.float32(scale)] * 16
                )
        else:
            blocks.append(bytes([code] * 4) + bytes([0x21] * 32))
            exponent, mantissa = (code >> 3) & 15, code & 7
            scale = np.float32(
                0
                if code in (0, 127)
                else (
                    mantissa * 2**-9
                    if exponent == 0
                    else (1 + mantissa / 8) * 2.0 ** (exponent - 7)
                )
            )
            expected.extend(([scale * np.float32(0.5)] * 8 + [scale] * 8) * 4)
    width = GGUF_BLOCK_SIZES[qtype][0]
    with np.errstate(over="ignore"):
        actual = _run(
            np.frombuffer(b"".join(blocks), dtype=np.uint8), qtype, [256, width]
        )
    np.testing.assert_array_equal(actual.ravel(), np.array(expected, dtype=np.float32))


def _infer(input_shape=(18,), input_type=TensorProto.UINT8, **attrs):
    attributes = {"format": "gguf:q4_0", "output_dtype": TensorProto.FLOAT, **attrs}
    shape = attributes.pop("shape", [32])
    node = helper.make_node("DequantizePacked", ["x"], ["y"], **attributes)
    node.attribute.append(
        helper.make_attribute("shape", shape, attr_type=AttributeProto.INTS)
    )
    graph = helper.make_graph(
        [node],
        "gguf",
        [helper.make_tensor_value_info("x", input_type, input_shape)],
        [helper.make_empty_tensor_value_info("y")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 29)])
    return shape_inference.infer_shapes(model, check_type=True, strict_mode=True)


@pytest.mark.parametrize(
    "elem_type",
    [TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.DOUBLE],
)
@pytest.mark.parametrize("input_shape", [(36,), (None,), ("bytes",), None])
def test_shape_inference(elem_type, input_shape) -> None:
    model = _infer(input_shape, shape=[2, 32], output_dtype=elem_type)
    out = model.graph.output[0].type.tensor_type
    assert out.elem_type == elem_type
    assert [dim.dim_value for dim in out.shape.dim] == [2, 32]


@pytest.mark.parametrize(
    "qtype,block,packed",
    [
        (2, 32, 18),
        (3, 32, 20),
        (6, 32, 22),
        (7, 32, 24),
        (8, 32, 34),
        (9, 32, 36),
        (10, 256, 84),
        (11, 256, 110),
        (12, 256, 144),
        (13, 256, 176),
        (14, 256, 210),
        (15, 256, 292),
        (16, 256, 66),
        (17, 256, 74),
        (18, 256, 98),
        (19, 256, 50),
        (20, 32, 18),
        (21, 256, 110),
        (22, 256, 82),
        (23, 256, 136),
        (29, 256, 56),
        (34, 256, 54),
        (35, 256, 66),
        (39, 32, 17),
        (40, 64, 36),
        (41, 128, 18),
        (42, 64, 18),
    ],
)
def test_shape_inference_formats(qtype, block, packed) -> None:
    model = _infer(
        (packed * 6,),
        format=f"gguf:{_GGMLType(qtype).name.lower()}",
        shape=[2, 3, block],
    )
    out = model.graph.output[0].type.tensor_type
    assert out.elem_type == TensorProto.FLOAT
    assert [dim.dim_value for dim in out.shape.dim] == [2, 3, block]


@pytest.mark.parametrize("shape", [[0, 32], [2, 0], [0, 2**62, 32]])
def test_shape_inference_empty(shape) -> None:
    model = _infer((0,), shape=shape)
    assert [
        dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim
    ] == shape


@pytest.mark.parametrize(
    "input_shape,attrs",
    [
        ((18,), {"shape": []}),
        ((18,), {"shape": [-1, 32]}),
        ((18,), {"shape": [31]}),
        ((0,), {"shape": [0, 31]}),
        ((17,), {"shape": [32]}),
        ((19,), {"shape": [32]}),
        ((1, 18), {"shape": [32]}),
        ((), {"shape": [32]}),
        ((18,), {"format": 2}),
        ((18,), {"output_dtype": TensorProto.INT32}),
        ((None,), {"shape": [2**62, 32]}),
    ],
)
def test_shape_inference_invalid(input_shape, attrs) -> None:
    with pytest.raises(shape_inference.InferenceError):
        _infer(input_shape, **attrs)


def test_shape_inference_invalid_input_type() -> None:
    with pytest.raises(shape_inference.InferenceError):
        _infer(input_type=TensorProto.INT8)
