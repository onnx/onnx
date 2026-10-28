# Copyright (c) ONNX Project Contributors
# Copyright (c) 2023-2026 The ggml authors
#
# SPDX-License-Identifier: Apache-2.0 AND MIT
"""GGML block decoding at revision da8e2a54b919557b816b1c2beb432ccb766f7ce0.

The upstream MIT notice is retained in GGML_LICENSE.
"""

from __future__ import annotations

import math
import struct
from enum import IntEnum
from functools import lru_cache

import numpy as np

from onnx import TensorProto, helper
from onnx.reference.op_run import OpRun
from onnx.reference.ops._gguf_codebooks import CODEBOOKS


class _GGMLType(IntEnum):
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    IQ1_M = 29
    TQ1_0 = 34
    TQ2_0 = 35
    MXFP4 = 39
    NVFP4 = 40
    Q1_0 = 41
    Q2_0 = 42


# Private GGML IDs used by the translated decoders.
# The operator accepts only the ONNX format identifiers in GGUF_FORMATS.
GGUF_BLOCK_SIZES = {
    2: (32, 18),
    3: (32, 20),
    6: (32, 22),
    7: (32, 24),
    8: (32, 34),
    9: (32, 36),
    10: (256, 84),
    11: (256, 110),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    15: (256, 292),
    16: (256, 66),
    17: (256, 74),
    18: (256, 98),
    19: (256, 50),
    20: (32, 18),
    21: (256, 110),
    22: (256, 82),
    23: (256, 136),
    29: (256, 56),
    34: (256, 54),
    35: (256, 66),
    39: (32, 17),
    40: (64, 36),
    41: (128, 18),
    42: (64, 18),
}
GGUF_FORMATS = {f"gguf:{kind.name.lower()}": kind for kind in _GGMLType}
_IQ4 = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float32,
)
_FP4 = np.array(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=np.float32,
)


@lru_cache(maxsize=6)
def _grid(name: str) -> np.ndarray:
    width, values, bits, text = CODEBOOKS[name]
    packed = np.frombuffer(bytes.fromhex(text), dtype=np.uint8)
    indices = (packed[:, None] >> np.arange(0, 8, bits, dtype=np.uint8)) & (
        (1 << bits) - 1
    )
    return np.array(values, dtype=np.float32)[indices.ravel()].reshape(-1, width)


def _half(block: bytes, offset: int = 0) -> np.float32:
    return np.float32(struct.unpack_from("<e", block, offset)[0])


def _uint(block: bytes, offset: int, size: int = 2) -> int:
    return int.from_bytes(block[offset : offset + size], "little")


def _unpack(data: bytes, bits: int, group: int) -> np.ndarray:
    """Unpack bit planes within groups, not adjacent element bits."""
    packed = np.frombuffer(data, dtype=np.uint8).reshape(-1, group)
    return (
        (
            (packed[:, None, :] >> np.arange(0, 8, bits, dtype=np.uint8)[None, :, None])
            & ((1 << bits) - 1)
        )
        .ravel()
        .astype(np.int32)
    )


def _signs(index: int, parity: bool = False) -> np.ndarray:
    if parity:
        index |= (index.bit_count() & 1) << 7
    return np.array([1 - 2 * ((index >> j) & 1) for j in range(8)], dtype=np.float32)


def _decode_iq(block: bytes, qtype: int) -> np.ndarray:
    d = _half(block)
    out = np.empty(256, dtype=np.float32)
    if qtype == _GGMLType.IQ1_M:
        scales = [_uint(block, 48 + 2 * j) for j in range(4)]
        scale_bits = sum((scales[j] >> 12) << (4 * j) for j in range(4))
        d = _half(scale_bits.to_bytes(2, "little"))
    for group in range(8):
        aux = 0
        if qtype == _GGMLType.IQ2_XXS:
            aux = _uint(block, 6 + 8 * group, 4)
        elif qtype == _GGMLType.IQ3_XXS:
            aux = _uint(block, 66 + 4 * group, 4)
        for lane in range(4):
            i = 4 * group + lane
            if qtype in (_GGMLType.IQ2_XXS, _GGMLType.IQ2_XS, _GGMLType.IQ2_S):
                if qtype == _GGMLType.IQ2_XXS:
                    index = block[2 + 8 * group + lane]
                    sign = _signs((aux >> (7 * lane)) & 127, parity=True)
                    scale = d * np.float32(0.5 + (aux >> 28)) * np.float32(0.25)
                    name = "iq2xxs_grid"
                elif qtype == _GGMLType.IQ2_XS:
                    word = _uint(block, 2 + 2 * i)
                    index = word & 511
                    sign = _signs(word >> 9, parity=True)
                    scale = (
                        d
                        * np.float32(
                            0.5 + ((block[66 + group] >> (4 * (lane // 2))) & 15)
                        )
                        * np.float32(0.25)
                    )
                    name = "iq2xs_grid"
                else:
                    index = block[2 + i] | (
                        ((block[66 + group] >> (2 * lane)) & 3) << 8
                    )
                    sign = _signs(block[34 + i])
                    scale = (
                        d
                        * np.float32(
                            0.5 + ((block[74 + group] >> (4 * (lane // 2))) & 15)
                        )
                        * np.float32(0.25)
                    )
                    name = "iq2s_grid"
                out[8 * i : 8 * i + 8] = (scale * _grid(name)[index]) * sign
            elif qtype in (_GGMLType.IQ3_XXS, _GGMLType.IQ3_S):
                if qtype == _GGMLType.IQ3_XXS:
                    indices = [block[2 + 2 * i + j] for j in range(2)]
                    sign = _signs((aux >> (7 * lane)) & 127, parity=True)
                    scale = d * np.float32(0.5 + (aux >> 28)) * np.float32(0.5)
                    name = "iq3xxs_grid"
                else:
                    indices = [
                        block[2 + 2 * i + j]
                        | (((block[66 + group] >> (2 * lane + j)) & 1) << 8)
                        for j in range(2)
                    ]
                    sign = _signs(block[74 + i])
                    scale = d * np.float32(
                        1 + 2 * ((block[106 + group // 2] >> (4 * (group % 2))) & 15)
                    )
                    name = "iq3s_grid"
                values = _grid(name)[indices].ravel()
                out[8 * i : 8 * i + 8] = (scale * values) * sign
            else:
                if qtype == _GGMLType.IQ1_S:
                    high = _uint(block, 34 + 2 * group)
                    index = block[2 + i] | (((high >> (3 * lane)) & 7) << 8)
                    scale = d * np.float32(2 * ((high >> 12) & 7) + 1)
                    delta = np.float32(-0.125 if high & 0x8000 else 0.125)
                else:
                    high = block[32 + i // 2] >> (4 * (i % 2))
                    index = block[i] | ((high & 7) << 8)
                    scale = d * np.float32(
                        2
                        * (
                            (scales[group // 2] >> (6 * (group % 2) + 3 * (lane // 2)))
                            & 7
                        )
                        + 1
                    )
                    delta = np.float32(-0.125 if high & 8 else 0.125)
                out[8 * i : 8 * i + 8] = scale * (_grid("iq1s_grid")[index] + delta)
    return out


def _decode_block(block: bytes, qtype: int) -> np.ndarray:  # noqa: PLR0911
    if qtype in (
        _GGMLType.IQ2_XXS,
        _GGMLType.IQ2_XS,
        _GGMLType.IQ3_XXS,
        _GGMLType.IQ1_S,
        _GGMLType.IQ3_S,
        _GGMLType.IQ2_S,
        _GGMLType.IQ1_M,
    ):
        return _decode_iq(block, qtype)
    if qtype in (
        _GGMLType.Q4_0,
        _GGMLType.Q4_1,
        _GGMLType.Q5_0,
        _GGMLType.Q5_1,
        _GGMLType.IQ4_NL,
    ):
        affine = qtype in (_GGMLType.Q4_1, _GGMLType.Q5_1)
        offset = 4 if affine else 2
        if qtype in (_GGMLType.Q5_0, _GGMLType.Q5_1):
            high = _uint(block, offset, 4)
            q = _unpack(block[offset + 4 :], 4, 16)
            q |= np.array([((high >> j) & 1) << 4 for j in range(32)])
        else:
            q = _unpack(block[offset:], 4, 16)
        if qtype == _GGMLType.IQ4_NL:
            return _half(block) * _IQ4[q]
        if not affine:
            q -= 16 if qtype == _GGMLType.Q5_0 else 8
        y = _half(block) * q.astype(np.float32)
        return y + _half(block, 2) if affine else y
    if qtype in (_GGMLType.Q8_0, _GGMLType.Q8_1, _GGMLType.Q8_K):
        offset = 2 if qtype == _GGMLType.Q8_0 else 4
        count = 256 if qtype == _GGMLType.Q8_K else 32
        d = (
            np.float32(struct.unpack_from("<f", block)[0])
            if qtype == _GGMLType.Q8_K
            else _half(block)
        )
        return d * np.frombuffer(block[offset : offset + count], dtype=np.int8).astype(
            np.float32
        )
    if qtype in (_GGMLType.Q2_K, _GGMLType.Q3_K):
        offset = 16 if qtype == _GGMLType.Q2_K else 32
        q = _unpack(block[offset : offset + 64], 2, 32)
        if qtype == _GGMLType.Q2_K:
            scales = np.frombuffer(block[:16], dtype=np.uint8)
            d, m = _half(block, 80), _half(block, 82)
            return (
                (d * (scales & 15).astype(np.float32))[:, None]
                * q.reshape(16, 16).astype(np.float32)
                - (m * (scales >> 4).astype(np.float32))[:, None]
            ).ravel()
        low = _unpack(block[96:104], 4, 8)
        high = _unpack(block[104:108], 2, 4)
        scales = (low | (high << 4)) - 32
        q -= (1 - _unpack(block[:32], 1, 32)) * 4
        return (
            (_half(block, 108) * scales.astype(np.float32))[:, None]
            * q.reshape(16, 16).astype(np.float32)
        ).ravel()
    if qtype in (_GGMLType.Q4_K, _GGMLType.Q5_K):
        scales = block[4:16]
        sc, mn = [], []
        scale_group_size = 4
        for j in range(8):
            sc.append(
                scales[j] & 63
                if j < scale_group_size
                else (scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4)
            )
            mn.append(
                scales[j + 4] & 63
                if j < scale_group_size
                else (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
            )
        q = _unpack(block[16 if qtype == _GGMLType.Q4_K else 48 :], 4, 32)
        if qtype == _GGMLType.Q5_K:
            q |= _unpack(block[16:48], 1, 32) << 4
        return (
            (_half(block) * np.array(sc, dtype=np.float32))[:, None]
            * q.reshape(8, 32).astype(np.float32)
            - (_half(block, 2) * np.array(mn, dtype=np.float32))[:, None]
        ).ravel()
    if qtype == _GGMLType.Q6_K:
        q = (_unpack(block[:128], 4, 64) | (_unpack(block[128:192], 2, 32) << 4)) - 32
        scales = np.frombuffer(block[192:208], dtype=np.int8).astype(np.float32)
        return (
            (_half(block, 208) * scales)[:, None] * q.reshape(16, 16).astype(np.float32)
        ).ravel()
    if qtype == _GGMLType.IQ4_XS:
        high = _uint(block, 2)
        scales = np.array(
            [
                ((block[4 + j // 2] >> (4 * (j % 2))) & 15)
                | (((high >> (2 * j)) & 3) << 4)
                for j in range(8)
            ],
            dtype=np.float32,
        ) - np.float32(32)
        q = _unpack(block[8:], 4, 16)
        return ((_half(block) * scales)[:, None] * _IQ4[q].reshape(8, 32)).ravel()
    if qtype == _GGMLType.TQ1_0:
        q = []
        for start, width, planes in ((0, 32, 5), (32, 16, 5), (48, 4, 4)):
            q.extend(
                ((block[start + j] * 3**plane) & 255) * 3 // 256 - 1
                for plane in range(planes)
                for j in range(width)
            )
        return _half(block, 52) * np.array(q, dtype=np.float32)
    if qtype == _GGMLType.TQ2_0:
        return _half(block, 64) * (_unpack(block[:64], 2, 32) - 1).astype(np.float32)
    if qtype == _GGMLType.MXFP4:
        e = block[0]
        bits = (0x00200000 << e) if e in (0, 1) else ((e - 1) << 23)
        d = np.float32(struct.unpack("<f", bits.to_bytes(4, "little"))[0])
        return d * _FP4[_unpack(block[1:], 4, 16)]
    if qtype == _GGMLType.NVFP4:
        out = np.empty(64, dtype=np.float32)
        for j in range(4):
            e = block[j]
            exponent, mantissa = (e >> 3) & 15, e & 7
            d = np.float32(
                0
                if e in (0, 127)
                else (
                    math.ldexp(mantissa, -10)
                    if exponent == 0
                    else math.ldexp(1 + mantissa / 8, exponent - 8)
                )
            )
            out[16 * j : 16 * j + 16] = (
                d * _FP4[_unpack(block[4 + 8 * j : 12 + 8 * j], 4, 8)]
            )
        return out
    if qtype == _GGMLType.Q1_0:
        q = np.unpackbits(np.frombuffer(block[2:], dtype=np.uint8), bitorder="little")
        return np.where(q != 0, _half(block), -_half(block))
    if qtype == _GGMLType.Q2_0:
        return _half(block) * (_unpack(block[2:], 2, 1) - 1).astype(np.float32)
    raise ValueError(f"Unsupported quantized ggml_type: {qtype}")


class DequantizePacked(OpRun):
    def _run(self, x, format=None, shape=None, output_dtype=None):  # noqa: A002
        if format not in GGUF_FORMATS:
            raise ValueError(f"Unsupported quantization format: {format}")
        qtype = GGUF_FORMATS[format]
        if x.dtype != np.uint8 or x.ndim != 1:
            raise ValueError("x must be a one-dimensional UINT8 tensor.")
        if output_dtype not in (
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.BFLOAT16,
            TensorProto.DOUBLE,
        ):
            raise ValueError(
                "output_dtype must be FLOAT, FLOAT16, BFLOAT16, or DOUBLE."
            )
        if not shape or any(dim < 0 for dim in shape):
            raise ValueError(
                "shape must have rank at least one and nonnegative dimensions."
            )
        block_size, block_bytes = GGUF_BLOCK_SIZES[qtype]
        if shape[-1] % block_size:
            raise ValueError(
                "The last output dimension must be a multiple of the quantization block size."
            )
        count = math.prod(shape)
        expected = count // block_size * block_bytes
        if expected > np.iinfo(np.int64).max:
            raise ValueError("The packed byte count exceeds int64 range.")
        if x.size != expected:
            raise ValueError(
                f"Packed input length must be exactly {expected} bytes, got {x.size}."
            )
        out = np.empty(count, dtype=np.float32)
        for j, block in enumerate(x.reshape(-1, block_bytes)):
            out[j * block_size : (j + 1) * block_size] = _decode_block(
                block.tobytes(), qtype
            )
        return (
            out.reshape(tuple(shape)).astype(
                helper.tensor_dtype_to_np_dtype(output_dtype)
            ),
        )
