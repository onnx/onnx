# Copyright (c) ONNX Project Contributors
# Copyright (c) 2023-2026 The ggml authors
#
# SPDX-License-Identifier: Apache-2.0 AND MIT
"""Packed GGML quantization profiles.

GGML decoding follows da8e2a54b919557b816b1c2beb432ccb766f7ce0;
the upstream MIT notice is retained in GGML_LICENSE.
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
_UINT_DTYPES = {2: "<u2", 4: "<u4"}
_NVFP4_ZERO_SCALE_CODE = 127


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


def _batch_half(blocks: np.ndarray, offset: int = 0) -> np.ndarray:
    return (
        np.ascontiguousarray(blocks[:, offset : offset + 2])
        .view("<f2")
        .reshape(-1)
        .astype(np.float32)
    )


def _batch_uint(blocks: np.ndarray, offset: int, size: int = 2) -> np.ndarray:
    data = np.ascontiguousarray(blocks[:, offset : offset + size])
    dtype = _UINT_DTYPES.get(size)
    if dtype is None:
        raise ValueError(f"Unsupported integer byte width: {size}")
    return data.view(dtype).reshape(-1)


def _batch_unpack(data: np.ndarray, bits: int, group: int) -> np.ndarray:
    packed = data.reshape(len(data), -1, group)
    shifts = np.arange(0, 8, bits, dtype=np.uint8)
    return (
        ((packed[:, :, None, :] >> shifts[None, None, :, None]) & ((1 << bits) - 1))
        .reshape(len(data), -1)
        .astype(np.int32)
    )


def _batch_signs(index: np.ndarray, parity: bool = False) -> np.ndarray:
    if parity:
        parity_bit = np.zeros_like(index)
        for shift in range(7):
            parity_bit ^= (index >> shift) & 1
        index = index | (parity_bit << 7)
    bits = ((index[:, None] >> np.arange(8, dtype=index.dtype)[None, :]) & 1).astype(
        np.float32
    )
    return np.float32(1) - np.float32(2) * bits


def _decode_iq_blocks(blocks: np.ndarray, out: np.ndarray, qtype: int) -> None:
    d = _batch_half(blocks)
    if qtype == _GGMLType.IQ1_M:
        scales = np.stack([_batch_uint(blocks, 48 + 2 * j) for j in range(4)], axis=1)
        scale_bits = np.zeros(len(blocks), dtype=np.uint16)
        for j in range(4):
            scale_bits |= ((scales[:, j] >> 12) << (4 * j)).astype(np.uint16)
        d = scale_bits.astype("<u2", copy=False).view("<f2").astype(np.float32)
    for group in range(8):
        aux = None
        if qtype == _GGMLType.IQ2_XXS:
            aux = _batch_uint(blocks, 6 + 8 * group, 4)
        elif qtype == _GGMLType.IQ3_XXS:
            aux = _batch_uint(blocks, 66 + 4 * group, 4)
        for lane in range(4):
            i = 4 * group + lane
            decoded = out[:, 8 * i : 8 * i + 8]
            if qtype in (_GGMLType.IQ2_XXS, _GGMLType.IQ2_XS, _GGMLType.IQ2_S):
                if qtype == _GGMLType.IQ2_XXS:
                    assert aux is not None
                    index = blocks[:, 2 + 8 * group + lane]
                    sign = _batch_signs(
                        ((aux >> (7 * lane)) & 127).astype(np.uint32), parity=True
                    )
                    scale = d * (np.float32(0.5) + (aux >> 28).astype(np.float32))
                    scale *= np.float32(0.25)
                    name = "iq2xxs_grid"
                elif qtype == _GGMLType.IQ2_XS:
                    word = _batch_uint(blocks, 2 + 2 * i)
                    index = word & 511
                    sign = _batch_signs((word >> 9).astype(np.uint16), parity=True)
                    nibble = (blocks[:, 66 + group] >> np.uint8(4 * (lane // 2))) & 15
                    scale = d * (np.float32(0.5) + nibble.astype(np.float32))
                    scale *= np.float32(0.25)
                    name = "iq2xs_grid"
                else:
                    index = blocks[:, 2 + i].astype(np.uint16) | (
                        ((blocks[:, 66 + group] >> np.uint8(2 * lane)) & 3).astype(
                            np.uint16
                        )
                        << 8
                    )
                    sign = _batch_signs(blocks[:, 34 + i])
                    nibble = (blocks[:, 74 + group] >> np.uint8(4 * (lane // 2))) & 15
                    scale = d * (np.float32(0.5) + nibble.astype(np.float32))
                    scale *= np.float32(0.25)
                    name = "iq2s_grid"
                np.multiply(scale[:, None], _grid(name)[index], out=decoded)
                np.multiply(decoded, sign, out=decoded)
            elif qtype in (_GGMLType.IQ3_XXS, _GGMLType.IQ3_S):
                if qtype == _GGMLType.IQ3_XXS:
                    assert aux is not None
                    indices = blocks[:, 2 + 2 * i : 2 + 2 * i + 2]
                    sign = _batch_signs(
                        ((aux >> (7 * lane)) & 127).astype(np.uint32), parity=True
                    )
                    scale = d * (np.float32(0.5) + (aux >> 28).astype(np.float32))
                    scale *= np.float32(0.5)
                    name = "iq3xxs_grid"
                else:
                    high = blocks[:, 66 + group]
                    indices = blocks[:, 2 + 2 * i : 2 + 2 * i + 2].astype(np.uint16)
                    indices |= np.stack(
                        [
                            ((high >> np.uint8(2 * lane + j)) & 1).astype(np.uint16)
                            << 8
                            for j in range(2)
                        ],
                        axis=1,
                    )
                    sign = _batch_signs(blocks[:, 74 + i])
                    nibble = (
                        blocks[:, 106 + group // 2] >> np.uint8(4 * (group % 2))
                    ) & 15
                    scale = d * (
                        np.float32(1) + np.float32(2) * nibble.astype(np.float32)
                    )
                    name = "iq3s_grid"
                values = _grid(name)[indices].reshape(len(blocks), 8)
                np.multiply(scale[:, None], values, out=decoded)
                np.multiply(decoded, sign, out=decoded)
            else:
                if qtype == _GGMLType.IQ1_S:
                    high = _batch_uint(blocks, 34 + 2 * group)
                    index = blocks[:, 2 + i].astype(np.uint16) | (
                        ((high >> (3 * lane)) & 7).astype(np.uint16) << 8
                    )
                    scale = d * (
                        np.float32(2) * ((high >> 12) & 7).astype(np.float32)
                        + np.float32(1)
                    )
                    delta = np.where(
                        high & 0x8000, np.float32(-0.125), np.float32(0.125)
                    )
                else:
                    high = blocks[:, 32 + i // 2] >> np.uint8(4 * (i % 2))
                    index = blocks[:, i].astype(np.uint16) | (
                        (high & 7).astype(np.uint16) << 8
                    )
                    scale_code = (
                        scales[:, group // 2] >> (6 * (group % 2) + 3 * (lane // 2))
                    ) & 7
                    scale = d * (
                        np.float32(2) * scale_code.astype(np.float32) + np.float32(1)
                    )
                    delta = np.where(high & 8, np.float32(-0.125), np.float32(0.125))
                values = _grid("iq1s_grid")[index] + delta[:, None]
                np.multiply(scale[:, None], values, out=decoded)


def _decode_blocks_chunk(blocks: np.ndarray, out: np.ndarray, qtype: int) -> None:  # noqa: PLR0911
    if qtype in (
        _GGMLType.IQ2_XXS,
        _GGMLType.IQ2_XS,
        _GGMLType.IQ3_XXS,
        _GGMLType.IQ1_S,
        _GGMLType.IQ3_S,
        _GGMLType.IQ2_S,
        _GGMLType.IQ1_M,
    ):
        _decode_iq_blocks(blocks, out, qtype)
        return
    if qtype in (
        _GGMLType.Q4_0,
        _GGMLType.Q4_1,
        _GGMLType.Q5_0,
        _GGMLType.Q5_1,
        _GGMLType.IQ4_NL,
    ):
        affine = qtype in (_GGMLType.Q4_1, _GGMLType.Q5_1)
        offset = 4 if affine else 2
        quant_offset = offset + (4 if qtype in (_GGMLType.Q5_0, _GGMLType.Q5_1) else 0)
        q = _batch_unpack(blocks[:, quant_offset:], 4, 16)
        if qtype in (_GGMLType.Q5_0, _GGMLType.Q5_1):
            high = _batch_uint(blocks, offset, 4)
            q |= (
                ((high[:, None] >> np.arange(32, dtype=np.uint32)[None, :]) & 1) << 4
            ).astype(np.int32)
        if qtype == _GGMLType.IQ4_NL:
            np.multiply(_batch_half(blocks)[:, None], _IQ4[q], out=out)
            return
        if not affine:
            q -= 16 if qtype == _GGMLType.Q5_0 else 8
        np.multiply(_batch_half(blocks)[:, None], q.astype(np.float32), out=out)
        if affine:
            np.add(out, _batch_half(blocks, 2)[:, None], out=out)
        return
    if qtype in (_GGMLType.Q8_0, _GGMLType.Q8_1, _GGMLType.Q8_K):
        offset = 2 if qtype == _GGMLType.Q8_0 else 4
        count = 256 if qtype == _GGMLType.Q8_K else 32
        if qtype == _GGMLType.Q8_K:
            d = (
                np.ascontiguousarray(blocks[:, :4])
                .view("<f4")
                .reshape(-1)
                .astype(np.float32)
            )
        else:
            d = _batch_half(blocks)
        np.multiply(
            blocks[:, offset : offset + count].view(np.int8).astype(np.float32),
            d[:, None],
            out=out,
        )
        return
    if qtype in (_GGMLType.Q2_K, _GGMLType.Q3_K):
        offset = 16 if qtype == _GGMLType.Q2_K else 32
        q = _batch_unpack(blocks[:, offset : offset + 64], 2, 32)
        if qtype == _GGMLType.Q2_K:
            scales = blocks[:, :16]
            d, m = _batch_half(blocks, 80), _batch_half(blocks, 82)
            group_scales = d[:, None] * (scales & 15).astype(np.float32)
            decoded = out.reshape(len(blocks), 16, 16)
            np.multiply(
                group_scales[:, :, None],
                q.reshape(len(blocks), 16, 16).astype(np.float32),
                out=decoded,
            )
            minima = m[:, None] * (scales >> 4).astype(np.float32)
            np.subtract(decoded, minima[:, :, None], out=decoded)
            return
        low = _batch_unpack(blocks[:, 96:104], 4, 8)
        high = _batch_unpack(blocks[:, 104:108], 2, 4)
        scales = low | (high << 4)
        scales -= 32
        q -= (1 - _batch_unpack(blocks[:, :32], 1, 32)) * 4
        group_scales = _batch_half(blocks, 108)[:, None] * scales.astype(np.float32)
        np.multiply(
            group_scales[:, :, None],
            q.reshape(len(blocks), 16, 16).astype(np.float32),
            out=out.reshape(len(blocks), 16, 16),
        )
        return
    if qtype in (_GGMLType.Q4_K, _GGMLType.Q5_K):
        scales = blocks[:, 4:16]
        sc = np.empty((len(blocks), 8), dtype=np.float32)
        mn = np.empty((len(blocks), 8), dtype=np.float32)
        sc[:, :4] = scales[:, :4] & 63
        sc[:, 4:] = (scales[:, 8:12] & 15) | ((scales[:, :4] >> 6) << 4)
        mn[:, :4] = scales[:, 4:8] & 63
        mn[:, 4:] = (scales[:, 8:12] >> 4) | ((scales[:, 4:8] >> 6) << 4)
        q = _batch_unpack(blocks[:, 16 if qtype == _GGMLType.Q4_K else 48 :], 4, 32)
        if qtype == _GGMLType.Q5_K:
            q |= _batch_unpack(blocks[:, 16:48], 1, 32) << 4
        decoded = out.reshape(len(blocks), 8, 32)
        group_scales = _batch_half(blocks)[:, None] * sc
        np.multiply(
            group_scales[:, :, None],
            q.reshape(len(blocks), 8, 32).astype(np.float32),
            out=decoded,
        )
        minima = _batch_half(blocks, 2)[:, None] * mn
        np.subtract(decoded, minima[:, :, None], out=decoded)
        return
    if qtype == _GGMLType.Q6_K:
        q = (
            _batch_unpack(blocks[:, :128], 4, 64)
            | (_batch_unpack(blocks[:, 128:192], 2, 32) << 4)
        ) - 32
        scales = blocks[:, 192:208].view(np.int8).astype(np.float32)
        group_scales = _batch_half(blocks, 208)[:, None] * scales
        np.multiply(
            group_scales[:, :, None],
            q.reshape(len(blocks), 16, 16).astype(np.float32),
            out=out.reshape(len(blocks), 16, 16),
        )
        return
    if qtype == _GGMLType.IQ4_XS:
        high = _batch_uint(blocks, 2)
        scales = np.stack(
            [
                ((blocks[:, 4 + j // 2] >> np.uint8(4 * (j % 2))) & 15).astype(np.int32)
                | (((high >> (2 * j)) & 3).astype(np.int32) << 4)
                for j in range(8)
            ],
            axis=1,
        )
        scales -= 32
        q = _batch_unpack(blocks[:, 8:], 4, 16)
        group_scales = _batch_half(blocks)[:, None] * scales.astype(np.float32)
        np.multiply(
            group_scales[:, :, None],
            _IQ4[q].reshape(len(blocks), 8, 32),
            out=out.reshape(len(blocks), 8, 32),
        )
        return
    if qtype == _GGMLType.TQ1_0:
        offset = 0
        for start, width, planes in ((0, 32, 5), (32, 16, 5), (48, 4, 4)):
            data = blocks[:, start : start + width].astype(np.uint32)
            for plane in range(planes):
                values = ((data * 3**plane) & 255) * 3 // 256
                out[:, offset : offset + width] = values.astype(np.float32) - 1
                offset += width
        np.multiply(out, _batch_half(blocks, 52)[:, None], out=out)
        return
    if qtype == _GGMLType.TQ2_0:
        q = _batch_unpack(blocks[:, :64], 2, 32)
        q -= 1
        np.multiply(_batch_half(blocks, 64)[:, None], q.astype(np.float32), out=out)
        return
    if qtype == _GGMLType.MXFP4:
        e = blocks[:, 0].astype(np.uint32)
        bits = np.where(
            e <= 1, np.uint32(0x00200000) << e, (e - 1) << np.uint32(23)
        ).astype("<u4")
        d = bits.view("<f4").astype(np.float32)
        q = _batch_unpack(blocks[:, 1:], 4, 16)
        np.multiply(d[:, None], _FP4[q], out=out)
        return
    if qtype == _GGMLType.NVFP4:
        e = blocks[:, :4]
        exponent = (e >> 3) & 15
        mantissa = e & 7
        d = np.where(
            (e == 0) | (e == _NVFP4_ZERO_SCALE_CODE),
            0.0,
            np.where(
                exponent == 0,
                np.ldexp(mantissa.astype(np.float64), -10),
                np.ldexp(
                    1.0 + mantissa.astype(np.float64) / 8.0,
                    exponent.astype(np.int16) - 8,
                ),
            ),
        ).astype(np.float32)
        q = _batch_unpack(blocks[:, 4:].reshape(-1, 8), 4, 8).reshape(
            len(blocks), 4, 16
        )
        np.multiply(
            d[:, :, None],
            _FP4[q],
            out=out.reshape(len(blocks), 4, 16),
        )
        return
    if qtype == _GGMLType.Q1_0:
        q = np.unpackbits(blocks[:, 2:], axis=1, bitorder="little")
        d = _batch_half(blocks)[:, None]
        np.copyto(out, np.where(q != 0, d, -d))
        return
    if qtype == _GGMLType.Q2_0:
        q = _batch_unpack(blocks[:, 2:], 2, 1)
        q -= 1
        np.multiply(_batch_half(blocks)[:, None], q.astype(np.float32), out=out)
        return
    raise ValueError(f"Unsupported quantized ggml_type: {qtype}")


def _decode_blocks(blocks: np.ndarray, out: np.ndarray, qtype: int) -> None:
    block_size = out.shape[1]
    chunk_size = max(1, 131072 // block_size)
    for start in range(0, len(blocks), chunk_size):
        end = start + chunk_size
        _decode_blocks_chunk(blocks[start:end], out[start:end], qtype)


class DequantizePacked(OpRun):
    def _run(self, x, format=None, shape=None, output_dtype=None):  # noqa: A002
        if not isinstance(format, str) or format not in GGUF_FORMATS:
            raise ValueError(f"Unsupported quantization format: {format}")
        qtype = GGUF_FORMATS[format]
        block_size, block_bytes = GGUF_BLOCK_SIZES[qtype]
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
        blocks = x.reshape(-1, block_bytes)
        _decode_blocks(blocks, out.reshape(-1, block_size), qtype)
        return (
            out.reshape(tuple(shape)).astype(
                helper.tensor_dtype_to_np_dtype(output_dtype), copy=False
            ),
        )
