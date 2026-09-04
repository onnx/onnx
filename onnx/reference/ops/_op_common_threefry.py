# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

_UINT32_MASK = (1 << 32) - 1
_KEY_PARITY = 0x1BD11BDA
_ROTATIONS = ((13, 15, 26, 6), (17, 29, 16, 24))


def _validate_state(state: np.ndarray) -> tuple[int, int]:
    state = np.asarray(state)
    if state.dtype != np.int64 or state.shape != (2,):
        raise ValueError(
            f"A Threefry2x32 PRNG state must be int64 with shape (2,), got "
            f"{state.dtype} with shape {state.shape}."
        )
    if np.any(state < 0) or np.any(state > _UINT32_MASK):
        raise ValueError("Each PRNG state word must be in the range [0, 2**32).")
    return int(state[0]), int(state[1])


def threefry_seed(seed: np.ndarray) -> np.ndarray:
    seed = np.asarray(seed)
    if seed.dtype != np.int64 or seed.shape != ():
        raise ValueError(
            f"A Threefry2x32 seed must be a scalar int64, got {seed.dtype} "
            f"with shape {seed.shape}."
        )
    seed_bits = int(seed) & ((1 << 64) - 1)
    return np.array([seed_bits >> 32, seed_bits & _UINT32_MASK], dtype=np.int64)


def _rotate_left(value: np.ndarray, distance: int) -> np.ndarray:
    return ((value << distance) | (value >> (32 - distance))) & _UINT32_MASK


def _threefry2x32_words(
    key0: int, key1: int, count0: np.ndarray, count1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    keys = (key0, key1, key0 ^ key1 ^ _KEY_PARITY)
    value0 = (np.asarray(count0, dtype=np.int64) + keys[0]) & _UINT32_MASK
    value1 = (np.asarray(count1, dtype=np.int64) + keys[1]) & _UINT32_MASK

    for injection in range(5):
        for rotation in _ROTATIONS[injection % 2]:
            value0 = (value0 + value1) & _UINT32_MASK
            value1 = _rotate_left(value1, rotation) ^ value0
        value0 = (value0 + keys[(injection + 1) % 3]) & _UINT32_MASK
        value1 = (value1 + keys[(injection + 2) % 3] + injection + 1) & _UINT32_MASK

    return value0, value1


def threefry_fold_in(state: np.ndarray, data: int) -> np.ndarray:
    key0, key1 = _validate_state(state)
    data_bits = int(data) & ((1 << 64) - 1)
    output0, output1 = _threefry2x32_words(
        key0,
        key1,
        np.asarray(data_bits >> 32, dtype=np.int64),
        np.asarray(data_bits & _UINT32_MASK, dtype=np.int64),
    )
    return np.array([int(output0), int(output1)], dtype=np.int64)


def threefry_split(state: np.ndarray, count: int) -> tuple[np.ndarray, ...]:
    key0, key1 = _validate_state(state)
    counters = np.arange(count, dtype=np.int64)
    output0, output1 = _threefry2x32_words(
        key0, key1, counters >> 32, counters & _UINT32_MASK
    )
    return tuple(
        np.array([output0[i], output1[i]], dtype=np.int64) for i in range(count)
    )
