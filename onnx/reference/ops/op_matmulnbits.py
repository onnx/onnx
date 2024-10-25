# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
from math import ceil

from onnx.reference.op_run import OpRun

def matmulnbits_unpack_zero_points(
    zero_points: np.ndarray,
    N: int, n_blocks_per_col: int,
    bits: int
) -> np.ndarray:
    unpacked_zp = []
    zp_bits_per_n = ceil(n_blocks_per_col * bits / 8)
    total_zp_bits = len(zero_points) * 8
    mask = (1 << bits) - 1

    for n in range(N):
        unpacked_row_buf = []
        current_bit_pos = 0
        while current_bit_pos < total_zp_bits and len(unpacked_row_buf) < n_blocks_per_col:
            byte_pos = (current_bit_pos // 8) + (n * zp_bits_per_n)
            bit_offset = current_bit_pos % 8

            bits_available = 8 - bit_offset
            if bits_available >= bits:
                value = (zero_points[byte_pos] >> bit_offset) & mask
            else:
                lower_bits = (zero_points[byte_pos] >> bit_offset) & mask
                upper_bits = (zero_points[byte_pos + 1] << bits_available) & mask
                value = upper_bits | lower_bits
            unpacked_row_buf.append(value)
            current_bit_pos += bits
        unpacked_zp.extend(unpacked_row_buf)

    return np.array(unpacked_zp, dtype=np.uint8)

def matmulnbits_unpack_b(
    B: np.ndarray,
    N: int,
    K: int,
    n_blocks_per_col: int,
    bits: int,
    block_size: int
) -> np.ndarray:
    total_bits = n_blocks_per_col * ceil((block_size * bits) / 8) * 8
    mask = (1 << bits) - 1
    unpacked_B = np.empty((N, K), dtype=np.uint8)
    for n in range(N):
        unpacked_row_buf = []
        current_bit_pos = 0
        for _n_bpc in range(n_blocks_per_col):
            unpacked_col_buf = []
            while len(unpacked_col_buf) < block_size and (len(unpacked_row_buf) + len(unpacked_col_buf)) < K and current_bit_pos < total_bits:
                byte_pos = (current_bit_pos // 8)
                bit_offset = current_bit_pos % 8

                bits_available = 8 - bit_offset
                if bits_available >= bits:
                    value = (B[n][byte_pos] >> bit_offset) & mask
                else:
                    lower_bits = (B[n][byte_pos] >> bit_offset) & mask
                    upper_bits = (B[n][byte_pos + 1] << bits_available) & mask
                    value = upper_bits | lower_bits
                unpacked_col_buf.append(value)
                current_bit_pos += bits
            unpacked_row_buf.extend(unpacked_col_buf)
        unpacked_B[n] = unpacked_row_buf

    return unpacked_B

def matmulnbits_dequantize_b(
    B: np.ndarray,
    scales: np.ndarray,
    zero_points: np.ndarray,
    N: int,
    K: int,
    bits: int,
    block_size: int
) -> np.ndarray:
    zero_points = zero_points if zero_points is not None else np.full(scales.shape, (2 ** (bits - 1))).astype(scales.dtype)
    n_blocks_per_col = (K + block_size - 1) // block_size

    if zero_points.dtype != scales.dtype:
        zero_points = matmulnbits_unpack_zero_points(zero_points, N, n_blocks_per_col, bits).astype(scales.dtype)

    unpacked_X = matmulnbits_unpack_b(B, N, K, n_blocks_per_col, bits, block_size).astype(scales.dtype)

    dq_B = np.empty((N, K), dtype=scales.dtype)
    for n in range(N):
        for n_bpc in range(n_blocks_per_col):
            start = n_bpc * block_size
            end = min(start + block_size, K)
            zeropoint = zero_points[n * n_blocks_per_col + n_bpc]
            scale = scales[n * n_blocks_per_col + n_bpc]
            dq_B[n, start:end] = (unpacked_X[n, start:end] - zeropoint) * scale

    return dq_B

def matmulnbits_quantize_a_block_wise_no_zp(
    X: np.ndarray,
    block_size: int
) -> tuple[np.ndarray, np.ndarray]:
    M = X.shape[0]
    K = X.shape[1]
    num_blocks = (K + block_size - 1) // block_size
    quantized_X = np.zeros_like(X, dtype=np.int8)
    scales = np.zeros(M * num_blocks)
    range_max = 127.0 #( 1 << 7) -1
    for m in range(M):
        for i in range(num_blocks):
            start = i * block_size
            end = min(start + block_size, K)
            block = X[m, start:end]
            block_max = np.max(np.abs(block))
            scale = block_max / range_max
            normalized_block = block / scale
            quantized_block = np.round(normalized_block).astype(np.uint8)
            quantized_X[m, start:end] = quantized_block
            scales[m * num_blocks + i] = scale
    return quantized_X, scales

def matmulnbits_dequantize_a_block_wise(
    quantized_A: np.ndarray,
    block_size: int,
    scales: np.ndarray,
    zero_points: np.ndarray | None = None
) -> np.ndarray:
    M, K = quantized_A.shape
    num_blocks = (K + block_size - 1) // block_size
    dequantized_X = np.zeros_like(quantized_A, dtype=np.float32)

    for m in range(M):
        for i in range(num_blocks):
            start = i * block_size
            end = min(start + block_size, K)
            quantized_block = quantized_A[m, start:end]
            scale = scales[m * num_blocks + i]
            zero_point = zero_points[m * num_blocks + i] if zero_points is not None else 0
            dequantized_block = (quantized_block - zero_point) * scale
            dequantized_X[m, start:end] = dequantized_block

    return dequantized_X

class MatMulNBits(OpRun):
    def _run(
        self,
        A: np.ndarray,
        B: np.ndarray,
        scales: np.ndarray,
        zero_points: np.ndarray | None = None,
        bias: np.ndarray| None = None,
        K: int | None = None,
        N: int | None = None,
        accuracy_level: int | None = None,
        bits: int | None = None,
        block_size: int | None = None) -> tuple[np.ndarray,]:
        # validate ndim of required inputs
        if A.ndim != 2:
            raise ValueError("Input A must be a 2-dimensional tensor of shape.")
        if B.ndim < 2 or B.ndim > 3:
            raise ValueError("Input B must be a 2-dimensional or 3-dimensional tensor.")
        if scales.ndim != 1:
            raise ValueError("Scales must be a 1-dimensional tensor.")
        if zero_points is not None and zero_points.ndim != 1:
            raise ValueError("Zero points must be a 1-dimensional tensor.")
        if bias is not None and bias.ndim != 1:
            raise ValueError("Bias must be a 1-dimensional tensor.")
        # Check attributes and set defaults if not provided
        if K is None:
            K = A.shape[1]
        if N is None:
            N = B.shape[0]
        if accuracy_level is None:
            accuracy_level = 0
        if bits is None:
            bits = 4
        if block_size is None:
            block_size = 128

        if accuracy_level < 0 or accuracy_level > 4:
            raise ValueError("accuracy_level must be between 0 and 4.")
        if block_size < 16 or (block_size & (block_size - 1)) != 0:
            raise ValueError("block_size must be a power of 2 and not smaller than 16.")

        # validate inputs shapes are valid
        if A.shape[1] != K:
            raise ValueError("K must be equal to the number of columns in A.")
        if B.shape[0] != N:
            raise ValueError("N must be equal to the number of rows in B.")
        n_blocks_per_col = (K + block_size - 1) // block_size
        blob_size = ceil((block_size * bits)/8)
        b_shape_error = ("B must have the shape [N][n_blocks_per_col][blob_size] or [N][n_blocks_per_col * blob_size]. "
                        "Where n_blocks_per_col = (K + block_size - 1) / block_size and "
                        "blob_size = CeilDiv((block_size * bits),8).")
        if B.ndim == 2:
            if B.shape[1] != (n_blocks_per_col * blob_size):
                raise ValueError(b_shape_error)
        if B.ndim == 3:
            if B.shape[1] != n_blocks_per_col or B.shape[2] != blob_size:
                raise ValueError(b_shape_error)
        if scales.shape[0] != N * n_blocks_per_col:
            raise ValueError("Scales must have the shape [N * n_blocks_per_col]. "
                             "Where n_blocks_per_col = (K + block_size - 1) / block_size.")
        if zero_points is None:
            zero_points = np.full(scales.shape, 2**(bits-1), dtype=A.dtype)
        zero_points_shape_error = ("Zero points must have the shape [N * n_blocks_per_col] if the data type is the "
                                   "same as A input. If the data type is uint8, then zero points must have the shape "
                                   "[N * CeilDiv((block_size * bits),8)].")
        # zero_points will had a different shape depending if it is uint8 or float
        if zero_points.dtype == B.dtype:
            if zero_points.shape[0] != N * ceil(n_blocks_per_col * bits / 8):
                raise ValueError(zero_points_shape_error)
        elif zero_points.shape[0] != N * n_blocks_per_col:
            raise ValueError(zero_points_shape_error)
        if bias is None:
            bias = np.zeros(N, dtype=A.dtype)
        if bias.shape[0] != N:
            raise ValueError("Bias must have the shape [N].")

        if B.ndim == 3:
            # reshape B from [N][n_blocks_per_col][blob_size] to [N][n_blocks_per_col * blob_size]
            # all the functions assume B is in this shape
            B = B.reshape((B.shape[0], -1))
        dq_B = matmulnbits_dequantize_b(B, scales, zero_points, N, K, bits, block_size)

        accuracy_map = {
            0: A.dtype,
            1: np.float32,
            2: np.float16,
            3: np.float32, # numpy does not support bfloat16
            4: np.int8
        }
        matmul_type = accuracy_map.get(accuracy_level, A.dtype)
        if accuracy_level == 4:
            # quantize/dequantize A block wise. This simiulates quantization
            # of input A with with accuracy_level 4 by quantizing A to int8 losing the
            # accuracy of the original A
            q_A, A_scales = matmulnbits_quantize_a_block_wise_no_zp(A, block_size)
            dq_A = matmulnbits_dequantize_a_block_wise(q_A, block_size, A_scales)
            c = np.matmul(dq_A.astype(A.dtype), np.transpose(dq_B.astype(A.dtype)))
        else:
            c = np.matmul(A.astype(matmul_type), np.transpose(dq_B.astype(matmul_type)))
        Y = c.astype(A.dtype) + bias
        return (Y,)
