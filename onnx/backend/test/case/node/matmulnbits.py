# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import math

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect

def matmulnbits_unpack_zero_points(
    zero_points: np.ndarray,
    N: int, n_blocks_per_col: int,
    bits: int
) -> np.ndarray:
    unpacked_zp = []
    zp_bits_per_n = math.ceil(n_blocks_per_col * bits / 8)
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

def matmulnbits_unpack_B(
    B: np.ndarray,
    N: int,
    K: int,
    n_blocks_per_col: int,
    bits: int,
    block_size: int
) -> np.ndarray:
    total_bits = n_blocks_per_col * math.ceil((block_size * bits) / 8) * 8
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

def matmulnbits_dequantize_B(
    B: np.ndarray,
    scales: np.ndarray,
    zero_points: np.ndarray | None = None,
    **kwargs
) -> np.ndarray:
    if 'N' not in kwargs:
        raise ValueError('"N" attribute is required')
    if 'K' not in kwargs:
        raise ValueError('"K" attribute is required')

    N = kwargs['N']
    K = kwargs['K']
    bits = kwargs.get('bits', 4)
    block_size = kwargs.get('block_size', 128)

    zero_points = zero_points if zero_points is not None else np.full(scales.shape, (2 ** (bits - 1))).astype(scales.dtype)
    n_blocks_per_col = (K + block_size - 1) // block_size

    if zero_points.dtype != scales.dtype:
        zero_points = matmulnbits_unpack_zero_points(zero_points, N, n_blocks_per_col, bits).astype(scales.dtype)

    unpacked_X = matmulnbits_unpack_B(B, N, K, n_blocks_per_col, bits, block_size).astype(scales.dtype)

    Y = np.empty((N, K), dtype=scales.dtype)
    for n in range(N):
        for n_bpc in range(n_blocks_per_col):
            start = n_bpc * block_size
            end = min(start + block_size, K)
            zeropoint = zero_points[n * n_blocks_per_col + n_bpc]
            scale = scales[n * n_blocks_per_col + n_bpc]
            Y[n, start:end] = (unpacked_X[n, start:end] - zeropoint) * scale

    return Y

def matmulnbits_quantize_A_block_wise_no_zp(
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

def matmulnbits_dequantize_A_block_wise(
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

def matmulnbits_reference_implementation(
    #inputs
    A: np.ndarray,
    B: np.ndarray,
    scales: np.ndarray,
    # optional inputs
    zero_points: np.ndarray | None = None,
    bias: np.ndarray | None = None,
    **kwargs
) -> np.ndarray:
    # read in attributes and optional inputs if not provided use default values
    K = kwargs.get('K', A.shape[1])
    N = kwargs.get('N', B.shape[0])
    accuracy_level = kwargs.get('accuracy_level', 0)
    bits = kwargs.get('bits', 4)
    block_size = kwargs.get('block_size', 128)
    zero_points = zero_points if zero_points is not None else np.full(scales.shape, (2 ** (bits - 1))).astype(A.dtype)
    bias = bias if bias is not None else np.array(0).astype(A.dtype)

    if (B.ndim == 3):
        # reshape B from [N][n_blocks_per_col][blob_size] to [N][n_blocks_per_col * blob_size]
        B = B.reshape((B.shape[0], -1))
    dq_B = matmulnbits_dequantize_B(B, scales, zero_points, K = K, N = N, bits = bits, block_size = block_size)

    # accuracy_level defaults to 0 which means the type used for matmul type matches A.dtype
    accuracy_map = {
       0: A.dtype,
       1: np.float32,
       2: np.float16,
       3: np.float32, # numpy does not support bfloat16
       4: np.int8
    }
    matmul_type = accuracy_map.get(accuracy_level, A.dtype)
    # the shape from B input is {N,K} so it is transposed for MatMul operation
    if accuracy_level == 4:
        # quantize/dequantize A block wise. This simiulates quantization
        # of input A with with accuracy_level 4 by quantizing A to int8 losing the
        # accuracy of the original A
        q_A, A_scales = matmulnbits_quantize_A_block_wise_no_zp(A, block_size)
        dq_A = matmulnbits_dequantize_A_block_wise(q_A, block_size, A_scales)
        c = np.matmul(dq_A.astype(A.dtype), np.transpose(dq_B.astype(A.dtype)))
    else:
        c = np.matmul(A.astype(matmul_type), np.transpose(dq_B.astype(matmul_type)))
    Y = c.astype(A.dtype) + bias
    return Y

class MatMulNBits(Base):
  @staticmethod
  def export_matmulnbits_required_inputs_only() -> None:
    node = onnx.helper.make_node(op_type = "MatMulNBits",
                                 inputs = ['a', 'b', 'scales'],
                                 outputs = ['y'])
    a = np.array([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0], dtype=np.float32).reshape((2,4))
    b = np.array([0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,],
                  dtype=np.uint8).reshape((3,64))
    scales = np.array([1.0,2.0,3.0], dtype=np.float32)
    y = matmulnbits_reference_implementation(a, b, scales)
    expect(node, inputs=[a, b, scales], outputs=[y], name="test_matmulnbits_required_inputs_only")

  @staticmethod
  def export_matmulnbits_all_inputs() -> None:
    node = onnx.helper.make_node(op_type = "MatMulNBits",
                                 inputs = ['a', 'b', 'scales', 'zero_points', 'bias'],
                                 outputs = ['y'],
                                 accuracy_level = 0,
                                 K = 33,
                                 N = 3,
                                 bits = 4,
                                 block_size = 16)
    a = np.array([1.0,   2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                  17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
                  33.0,
                  -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0,
                  -16.0,-17.0, -18.0, -19.0, -20.0, -21.0, -22.0, -23.0, -24.0, -25.0, -26.0, -27.0, -28.0, -29.0,
                  -30.0, -31.0, -32.0, -33.0,], dtype=np.float32).reshape((2,33))
    b = np.array([0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,
                  0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,
                  0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,
                  0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,
                  0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,
                  0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,
                  0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,], dtype=np.uint8).reshape((3,3,8))
    scales = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=np.float32)
    zero_points = np.array([7.0, 8.0, 9.0, 7.0, 8.0, 9.0, 7.0, 8.0, 9.0], dtype=np.float32)
    bias = np.array([1.2, 3.4, 5.6], dtype=np.float32)
    y = matmulnbits_reference_implementation(a, b, scales, zero_points, bias,
                                             K=33, N=3, accuracy_level = 0, bits=4, block_size=16)
    expect(node, inputs=[a, b, scales, zero_points, bias], outputs=[y], name="test_matmulnbits_all_inputs")

  @staticmethod
  def export_matmulnbits_with_zero_points_f32() -> None:
    node = onnx.helper.make_node(op_type = "MatMulNBits",
                                 inputs = ['a', 'b', 'scales', 'zero_points'],
                                 outputs = ['y'],
                                 K = 4,
                                 N = 3,
                                 bits = 4,
                                 block_size = 16)
    a = np.array([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0], dtype=np.float32).reshape((2,4))
    b = np.array([0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00], dtype=np.uint8).reshape((3,8))
    scales = np.array([1.0,2.0,3.0], dtype=np.float32)
    zero_points = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    y = matmulnbits_reference_implementation(a, b, scales, zero_points, K=4, N=3, bits=4, block_size=16)
    expect(node, inputs=[a, b, scales, zero_points], outputs=[y], name="test_matmulnbits_with_zero_points_f32")

  @staticmethod
  def export_matmulnbits_with_zero_points_u8() -> None:
    node = onnx.helper.make_node(op_type = "MatMulNBits",
                                 inputs = ['a', 'b', 'scales', 'zero_points'],
                                 outputs = ['y'],
                                 K = 4,
                                 N = 3,
                                 bits = 4,
                                 block_size = 16)
    a = np.array([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0], dtype=np.float32).reshape((2,4))
    b = np.array([0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00], dtype=np.uint8).reshape((3,8))
    scales = np.array([1.0,2.0,3.0], dtype=np.float32)
    zero_points = np.array([0x07, 0x08, 0x09], dtype=np.uint8)
    y = matmulnbits_reference_implementation(a, b, scales, zero_points, K=4, N=3, bits=4, block_size=16)
    expect(node, inputs=[a, b, scales, zero_points], outputs=[y], name="test_matmulnbits_with_zero_points_u8")

  @staticmethod
  def export_matmulnbits_with_bias() -> None:
    node = onnx.helper.make_node(op_type = "MatMulNBits",
                                 inputs = ['a', 'b', 'scales', '', 'bias'],
                                 outputs = ['y'],
                                 K = 4,
                                 N = 3,
                                 bits = 4,
                                 block_size = 16)
    a = np.array([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0], dtype=np.float32).reshape((2,4))
    b = np.array([0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00], dtype=np.uint8).reshape((3,8))
    scales = np.array([1.0,2.0,3.0], dtype=np.float32)
    bias = np.array([1.2, 3.4, 5.6], dtype=np.float32)
    y = matmulnbits_reference_implementation(a, b, scales, zero_points=None, bias=bias, K=4, N=3, bits=4, block_size=16)
    expect(node, inputs=[a, b, scales, bias], outputs=[y], name="test_matmulnbits_with_bias")

  @staticmethod
  def export_matmulnbits_accuracy_level_1() -> None:
    node = onnx.helper.make_node(op_type = "MatMulNBits",
                                 inputs = ['a', 'b', 'scales'],
                                 outputs = ['y'],
                                 K = 4,
                                 N = 3,
                                 bits = 4,
                                 block_size = 16,
                                 accuracy_level = 1)
    a = np.array([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0], dtype=np.float32).reshape((2,4))
    b = np.array([0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00], dtype=np.uint8).reshape((3,8))
    scales = np.array([1.0,2.0,3.0], dtype=np.float32)
    y = matmulnbits_reference_implementation(a, b, scales, K=4, N=3, accuracy_level=1, bits=4, block_size=16)
    expect(node, inputs=[a, b, scales], outputs=[y], name="test_matmulnbits_accuracy_level_1")

  @staticmethod
  def export_matmulnbits_accuracy_level_2() -> None:
    node = onnx.helper.make_node(op_type = "MatMulNBits",
                                 inputs = ['a', 'b', 'scales'],
                                 outputs = ['y'],
                                 K = 4,
                                 N = 3,
                                 bits = 4,
                                 block_size = 16,
                                 accuracy_level = 2)
    a = np.array([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0], dtype=np.float32).reshape((2,4))
    b = np.array([0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00], dtype=np.uint8).reshape((3,8))
    scales = np.array([1.0,2.0,3.0], dtype=np.float32)
    y = matmulnbits_reference_implementation(a, b, scales, K=4, N=3, accuracy_level=2, bits=4, block_size=16)
    expect(node, inputs=[a, b, scales], outputs=[y], name="test_matmulnbits_accuracy_level_2")

  @staticmethod
  def export_matmulnbits_accuracy_level_3() -> None:
    node = onnx.helper.make_node(op_type = "MatMulNBits",
                                 inputs = ['a', 'b', 'scales'],
                                 outputs = ['y'],
                                 K = 4,
                                 N = 3,
                                 bits = 4,
                                 block_size = 16,
                                 accuracy_level = 3)
    a = np.array([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0], dtype=np.float32).reshape((2,4))
    b = np.array([0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00], dtype=np.uint8).reshape((3,8))
    scales = np.array([1.0,2.0,3.0], dtype=np.float32)
    y = matmulnbits_reference_implementation(a, b, scales, K=4, N=3, accuracy_level=3, bits=4, block_size=16)
    expect(node, inputs=[a, b, scales], outputs=[y], name="test_matmulnbits_accuracy_level_3")

  @staticmethod
  def export_matmulnbits_accuracy_level_4() -> None:
    node = onnx.helper.make_node(op_type = "MatMulNBits",
                                 inputs = ['a', 'b', 'scales'],
                                 outputs = ['y'],
                                 K = 4,
                                 N = 3,
                                 bits = 4,
                                 block_size = 16,
                                 accuracy_level = 4)
    a = np.array([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0], dtype=np.float32).reshape((2,4))
    b = np.array([0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x11,0x11,0x00,0x00,0x00,0x00,0x00,0x00], dtype=np.uint8).reshape((3,8))
    scales = np.array([1.0,2.0,3.0], dtype=np.float32)
    y = matmulnbits_reference_implementation(a, b, scales, K=4, N=3, accuracy_level=4, bits=4, block_size=16)
    expect(node, inputs=[a, b, scales], outputs=[y], name="test_matmulnbits_accuracy_level_4")

  @staticmethod
  def export_matmulnbits_2bit() -> None:
    node = onnx.helper.make_node(op_type = "MatMulNBits",
                                 inputs = ['a', 'b', 'scales', 'zero_points', 'bias'],
                                 outputs = ['y'],
                                 accuracy_level = 0,
                                 K = 33,
                                 N = 3,
                                 bits = 2,
                                 block_size = 16)
    a = np.array([1.0,   2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                  17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
                  33.0,
                  -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0,
                  -16.0,-17.0, -18.0, -19.0, -20.0, -21.0, -22.0, -23.0, -24.0, -25.0, -26.0, -27.0, -28.0, -29.0,
                  -30.0, -31.0, -32.0, -33.0,], dtype=np.float32).reshape((2,33))
                #  4    8    12   16   20   24   28   32  36   40   44   48 
    b = np.array([0x55,0x55,0x55,0x55,0x55,0x55,0x55,0x55,0x01,0x00,0x00,0x00,
                  0x55,0x55,0x55,0x55,0x55,0x55,0x55,0x55,0x01,0x00,0x00,0x00,
                  0x55,0x55,0x55,0x55,0x55,0x55,0x55,0x55,0x01,0x00,0x00,0x00,], dtype=np.uint8).reshape((3,3,4))
    scales = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    zero_points = np.array([0x00,0x00,0x00], dtype=np.uint8)
    bias = np.array([0, 0, 0], dtype=np.float32)
    y = matmulnbits_reference_implementation(a, b, scales, zero_points, bias,
                                             K=33, N=3, accuracy_level=0, bits=2, block_size=16)
    expect(node, inputs=[a, b, scales, zero_points, bias], outputs=[y], name="test_matmulnbits_2bit")

  @staticmethod
  def export_matmulnbits_3bit() -> None:
    node = onnx.helper.make_node(op_type = "MatMulNBits",
                                 inputs = ['a', 'b', 'scales', 'zero_points', 'bias'],
                                 outputs = ['y'],
                                 K = 33,
                                 N = 3,
                                 bits = 3,
                                 block_size = 16)
    a = np.array([1.0,   2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                  17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
                  33.0,
                  -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0,
                  -16.0,-17.0, -18.0, -19.0, -20.0, -21.0, -22.0, -23.0, -24.0, -25.0, -26.0, -27.0, -28.0, -29.0,
                  -30.0, -31.0, -32.0, -33.0,], dtype=np.float32).reshape((2,33))
                #            8              16             24             32
    b = np.array([0x49,0x92,0x24,0x49,0x92,0x24,0x49,0x92,0x24,0x49,0x92,0x24,0x01,0x00,0x00,0x00,0x00,0x00,
                  0x49,0x92,0x24,0x49,0x92,0x24,0x49,0x92,0x24,0x49,0x92,0x24,0x01,0x00,0x00,0x00,0x00,0x00,
                  0x49,0x92,0x24,0x49,0x92,0x24,0x49,0x92,0x24,0x49,0x92,0x24,0x01,0x00,0x00,0x00,0x00,0x00,],
                  dtype=np.uint8).reshape((3,3,6))
    scales = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    zero_points = np.array([0x00,0x00,0x00,0x00,0x00,0x00], dtype=np.uint8)
    bias = np.array([0, 0, 0], dtype=np.float32)
    y = matmulnbits_reference_implementation(a, b, scales, zero_points, bias, K=33, N=3, bits=3, block_size=16)
    expect(node, inputs=[a, b, scales, zero_points, bias], outputs=[y], name="test_matmulnbits_3bit")

  @staticmethod
  def export_matmulnbits_5bit() -> None:
    node = onnx.helper.make_node(op_type = "MatMulNBits",
                                 inputs = ['a', 'b', 'scales', 'zero_points', 'bias'],
                                 outputs = ['y'],
                                 K = 20,
                                 N = 3,
                                 bits = 5,
                                 block_size = 16)
    a = np.array([1.0,   2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                  17.0, 18.0, 19.0, 20.0,
                  -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0,
                  -16.0,-17.0, -18.0, -19.0, -20.0,], dtype=np.float32).reshape((2,20))
                #                       8                       16                       24
    b = np.array([0x21,0x84,0x10,0x42,0x08,0x21,0x84,0x10,0x42,0x08,0x21,0x84,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x21,0x84,0x10,0x42,0x08,0x21,0x84,0x10,0x42,0x08,0x21,0x84,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x21,0x84,0x10,0x42,0x08,0x21,0x84,0x10,0x42,0x08,0x21,0x84,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,],
                 dtype=np.uint8).reshape((3,2,10))
    scales = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    zero_points = np.array([0x00,0x00,0x00,0x00,0x00,0x00], dtype=np.uint8)
    bias = np.array([0, 0, 0], dtype=np.float32)
    y = matmulnbits_reference_implementation(a, b, scales, zero_points, bias, K=20, N=3, bits=5, block_size=16)
    expect(node, inputs=[a, b, scales, zero_points, bias], outputs=[y], name="test_matmulnbits_5bit")

  @staticmethod
  def export_matmulnbits_6bit() -> None:
    node = onnx.helper.make_node(op_type = "MatMulNBits",
                                 inputs = ['a', 'b', 'scales', 'zero_points', 'bias'],
                                 outputs = ['y'],
                                 K = 20,
                                 N = 3,
                                 bits = 6,
                                 block_size = 16)
    a = np.array([1.0,   2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                  17.0, 18.0, 19.0, 20.0,
                  -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0,
                  -16.0,-17.0, -18.0, -19.0, -20.0,], dtype=np.float32).reshape((2,20))
                #            4              8              12            16              20
    b = np.array([0x41,0x10,0x04,0x41,0x10,0x04,0x41,0x10,0x04,0x41,0x10,0x04,0x41,0x10,0x04,
                  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x41,0x10,0x04,0x41,0x10,0x04,0x41,0x10,0x04,0x41,0x10,0x04,0x41,0x10,0x04,
                  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                  0x41,0x10,0x04,0x41,0x10,0x04,0x41,0x10,0x04,0x41,0x10,0x04,0x41,0x10,0x04,
                  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,],
                  dtype=np.uint8).reshape((3,2,12))
    scales = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    zero_points = np.array([0x00,0x00,0x00,0x00,0x00,0x00], dtype=np.uint8)
    bias = np.array([0, 0, 0], dtype=np.float32)
    y = matmulnbits_reference_implementation(a, b, scales, zero_points, bias, K=20, N=3, bits=6, block_size=16)
    expect(node, inputs=[a, b, scales, zero_points, bias], outputs=[y], name="test_matmulnbits_6bit")

  @staticmethod
  def export_matmulnbits_7bit() -> None:
    node = onnx.helper.make_node(op_type = "MatMulNBits",
                                 inputs = ['a', 'b', 'scales', 'zero_points', 'bias'],
                                 outputs = ['y'],
                                 K = 16,
                                 N = 3,
                                 bits = 7,
                                 block_size = 16)
    a = np.array([1.0,   2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                  -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0,
                  -16.0,], dtype=np.float32).reshape((2,16))
                #                                 8                                  16
    b = np.array([0x81,0x40,0x20,0x10,0x08,0x04,0x02,0x81,0x40,0x20,0x10,0x08,0x04,0x02,
                  0x81,0x40,0x20,0x10,0x08,0x04,0x02,0x81,0x40,0x20,0x10,0x08,0x04,0x02,
                  0x81,0x40,0x20,0x10,0x08,0x04,0x02,0x81,0x40,0x20,0x10,0x08,0x04,0x02,],
                 dtype=np.uint8).reshape((3,14))
    scales = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    zero_points = np.array([0x00,0x00,0x00], dtype=np.uint8)
    bias = np.array([0, 0, 0], dtype=np.float32)
    y = matmulnbits_reference_implementation(a, b, scales, zero_points, bias, K=16, N=3, bits=7, block_size=16)
    expect(node, inputs=[a, b, scales, zero_points, bias], outputs=[y], name="test_matmulnbits_7bit")
