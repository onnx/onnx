<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->
(onnx-detail-int2) =

# 2 bit integer types

## Papers

[T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge](https://arxiv.org/abs/2407.00088)

T-MAC, an innovative lookup table(LUT)-based method designed for efficient low-bit LLM (i.e., weight-quantized LLM) inference on CPUs. T-MAC directly supports mpGEMM without dequantization, while simultaneously eliminating multiplications and reducing additions required. Specifically, T-MAC transforms the traditional data-type-centric multiplication to bit-wise table lookup, and enables a unified and scalable mpGEMM solution.

## Cast

Cast from 2 bit to any higher precision type is exact.
Cast to a 2 bit type is done by rounding to the nearest-integer (with ties to even)
nearest-even integer and truncating.


## Packing and Unpacking (2-bit)
All 2-bit types are stored as 4×2-bit values in a single byte. The elements are packed from least significant bits (LSB) to most significant bits (MSB). That is, for consecutive elements x0, x1, x2, x3 in the array:

Packing:
```
pack(x0, x1, x2, x3):
    (x0 & 0x03) |
    ((x1 & 0x03) << 2) |
    ((x2 & 0x03) << 4) |
    ((x3 & 0x03) << 6)
```

Unpacking:
```
x0 = z & 0x03
x1 = (z >> 2) & 0x03
x2 = (z >> 4) & 0x03
x3 = (z >> 6) & 0x03
```
In case the total number of elements is not divisible by 4, zero-padding will be applied in the remaining higher bits of the final byte.
The storage size of a 2-bit tensor of size N is: ceil(N / 4) bytes
