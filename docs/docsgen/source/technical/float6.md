"<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

(onnx-detail-float6)=

# Float stored in 6 bits

## Papers

Based on OCP Microscaling Formats (MX) v1.0 spec. FP6 introduced for reduced precision in model inference and training.

As a result, two new types were introduced in `onnx==1.19.0` to support a limited set of operators.

- `FLOAT6E2M3`: 1 sign, 2 exp, 3 mant
- `FLOAT6E3M2`: 1 sign, 3 exp, 2 mant

## E2M3 and E3M2

[Tables for bit layouts, biases, special values, max/min, etc.]

## Cast

[Upcasting exact, downcasting with RNE and saturation tables similar to float8.]

## Packing and Unpacking

[Describe 6-bit packing into bytes, e.g., 4 bytes for 5 values with padding.]" 