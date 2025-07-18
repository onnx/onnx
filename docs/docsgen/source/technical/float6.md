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

```{eval-rst}
.. list-table:: Float6 types
   :widths: 10 10 10
   :header-rows: 1

   * -
     - FLOAT6E2M3
     - FLOAT6E3M2
   * - Exponent bias
     - 3
     - 4
   * - Infinities
     - no
     - no
   * - NaN
     - no
     - no
   * - Zeros
     - S.00.000
     - S.000.00
   * - Max
     - S.11.111 = 24.0
     - S.111.11 = 48.0
   * - Min
     - S.00.001 = 0.125
     - S.000.01 = 0.0625
```

## Cast

Upcasting from FP6 to float16, bfloat16, float32 is exact.

Downcasting to FP6 uses RNE rounding with saturation:

| x | FLOAT6E2M3 | FLOAT6E3M2 |
| - | - | - |
| within range | RNE | RNE |
| out of range | saturate to max/min | saturate to max/min |
| Inf/-Inf | saturate to max/min | saturate to max/min |
| NaN | saturate to max | saturate to max |

## Packing and Unpacking

6-bit values are packed into uint8 arrays. For efficiency, pack 4 values (24 bits) into 3 bytes, with padding (0s) if the number of elements isn't multiple of 4. Unpacking reverses this. 