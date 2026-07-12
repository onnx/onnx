"<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

(onnx-detail-float6)=

# Float stored in 6 bits

## Papers

Based on OCP Microscaling Formats (MX) v1.0 spec. FP6 introduced for reduced precision in model inference and training.

As a result, two new types were introduced in `onnx==1.23.0` to support a limited set of operators.

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
   * - Bits
     - sign:1 exp:2 mant:3
     - sign:1 exp:3 mant:2
   * - Exponent bias
     - 1
     - 3
   * - Infinities
     - No
     - No
   * - NaN
     - No
     - No
   * - Zeros
     - +/-0: 0x00 / 0x20
     - +/-0: 0x00 / 0x20
   * - Max normalized
     - 1.111 * 2^2 = 7.5
     - 1.11 * 2^4 = 28
   * - Min normalized
     - 1.000 * 2^0 = 1
     - 1.00 * 2^{-2} = 0.25
   * - Min denorm
     - 0.001 * 2^0 = 0.125
     - 0.01 * 2^{-2} = 0.0625
```

## Cast
Upcasting exact. Downcasting RNE with saturation. Examples:
```{eval-rst}
.. list-table:: Downcast examples
   :header-rows: 1

   * - float32
     - FLOAT6E2M3 (sat=true)
     - FLOAT6E3M2 (sat=true)
   * - 25.0
     - 7.5 (sat)
     - 24.0 (round)
   * - -0.0
     - -0.0
     - -0.0
   * - inf
     - 7.5 (sat)
     - 28.0 (sat)
   * - nan
     - -0.0 (unspecified; matches FLOAT4E2M1's cast behavior, not saturated)
     - -0.0 (unspecified; matches FLOAT4E2M1's cast behavior, not saturated)
```

## Packing and Unpacking
Pack 4 vals (24 bits) into 3 bytes, pad with 0s if not multiple of 4. Little-endian bit order.
