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
   * - Bits
     - sign:1 exp:2 mant:3
     - sign:1 exp:3 mant:2
   * - Exponent bias
     - 3
     - 4
   * - Infinities
     - No
     - No
   * - NaN
     - No
     - No
   * - Zeros
     - +/-0: 0x00 / 0x20 (but -0 saturates to 0)
     - +/-0: 0x00 / 0x20
   * - Max normalized
     - 1.111 * 2^0 = 15 (but effective max 24 with denorm? Confirm spec)
     - 1.11 * 2^3 = 48
   * - Min normalized
     - 0.001 * 2^{-3} = 0.03125
     - 0.01 * 2^{-4} = 0.00390625
   * - Min denorm
     - 0.001 * 2^{-3} = 0.03125 (spec)
     - 0.01 * 2^{-4}
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
     - 24.0 (sat)
     - 25.0 (round)
   * - -0.0
     - 0.0
     - 0.0
   * - inf
     - 24.0 (sat)
     - 48.0 (sat)
   * - nan
     - 24.0 (sat)
     - 48.0 (sat)
```

## Packing and Unpacking
Pack 4 vals (24 bits) into 3 bytes, pad with 0s if not multiple of 4. Little-endian bit order. 