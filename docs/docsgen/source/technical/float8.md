<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

(onnx-detail-float8)=

# Float stored in 8 bits

## Papers

Two papers have been published in 2022 to introduce floats
stored on a byte as opposed to float 32 stored on 4 bytes.
The float precision is much lower but the training accuracy
does not suffer too much.

[FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
from NVIDIA, Intel and ARM introduces two types following
[IEEE specifciations](https://en.wikipedia.org/wiki/IEEE_754).
First one is E4M3, 1 bit for the sign, 4 bits for the exponents and 3
bits for the mantissa. Second one is E5M2, 1 bit for the sign,
5 bits for the exponents and 2 for the mantissa. The first types
is mostly used for the weights, the second one for the gradient.

Second paper [8-bit Numerical Formats For Deep Neural Networks](https://arxiv.org/pdf/2206.02915.pdf) introduces
similar types. IEEE standard gives the same value
to `+0` (or integer 0) and `-0` (or integer 128).
They chose to give distinct float values to these two
numbers. The paper experiments different split between
exponent and mantissa and shows and E4M3 and E5M2 are
the best ones.

As a result, four new types were introduced in `onnx==1.15.0`
to support a limited set of operators to enable computation
with float 8.

- `E4M3FN`: 1 bit for the sign, 4 bits for the exponents, 3 bits for the mantissa,
  only nan values and no infinite values (FN),
- `E4M3FNUZ`: 1 bit for the sign, 4 bits for the exponents, 3 bits for the mantissa,
  only nan values and no infinite values (FN), no negative zero (UZ)
- `E5M2`: 1 bit for the sign, 5 bits for the exponents, 2 bits for the mantissa,
- `E5M2FNUZ`: 1 bit for the sign, 5 bits for the exponents, 2 bits for the mantissa,
  only nan values and no infinite values (FN), no negative zero (UZ)

The implementation is usually hardware dependant.
NVIDIA, Intel and Arm implement `E4M3FN` and `E5M2` is its latest graphical processor.
GraphCore does the same only with `E4M3FNUZ` and `E5M2FNUZ`.

## E4M3FN and E5M2

$S$ stands for the sign. $10_2$ describe a number base 2.

```{eval-rst}
.. list-table:: Float8 types
   :widths: 10 10 10
   :header-rows: 1

   * -
     - E4M3FN
     - E5M2
   * - Exponent bias
     - 7
     - 15
   * - Infinities
     -
     - :math:`S.11111.00_2`
   * - NaN
     - :math:`S.1111.111_2`
     - :math:`S.11111.\{01, 10, 11\}_2`
   * - Zeros
     - :math:`S.0000.000_2`
     - :math:`S.00000.00_2`
   * - Max
     - :math:`S.1111.110_2`
     - :math:`1.75 \times 2^{15}= 57344`
   * - Min
     - :math:`S.0000.001_2 = 2^{-9}`
     - :math:`S.00000.01_2 = 2^{-16}`

```

Let's denote the bit representation as $S.b_6 b_5 b_4 b_3 b_2 b_1 b_0$.
The float value is defined by the following expressions:

```{eval-rst}
.. list-table:: Float8 types values
   :widths: 10 10 10
   :header-rows: 1

   * -
     - E4M3FN
     - E5M2
   * - exponent :math:`\neq` 0
     - :math:`(-1)^S 2^{\sum_{i=3}^6 b_i 2^{i-3} - 7} \left( 1 + \sum_{i=0}^2 b_i 2^{i-3} \right)`
     - :math:`(-1)^S 2^{\sum_{i=2}^6 b_i 2^{i-2} - 15} \left( 1 + \sum_{i=0}^1 b_i 2^{i-2} \right)`
   * - exponent :math:`=` 0
     - :math:`(-1)^S 2^{-6} \sum_{i=0}^2 b_i 2^{i-3}`
     - :math:`(-1)^S 2^{-14} \sum_{i=0}^1 b_i 2^{i-2}`
```

## E4M3FNUZ and E5M2FNUZ

The previous types support positive and negative zero, positive and negative nan.
Another type definition was introduced by GraphCore to make a better use
of these four values. Every type including UZ in its name have only one zero
and one nan (= negative zero). The other difference comes from the exponent bias.
As a result, a float 8 *FLOAT8E4M3FN*, not null, not nan, cannot be simply
converted into *FLOAT8E4M3FNUZ* due to this exponent bias difference.
Even if the mantissa is the same, the exponent is not.

```{eval-rst}
.. list-table:: Float8 types
   :widths: 10 10 10
   :header-rows: 1

   * -
     - E4M3FNUZ
     - E5M2FNUZ
   * - Exponent bias
     - 8
     - 16
   * - Infinities
     -
     -
   * - NaN
     - :math:`1.0000.000_2`
     - :math:`1.00000.00_2`
   * - Zeros
     - :math:`0.0000.000_2`
     - :math:`0.00000.00_2`
   * - Max
     - :math:`S.1111.111_2`
     - :math:`S.11111.11_2`
   * - Min
     - :math:`S.0000.001_2 = 2^{-10}`
     - :math:`S.00000.01_2 = 2^{-17}`
```

The float value is defined by the following expressions:

```{eval-rst}
.. list-table:: Float8 types values
   :widths: 10 10 10
   :header-rows: 1

   * -
     - E4M3FNUZ
     - E5M2FNUZ
   * - exponent :math:`\neq` 0
     - :math:`(-1)^S 2^{\sum_{i=3}^6 b_i 2^{i-3} - 8} \left( 1 + \sum_{i=0}^2 b_i 2^{i-3} \right)`
     - :math:`(-1)^S 2^{\sum_{i=2}^6 b_i 2^{i-2} - 16} \left( 1 + \sum_{i=0}^1 b_i 2^{i-2} \right)`
   * - exponent :math:`=` 0
     - :math:`(-1)^S 2^{-7} \sum_{i=0}^2 b_i 2^{i-3}`
     - :math:`(-1)^S 2^{-15} \sum_{i=0}^1 b_i 2^{i-2}`
```

## Cast

Cast from float 8 to
[float 16](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) (or E5M10),
[bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) (or E8M7),
[float32](https://en.wikipedia.org/wiki/Single-precision_floating-point_format) (or E8M23) is easier.
The cast is exact. The conversion does not necessarily preserve the sign for
specific values such as `-0` or `-NaN`.

Cast to float 8 consists in finding the closest float 8
to the original float 32 value. It is usually done by shifting
and truncating.

The conversion may with saturation, every value out of range
becomes the highest available value. Next table summarizes
all the case. `[x]` means the value rounded to
the target mantissa width.

| x                 | E4M3FN   | E4M3FNUZ | E5M2     | E5M2FNUZ |
| ----------------- | -------- | -------- | -------- | -------- |
| 0                 | 0        | 0        | 0        | 0        |
| -0                | -0       | 0        | -0       | 0        |
| NaN               | NaN      | NaN      | NaN      | NaN      |
| Inf               | FLT_MAX  | NaN      | FLT_MAX  | NaN      |
| -Inf              | -FLT_MAX | NaN      | -FLT_MAX | NaN      |
| \[x\] > FLT_MAX   | FLT_MAX  | FLT_MAX  | FLT_MAX  | FLT_MAX  |
| \[x\] \< -FLT_MAX | -FLT_MAX | -FLT_MAX | -FLT_MAX | -FLT_MAX |
| else              | RNE      | RNE      | RNE      | RNE      |

The conversion may also be defined without any saturation.

| x                 | E4M3FN | E4M3FNUZ | E5M2 | E5M2FNUZ |
| ----------------- | ------ | -------- | ---- | -------- |
| 0                 | 0      | 0        | 0    | 0        |
| -0                | -0     | 0        | -0   | 0        |
| NaN               | NaN    | NaN      | NaN  | NaN      |
| -NaN              | -NaN   | NaN      | -NaN | NaN      |
| Inf               | NaN    | NaN      | Inf  | NaN      |
| -Inf              | -NaN   | NaN      | -Inf | NaN      |
| \[x\] > FLT_MAX   | NaN    | NaN      | Inf  | NaN      |
| \[x\] \< -FLT_MAX | NaN    | NaN      | -Inf | NaN      |
| else              | RNE    | RNE      | RNE  | RNE      |
