<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

(onnx-detail-float4)=

# Float stored in 4 bits

## Papers

4 bit floating point formats have emerged as a solution to the
rising cost and deployment challenges of large language models.
The S1E2M1 format has been part of the [Open Compute Project (OCP)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
standard.

As a result, a new data type was introduced in `onnx==1.18.0`
to support a limited set of operators to enable computation
with float4.

- `FLOAT4E2M1`: 1 bit for the sign, 2 bits for the exponents, and 1 bit for the mantissa.
  No nan or infinities.

## E2M1

$S$ stands for the sign. $10_2$ describe a number base 2.

```{eval-rst}
.. list-table:: Float4 type
   :widths: 10 10
   :header-rows: 1

   * -
     - E2M1
   * - Exponent bias
     - 1
   * - Infinities
     -
   * - NaN
     -
   * - Zeros
     - :math:`S.00.0_2`
   * - Max
     - :math:`S.11.1_2`
   * - Min
     - :math:`S.00.1_2 = 2^{-1}`

```

Let's denote the bit representation as $S.b_2 b_1 b_0$.
The float value is defined by the following expressions:

```{eval-rst}
.. list-table:: Float4 type values
   :widths: 10 10
   :header-rows: 1

   * -
     - E2M1
   * - exponent :math:`\neq` 0
     - :math:`(-1)^S 2^{\sum_{i=1}^2 b_i 2^{i-1} - 1} \left( 1 + b_0 2^{-1} \right)`
   * - exponent :math:`=` 0
     - :math:`(-1)^S  b_0 2^{-1}`
```

The following table lists all the representable values by float4 E2M1, ignoring the sign bit:
```{eval-rst}
.. list-table:: Float4 type values
   :widths: 10 10
   :header-rows: 1

   * - bits (ignoring sign bit)
     - E2M1
   * - 000
     - 0
   * - 001
     - 0.5
   * - 010
     - 1
   * - 011
     - 1.5
   * - 100
     - 2
   * - 101
     - 3
   * - 110
     - 4
   * - 111
     - 6
```

## Cast

Upcasting from float4 to float32, float16, bfloat16, and float8 is exact.
The behavior for downcasting to float 4 is summarized below

| x                 | E2M1                                              |
| ----------------- | ------------------------------------------------- |
| -6<=x<=6          | E2M1 converted value of x. Round to nearest even. |
| x=+/-0            | +/-0                                              |
| x>6               | 6                                                 |
| x<-6              | -6                                                |
| +Inf              | 6                                                 |
| -Inf              | -6                                                |
| NaN               | 6                                                 |

## Packing and Unpacking

Float4 is stored as 2x4bit in a single byte.
The first element is stored in the 4 LSB and the second element is stored in the 4 MSB,
i.e. for elements `x` and `y` that are consecutive elements in the array:
```
pack(x,y): y << 4 | x & 0x0F
unpack(z): x = z & 0x0F, y = z >> 4
```
In case the total number of elements is odd, padding of 4 bits will be appended.
The storage size of a 4 bit tensor of size `N` is `ceil(N/2)`.
