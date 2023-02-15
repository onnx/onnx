
.. _onnx-detail-float8:

Float stored on 8 bits
======================

.. contents::
    :local:

Papers
++++++

Two papers have been published in 2022 to introduce floats
stored on a byte as opposed to float 32 stored on 4 bytes.
The float precision is much lower but the training precision
does not suffer too much.

`FP8 Formats for Deep Learning <https://arxiv.org/abs/2209.05433>`_
from NVIDIA introduces two types following
`IEEE specifciations <https://en.wikipedia.org/wiki/IEEE_754>`_.
First one is E4M3, 1 bit for the sign, 4 bits for the exponents and 3
bits for the mantissa. Second one is E5M2, 1 bit for the sign,
3 bits for the exponents and 2 for the mantissa. The first types
is mostly used for the coefficients, the second one for the gradient.

Second paper `8-bit Numerical Formats For Deep Neural Networks
<https://arxiv.org/pdf/2206.02915.pdf>`_ introduces
similar types. IEEE standard gives the same value
to `+0` (or integer 0) and `-0` (or integer 128).
They chose to give distinct float values to these two
numbers. The paper experiments different split between
exponent and mantissa and shows and E4M3 and E5M2 are
the best ones.

FP8 from IEEE
+++++++++++++






