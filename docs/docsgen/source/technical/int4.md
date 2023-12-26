<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

(onnx-detail-int4)=

# 4 bit integer types

## Papers

Several papers have been published in 2023 to introduce 4 bit integers and their usage in LLMs. Although their range is
limited, with careful selection of scaling parameters, good accuracy is obtained when used for compression of weights
(weight-only quantization), and in some cases for quantization of activations as well.

[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
Activation-aware Weight Quantization (AWQ) focuses on the quantization of weights in LLMs by considering the
observation that not all weights are equally important. The method aims to protect salient weights based on the
activation, rather than relying on backpropagation or reconstruction techniques. By searching for the optimal
per-channel scaling that preserves the crucial weights, AWQ aims to minimize quantization errors.

[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
GPTQ proposes a one-shot weight quantization method based on approximate second-order information. GPTQ achieves
significant compression gains, reducing the bit-width to 3 or 4 bits per weight with negligible accuracy degradation
compared to the uncompressed baseline.

[Understanding INT4 Quantization for Transformer Models: Latency Speedup, Composability, and Failure Cases](https://arxiv.org/abs/2301.12017)
This paper discusses quantization of both weights and activations to 4 bit (W4A4). Results indicate that W4A4
quantization leads to little to no accuracy degradation for encoder-only and encoder-decoder models but results in
a significant accuracy drop for decoder-only models. To realize the performance gains using W4A4, the study introduces
a highly optimized end-to-end W4A4 encoder inference pipeline that supports various quantization strategies.

As a result, two new types were introduced in `onnx==1.17.0` supporting a limited set of operators to enable compression using
4 bit data-types:
- `UINT4`: 4 bit unsigned integer, values in range [0, 15]
- `INT4`: 4 bit signed integer, using two's complement represntation. Values in range [-8, 7].

## Cast

Cast from 4 bit to any higher precision type is exact.
Cast to a 4 bit type is done by rounding to the nearest-integer (with ties to even)
nearest-even integer and truncating.

## Packing and Unpacking

All 4 bit types are stored as 2x4bit in a single byte.
The first element is stored in the 4 LSB and the second element is stored in the 4 MSB.
i.e. for elements x, y, that are consecutive elements in the array:
```{eval-rst}
pack(x,y): y << 4 | x & 0x0F
unpack(z): x = z & 0x0F, y = z >> 4
```
In case the total number of elements is odd, padding of 4 bits will be appended.
The storage size of a 4 bit tensor of size `N` is `ceil(N/2)`.
