// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/doc_strings.h"

namespace ONNX_NAMESPACE {
#ifndef __ONNX_NO_DOC_STRINGS
const char kDoc_GRU_ver14[] = R"DOC(
Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.

Notations:

* `X` - input tensor
* `z` - update gate
* `r` - reset gate
* `h` - hidden gate
* `t` - time step (t-1 means previous time step)
* `W[zrh]` - W parameter weight matrix for update, reset, and hidden gates
* `R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates
* `Wb[zrh]` - W bias vectors for update, reset, and hidden gates
* `Rb[zrh]` - R bias vectors for update, reset, and hidden gates
* `WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates
* `RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates
* `WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates
* `RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates
* `H` - Hidden state
* `num_directions` - 2 if direction == bidirectional else 1

Activation functions:

* Relu(x)                - max(0, x)
* Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
* Sigmoid(x)             - 1/(1 + e^{-x})

NOTE:
  Below are optional

* Affine(x)              - alpha * x + beta
* LeakyRelu(x)           - x if x >= 0 else alpha * x
* ThresholdedRelu(x)     - x if x >= alpha else 0
* ScaledTanh(x)          - alpha * Tanh(beta * x)
* HardSigmoid(x)         - min(max(alpha * x + beta, 0), 1)
* Elu(x)                 - x if x >= 0 else alpha * (e^x - 1)
* Softsign(x)            - x/(1 + |x|)
* Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh):

* zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
* rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
* ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
* ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
* Ht = (1 - zt) (.) ht + zt (.) Ht-1
)DOC";

const char kDoc_Squeeze_ver24[] = R"DOC(
Remove single-dimensional entries from the shape of a tensor.
Takes an input `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.
)DOC";

const char kDoc_MaxUnpool_ver11[] = R"DOC(
MaxUnpool essentially computes the partial inverse of the MaxPool op.
 The input information to this op is typically the output information from a MaxPool op. The first
 input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)
 from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corresponding
 to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.
 The third (optional) input is a tensor that specifies the output size of the unpooling operation.

MaxUnpool is intended to do 'partial' inverse of the MaxPool op. 'Partial' because all the non-maximal
 values from the original input to MaxPool are set to zero in the output of the MaxUnpool op. Pooling
 the result of an unpooling operation should give back the original input to the unpooling op.

MaxUnpool can produce the same output size for several input sizes, which makes unpooling op ambiguous.
 The third input argument, output_size, is meant to disambiguate the op and produce output tensor of
 known/predictable size.

In addition to the inputs, MaxUnpool takes three attributes, namely kernel_shape, strides, and pads,
 which define the exact unpooling op. The attributes typically have the same values as the corresponding
 pooling op that the unpooling op is trying to invert.
)DOC";

const char kDoc_Size_ver24[] = R"DOC(
Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
)DOC";

const char kDoc_RandomUniform_ver1[] = R"DOC(
Generate a tensor with random values drawn from a uniform distribution. The shape
of the tensor is specified by the `shape` argument and the range by `low` and `high`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
)DOC";

const char kDoc_Range_ver11[] = R"DOC(
Generate a tensor containing a sequence of numbers that begin at `start` and extends by increments of `delta`
up to `limit` (exclusive).

The number of elements in the output of range is computed as below:
```
number_of_elements = max( ceil( (limit - start) / delta ) , 0 )
```
The pseudocode determining the contents of the output is shown below:
```
for(int i=0; i<number_of_elements; ++i) {
  output[i] =  start + (i * delta);
}
```
Example 1:
```
Inputs: start = 3, limit = 9, delta = 3
Output: [3, 6]
```
Example 2:
```
Inputs: start = 10, limit = 4, delta = -2
Output: [10, 8, 6]
```
)DOC";

const char kDoc_DequantizeLinear_ver24[] = R"DOC(
The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the
full-precision tensor. The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point`
must have the same shape, determining the quantization's granularity: a scalar for per-tensor/per-layer quantization,
a 1-D tensor for per-axis quantization, or have a rank identical to the input for blocked quantization.
See QuantizeLinear for details on quantization granularity.

`x_zero_point` and `x` must have the same type. `x` and `y` must have the same shape. In the case of dequantizing
`int32`, there's no zero point (zero point is supposed to be 0).
`zero-point` is usually not used in the case of float8 and 4-bit types quantization, but the dequantization formula remains the same
for consistency. The output type is determined by the attribute `output_dtype`. If `output_dtype` is not supplied then the output type
is the same as `x_scale`. The output type also determines the precision of the multiplication operation.

)DOC";

const char kDoc_RandomNormal_ver1[] = R"DOC(
Generate a tensor with random values drawn from a normal distribution. The shape
of the tensor is specified by the `shape` argument and the parameter of the normal distribution
specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
)DOC";

const char kDoc_Round_ver11[] = R"DOC(
Round takes one input Tensor and rounds the values, element-wise, meaning
it finds the nearest integer for each value.
In case of halves, the rule is to round them to the nearest even integer.
If input x is integral, +0, -0, NaN,  or infinite, x itself is returned.
The output tensor has the same shape and type as the input.

Examples:
```
round([0.9]) = [1.0]
round([2.5]) = [2.0]
round([2.3]) = [2.0]
round([1.5]) = [2.0]
round([-4.5]) = [-4.0]
```
)DOC";

const char kDoc_SpaceToDepth_ver1[] =
    R"DOC(SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
this op outputs a copy of the input tensor where values from the height and width dimensions
are moved to the depth dimension.
)DOC";

const char kDoc_InstanceNormalization_ver6[] = R"DOC(
Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B,
where mean and variance are computed per instance per channel.

)DOC";

const char kDoc_ThresholdedRelu_ver10[] = R"DOC(
ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise.
)DOC";

const char kDoc_Acosh_ver9[] = R"DOC(
Calculates the hyperbolic arccosine of the given input tensor element-wise.
)DOC";

const char kDoc_Dropout_ver13[] = R"DOC(
Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,
output (floating-point tensor) and mask (optional `Tensor<bool>`). If `training_mode` is true then the output Y will be a random dropout;
Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,
the user can simply not pass `training_mode` input or set it to false.
```
output = scale * data * mask,
```
where
```
scale = 1. / (1. - ratio).
```
)DOC";

const char kDoc_DeformConv_ver19[] = R"DOC(
Performs deformable convolution as described in https://arxiv.org/abs/1703.06211 and https://arxiv.org/abs/1811.11168.
This operator specification supports the general N-D case. Note that most common use cases have 2D or 3D data.
)DOC";

const char kDoc_Softplus_ver1[] = R"DOC(
Softplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
the tensor elementwise.
)DOC";

const char kDoc_Tile_ver6[] = R"DOC(Constructs a tensor by tiling a given tensor.
This is the same as function `tile` in Numpy, but no broadcast.
For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]
)DOC";

const char kDoc_Unsqueeze_ver24[] = R"DOC(
Insert single-dimensional entries to the shape of an input tensor (`data`).
Takes one required input `axes` - which contains a list of dimension indices and this operator will insert a dimension of value `1` into the corresponding index of the output tensor (`expanded`).

For example, given an input tensor (`data`) of shape [3, 4, 5], then
Unsqueeze(data, axes=[0, 4]) outputs a tensor (`expanded`) containing same data as `data` but with shape [1, 3, 4, 5, 1].

The input `axes` should not contain any duplicate entries. It is an error if it contains duplicates.
The rank of the output tensor (`output_rank`) is the rank of the input tensor (`data`) plus the number of values in `axes`.
Each value in `axes` should be within the (inclusive) range [-output_rank , output_rank - 1].
The order of values in `axes` does not matter and can come in any order.
)DOC";

const char kDoc_ConstantOfShape_ver24[] = R"DOC(
Generate a tensor with given value and shape.
)DOC";

const char kDoc_Elu_ver6[] = R"DOC(
Elu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.

)DOC";

const char kDoc_CumSum_ver11[] = R"DOC(
Performs cumulative sum of the input elements along the given axis.
By default, it will do the sum inclusively meaning the first element is copied as is.
Through an `exclusive` attribute, this behavior can change to exclude the first element.
It can also perform summation in the opposite direction of the axis. For that, set `reverse` attribute to 1.

Example:
```
input_x = [1, 2, 3]
axis=0
output = [1, 3, 6]
exclusive=1
output = [0, 1, 3]
exclusive=0
reverse=1
output = [6, 5, 3]
exclusive=1
reverse=1
output = [5, 3, 0]
```
 )DOC";

const char kDoc_Acos_ver7[] = R"DOC(
Calculates the arccosine (inverse of cosine) of the given input tensor, element-wise.
)DOC";

const char kDoc_LSTM_ver14[] = R"DOC(
Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

* `X` - input tensor
* `i` - input gate
* `o` - output gate
* `f` - forget gate
* `c` - cell gate
* `t` - time step (t-1 means previous time step)
* `W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates
* `R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates
* `Wb[iofc]` - W bias vectors for input, output, forget, and cell gates
* `Rb[iofc]` - R bias vectors for input, output, forget, and cell gates
* `P[iof]`  - P peephole weight vector for input, output, and forget gates
* `WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates
* `RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates
* `WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates
* `RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates
* `PB[iof]`  - P peephole weight vector for backward input, output, and forget gates
* `H` - Hidden state
* `num_directions` - 2 if direction == bidirectional else 1

Activation functions:

* Relu(x)                - max(0, x)
* Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
* Sigmoid(x)             - 1/(1 + e^{-x})

NOTE: Below are optional

* Affine(x)              - alpha*x + beta
* LeakyRelu(x)           - x if x >= 0 else alpha * x
* ThresholdedRelu(x)     - x if x >= alpha else 0
* ScaledTanh(x)          - alpha*Tanh(beta*x)
* HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
* Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
* Softsign(x)            - x/(1 + |x|)
* Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

* it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
* ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
* ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
* Ct = ft (.) Ct-1 + it (.) ct
* ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
* Ht = ot (.) h(Ct)
)DOC";

const char kDoc_Bernoulli_ver15[] = R"DOC(
Draws binary random numbers (0 or 1) from a Bernoulli distribution. The input tensor should be a tensor
containing probabilities p (a value in the range [0,1]) to be used for drawing the binary random number,
where an output of 1 is produced with probability p and an output of 0 is produced with probability (1-p).

This operator is non-deterministic and may not produce the same values in different
implementations (even if a seed is specified).
)DOC";

const char kDoc_Softsign_ver1[] = R"DOC(
Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.
)DOC";

const char kDoc_Selu_ver6[] = R"DOC(
Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.
)DOC";

const char kDoc_Shape_ver24[] = R"DOC(
Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
Optional attributes start and end can be used to compute a slice of the input tensor's shape.
If start axis is omitted, the slice starts from axis 0.
The end axis, if specified, is exclusive (and the returned value will not include the size of that axis).
If the end axis is omitted, the axes upto the last one will be included.
Negative axes indicate counting back from the last axis.
Note that axes will be clamped to the range [0, r], where r is the
rank of the input tensor if they are out-of-range (after adding r in the case of
negative axis). Thus, specifying any end value > r is equivalent to specifying an end
value of r, and specifying any start value < -r is equivalent to specifying a start
value of 0. If start > end, the result will be an empty shape.

Examples:

```
Input tensor with shape: [2, 3, 4]
No attributes specified.
Output: [2, 3, 4]
```

```
Input tensor with shape: [2, 3, 4]
start: -1
Output: [4]
```

```
Input tensor with shape: [2, 3, 4]
end: -1
Output: [2, 3]
```

```
Input tensor with shape: [2, 3, 4]
start: 1
end: 2
Output: [3]
```
)DOC";

const char kDoc_Upsample_ver7[] = R"DOC(
Upsample the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).
)DOC";

const char kDoc_Tanh_ver6[] = R"DOC(
Calculates the hyperbolic tangent of the given input tensor element-wise.
)DOC";

const char kDoc_Cast_ver24[] = R"DOC(
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.

Casting from string tensor in plain (e.g., "3.14" and "1000") and scientific numeric representations
(e.g., "1e-5" and "1E8") to float types is supported. For example, converting string "100.5" to an integer may
yield result 100. There are some string literals reserved for special floating-point values;
"+INF" (and "INF"), "-INF", and "NaN" are positive infinity, negative infinity, and not-a-number, respectively.
Any string which can exactly match "+INF" in a case-insensitive way would be mapped to positive infinite. Similarly,
this case-insensitive rule is applied to "INF" and "NaN". When casting from numeric tensors
to string tensors, plain floating-point representation (such as "314.15926") would be used.
Converting non-numerical-literal string such as "Hello World!" is an undefined behavior. Cases
of converting string representing floating-point arithmetic value, such as "2.718", to INT is an undefined behavior.

Conversion from a numerical type to any numerical type is always allowed.
User must be aware of precision loss and value change caused by range difference between two types.
For example, a 64-bit float 3.1415926459 may be round to a 32-bit float 3.141592. Similarly, converting
an integer 36 to Boolean may produce 1 because we truncate bits which can't be stored in the targeted type.

In more detail, the conversion among numerical types should follow these rules
if the destination type is not a float 8 type.

* Casting from floating point to:
  * floating point: +/- infinity if OOR (out of range).
  * fixed point: undefined if OOR.
  * bool: +/- 0.0 to False; all else to True.
* Casting from fixed point to:
  * floating point: +/- infinity if OOR. (+ infinity in the case of uint)
  * fixed point: when OOR, discard higher bits and reinterpret (with respect to two's complement representation for
    signed types). For example, 200 (int16) -> -56 (int8).
  * bool: zero to False; nonzero to True.
* Casting from bool to:
  * floating point: `{1.0, 0.0}`.
  * fixed point: `{1, 0}`.
  * bool: no change.

Float 8 types (E4M3FN, E4M3FNUZ, E5M2, E5M2FNUZ) were introduced to speed up the training of
deep models. By default the conversion of a float *x* obeys
to the following rules. `[x]` means the value rounded to
the target mantissa width.

| x                 | E4M3FN   | E4M3FNUZ | E5M2     | E5M2FNUZ |
| ----------------- | -------- | -------- | -------- | -------- |
| 0                 | 0        | 0        | 0        | 0        |
| -0                | -0       | 0        | -0       | 0        |
| NaN               | NaN      | NaN      | NaN      | NaN      |
| Inf               | FLT_MAX  | FLT_MAX  | FLT_MAX  | FLT_MAX  |
| -Inf              | -FLT_MAX | -FLT_MAX | -FLT_MAX | -FLT_MAX |
| \[x\] > FLT_MAX   | FLT_MAX  | FLT_MAX  | FLT_MAX  | FLT_MAX  |
| \[x\] \< -FLT_MAX | -FLT_MAX | -FLT_MAX | -FLT_MAX | -FLT_MAX |
| else              | RNE      | RNE      | RNE      | RNE      |

The behavior changes if the parameter 'saturate' is set to False.
The rules then become:

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

FLOAT8E8M0 type was introduced to enable [Microscaling (MX) formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
When casting to FLOAT8E8M0, the rounding behavior can be specified using the `round_mode` and `saturate` attributes.
The current CUDA behavior is to round up and saturate. Casting negative values to FLOAT8E8M0 gives undefined behavior.
The following table describes the casting behavior of special values to FLOAT8E8M0 in the two most common cases.

| x                 | saturate + up | non-saturate + nearest |
| ----------------- | ------------- | ---------------------  |
| 0                 | 0             | NaN                    |
| -0                | Unspecified   | Unspecified            |
| NaN               | NaN           | NaN                    |
| Inf               | E8M0_MAX      | NaN                    |
| x > E8M0_MAX      | E8M0_MAX      | NaN                    |
| x \< E8M0_MIN     | E8M0_MIN      | NaN                    |
| x \< 0            | Unspecified   | Unspecified            |
)DOC";

const char kDoc_Multinomial_ver7[] = R"DOC(
Generate a tensor of samples from a multinomial distribution according to the probabilities
of each of the possible outcomes.
)DOC";

const char kDoc_Constant_ver24[] = R"DOC(
This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
or value_* must be specified.
)DOC";

const char kDoc_Where_ver9[] = R"DOC(
Return elements, either from X or Y, depending on condition.
Where behaves like
[numpy.where](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html)
with three parameters.

)DOC";

const char kDoc_CastLike_ver24[] = R"DOC(
The operator casts the elements of a given input tensor (the first input) to
the same data type as the elements of the second input tensor.
See documentation of the Cast operator for further details.
)DOC";

const char kDoc_GridSample_ver20[] = R"DOC(
Given an input `X` and a flow-field `grid`, computes the output `Y` using `X` values and pixel locations from the `grid`.
For spatial input `X` with shape (N, C, H, W), the `grid` will have shape (N, H_out, W_out, 2),
the output `Y` will have shape (N, C, H_out, W_out). For volumetric input `X` with shape (N, C, D, H, W),
the `grid` will have shape (N, D_out, H_out, W_out, 3), the output `Y` will have shape (N, C, D_out, H_out, W_out).
More generally, for an input `X` of rank r+2 with shape (N, C, d1, d2, ..., dr),
the `grid` will have shape (N, D1_out, D2_out, ..., Dr_out, r), the output `Y` will have shape (N, C, D1_out, D2_out, ..., Dr_out).

The tensor `X` contains values at centers of square pixels (voxels, etc) locations such as (n, c, d1_in, d2_in, ..., dr_in).
The (n, d1_out, d2_out, ..., dr_out, :) values from the tensor `grid` are the normalized positions for interpolating the values
at the (n, c, d1_out, d2_out, ..., dr_out) locations from the output tensor `Y` using a specified interpolation method (the mode)
and a padding mode (for `grid` positions falling outside the 2-dimensional image).

For example, the values in `grid[n, h_out, w_out, :]` are size-2 vectors specifying normalized positions in the 2-dimensional space of `X`.
They are used to interpolate output values of `Y[n, c, h_out, w_out]`.

The GridSample operator is often used in doing grid generator and sampler in the
[Spatial Transformer Networks](https://arxiv.org/abs/1506.02025).
See also in [torch.nn.functional.grid_sample](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html).
)DOC";

const char kDoc_Atanh_ver9[] = R"DOC(
Calculates the hyperbolic arctangent of the given input tensor element-wise.
)DOC";

const char kDoc_Flatten_ver24[] = R"DOC(
Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
)DOC";

const char kDoc_Reciprocal_ver6[] = R"DOC(
Reciprocal takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the reciprocal is, y = 1/x, is applied to
the tensor elementwise.
)DOC";

const char kDoc_Pow_ver13[] = R"DOC(
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
)DOC";

const char kDoc_Tan_ver7[] = R"DOC(
Calculates the tangent of the given input tensor, element-wise.
)DOC";

const char kDoc_mish_ver18[] = R"DOC(
Mish: A Self Regularized Non-Monotonic Neural Activation Function.

Perform the linear unit element-wise on the input tensor X using formula:

```
mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
```
)DOC";

const char kDoc_RoiAlign_ver16[] = R"DOC(
Region of Interest (RoI) align operation described in the
[Mask R-CNN paper](https://arxiv.org/abs/1703.06870).
RoiAlign consumes an input tensor X and region of interests (rois)
to apply pooling across each RoI; it produces a 4-D tensor of shape
(num_rois, C, output_height, output_width).

RoiAlign is proposed to avoid the misalignment by removing
quantizations while converting from original image into feature
map and from feature map into RoI feature; in each ROI bin,
the value of the sampled locations are computed directly
through bilinear interpolation.
)DOC";

const char kDoc_LeakyRelu_ver1[] = R"DOC(
LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.
)DOC";

const char kDoc_Det_ver11[] = R"DOC(
Det calculates determinant of a square matrix or batches of square matrices.
Det takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
and the inner-most 2 dimensions form square matrices.
The output is a tensor of shape `[*]`, containing the determinants of all input submatrices.
e.g., When the input is 2-D, the output is a scalar(shape is empty: `[]`).
)DOC";

const char kDoc_RandomUniformLike_ver1[] = R"DOC(
Generate a tensor with random values drawn from a uniform distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the uniform distribution are specified by `low` and `high`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
)DOC";

const char kDoc_Relu_ver6[] = R"DOC(
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
)DOC";

const char kDoc_RandomNormalLike_ver1[] = R"DOC(
Generate a tensor with random values drawn from a normal distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the normal distribution are specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message, and be valid as an output type.
)DOC";

const char kDoc_Exp_ver6[] = R"DOC(
Calculates the exponential of the given input tensor, element-wise.
)DOC";

const char kDoc_Sign_ver9[] = R"DOC(
Calculate the sign of the given input tensor element-wise.
If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.
)DOC";

const char kDoc_Cosh_ver9[] = R"DOC(
Calculates the hyperbolic cosine of the given input tensor element-wise.
)DOC";

const char kDoc_Sigmoid_ver6[] = R"DOC(
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
)DOC";

const char kDoc_HardSigmoid_ver6[] = R"DOC(
HardSigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
is applied to the tensor elementwise.
)DOC";

const char kDoc_LpNormalization_ver1[] = R"DOC(
Given a matrix, apply Lp-normalization along the provided axis.
The output is computed as: `output = input / Lp_norm(input, axis)`.
When the Lp norm is zero (i.e., all elements along the axis are zero),
the output is defined to be zero to avoid division by zero.
)DOC";

const char kDoc_Erf_ver9[] = R"DOC(
Computes the error function of the given input tensor element-wise.
)DOC";

const char kDoc_Asinh_ver9[] = R"DOC(
Calculates the hyperbolic arcsine of the given input tensor element-wise.
)DOC";

const char kDoc_Sinh_ver9[] = R"DOC(
Calculates the hyperbolic sine of the given input tensor element-wise.
)DOC";

const char kDoc_Atan_ver7[] = R"DOC(
Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.
)DOC";

const char kDoc_Sqrt_ver6[] = R"DOC(
Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.
)DOC";

const char kDoc_Asin_ver7[] = R"DOC(
Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.
)DOC";

const char kDoc_Expand_ver8[] = R"DOC(
Broadcast the input tensor following the given shape and the broadcast rule.
The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
Dimensions are right alignment;
Two corresponding dimensions must have the same value, or one of them is equal to 1.
Also, this operator is similar to numpy.broadcast_to(input, shape),
but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
or the shape.ndim < input.shape.ndim.
)DOC";

const char kDoc_scan_24[] = R"DOC(
Scan can be used to iterate over one or more scan_input tensors,
constructing zero or more scan_output tensors. It combines ideas from general recurrences,
functional programming constructs such as scan, fold, map, and zip, and is intended to enable
generalizations of RNN-like constructs for sequence-to-sequence processing.
Other tensors (referred to as state_variables here) can be used to carry a state
when iterating from one element to another (similar to hidden-state in RNNs, also referred
to as loop-carried dependences in the context of loops).
Many common usages involve a single scan_input tensor (where functionality
similar to scan, fold and map can be obtained). When more than one scan_input is used,
a behavior similar to zip is obtained.

The attribute body must be a graph, specifying the computation to be performed in
every iteration. It takes as input the current values of the state_variables and
the current iterated element of the scan_inputs. It must return the (updated) values
of the state_variables and zero or more scan_output_element tensors. The values of the
scan_output_element tensors are concatenated over all the iterations to produce the
scan_output values of the scan construct (similar to the concatenated intermediate
hidden-state values of RNN-like constructs). All the output tensors (state_variables as
well as scan_output_element tensors) are required to have the same shape in each iteration
of the loop (a restriction imposed to enable efficient memory allocation).

Note that the iterated element passed to the body subgraph does not have a sequence
axis. It will have a rank one less than the rank of the corresponding scan_input.

The scan operation returns the final values of the state_variables as well as the
scan_outputs.

The optional attribute scan_input_directions specifies the direction (forward or backward)
for each scan input. If this attribute is omitted, all sequences are scanned in the forward
direction. A bidirectional scan may be performed by specifying the same tensor input twice
in the scan_inputs, once with a forward direction, and once with a backward direction.

The scan_output of the operation is produced by concatenating the scan_output_element
values produced by the body in each iteration.  The optional attribute scan_output_directions
specifies the direction in which scan_output is constructed (by appending or prepending the
scan_output_element to scan_output in each iteration) for each scan_output. If this attribute
is omitted, the scan_output_element is appended to the scan_output in each iteration.

The optional attribute scan_input_axes specifies the axis to be scanned for each scan_input.
If omitted, every scan_input will be scanned in axis 0. For example, if axis 0 is the
batch axis and axis 1 is the time axis (to be scanned), specify an axis value of 1.
Note that scanning a non-zero axis may be less efficient than scanning axis zero.

The optional attribute scan_output_axes specifies the axis along which the scan_outputs
are accumulated for each scan_output. For example, if axis 1 is the time axis (to be
scanned) for both inputs and outputs, specify a scan_input axis and scan_output axis
value of 1.

Note that because of the ONNX restriction that only the last parameter of an operator can
be variadic, the initial-states and scan-inputs are listed together as one input parameter.
Similarly, the final-states and scan-outputs are listed together as one output parameter.
The attribute num_scan_inputs indicates the number M of scan-inputs.

The behavior of

    Scan <
        num_scan_inputs = m,
        body = loop-body,
        scan_input_axes = [axis_1, ..., axis_m]
    > (init_1, ..., init_n, scan_1, ..., scan_m)

is equivalent to the following pseudo-code:

    // scan_i.shape[axis_i] denotes the (max) sequence-length of scan_i
    // scan_i.shape[axis_i] is required to be equal to scan_j.shape[axis_j] for all i,j.
    sequence_length = scan_1.shape[axis_1];

    // initialize state-variables
    st_1 = init_1; ... st_n = init_n;
    // initialize scan-output variables: [] denotes an empty tensor
    scan_out_1 = []; ...; scan_out_k = [];
    // identify number of iterations:

    // execute loop
    for (int t = 0; t < sequence_length; ++t) {
        // generate the scan-input elements: the notation T<axis=k>[t] indicates the sub-tensor
        // of rank one less than T obtained by indexing T at position t along axis k.
        si_1 = scan_1<axis=axis_1>[t];
        ... ;
        si_m = scan_m<axis=axis_m>[t];
        // execute loop-body
        st_1, ..., st_n, so_1, ..., so_k = loop-body(st_1, ..., st_n, si_1, ..., si_m)
        // accumulate the scan-output elements
        scan_out_1 = Concat<axis=0>(scan_out_1, so_1); ... ; scan_out_k = Concat<axis=0>(scan_out_k, so_k);
    }

    return st_1, ..., st_n, scan_out_1, ..., scan_out_k;

*Sample usage: Encoding RNN using a Scan*

The following example shows how a simple RNN over an input tensor %X, with weight tensor %Wi,
recurrence weight tensor %Ri, bias tensors %Wbi and %Rbi, and initial hidden-state %H_0 can
be encoded as a ScanLoop. Note that the loop-body is a nested graph, and it directly computes
%Wi, %Ri, %Wbi, and %Rbi (typically constants or initializers in the body graph). If these
values are computed in the outer graph, they need to be passed in as extra state_variables.

    graph rnn-encoding {
      %H_0 = ...
      %X = ...
      %Y_h, %Y = Scan[body = <graph rnn-cell-1>, num_scan_inputs=1](%H_0, %X)
      return %Y, %Y_h
    }

    graph rnn-cell-1 (
      %H_tminus1[FLOAT, tensor]
      %X_t[FLOAT, tensor]
    ) {
      %Wi = ...
      %Ri = ...
      %Wbi = ...
      %Rbi = ...
      %t1 = X_t * (Wi^T)
      %t2 = H_tminus1*(Ri^T)
      %t3 = Add(%t1, %t2)
      %t4 = Add(%t3, %Wbi)
      %t5 = Add(%t4, %Rbi)
      %Ht = Tanh(%t5)
      %Accumulate = Identity(%Ht)
      return %Ht, %Accumulate
    }

)DOC";

const char kDoc_Pad_ver24[] = R"DOC(
Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
a padded tensor (`output`) is generated.

The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):

1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0, empty string, or False)

2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

3) `edge` - pads with the edge values of array

4) `wrap` - wrap-around padding as if the data tensor forms a torus


Example 1 (`constant` mode):

Insert 0 pads to the beginning of the second dimension.

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'constant'

constant_value = 0.0

output = [
    [0.0, 0.0, 1.0, 1.2],
    [0.0, 0.0, 2.3, 3.4],
    [0.0, 0.0, 4.5, 5.7],
]
```

Example 2 (`reflect` mode):

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'reflect'

output = [
    [1.0, 1.2, 1.0, 1.2],
    [2.3, 3.4, 2.3, 3.4],
    [4.5, 5.7, 4.5, 5.7],
]
```

Example 3 (`edge` mode):

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'edge'

output = [
    [1.0, 1.0, 1.0, 1.2],
    [2.3, 2.3, 2.3, 3.4],
    [4.5, 4.5, 4.5, 5.7],
]
```

Example 4 (`wrap` mode):

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [2, 1, 1, 1]

mode = 'wrap'

output = [
    [3.4, 2.3, 3.4, 2.3],
    [5.7, 4.5, 5.7, 4.5],
    [1.2, 1.0, 1.2, 1.0],
    [3.4, 2.3, 3.4, 2.3],
    [5.7, 4.5, 5.7, 4.5],
    [1.2, 1.0, 1.2, 1.0],
]
```
)DOC";

const char kDoc_Cos_ver7[] = R"DOC(
Calculates the cosine of the given input tensor, element-wise.
)DOC";

const char kDoc_HardSwish_ver14[] = R"DOC(
HardSwish takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where
the HardSwish function, y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x),
where alpha = 1/6 and beta = 0.5, is applied to the tensor elementwise.
)DOC";

const char kDoc_MatMul_ver9[] = R"DOC(
Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
)DOC";

const char kDoc_NegativeLogLikelihoodLoss_ver13[] = R"DOC(
A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
Its "input" tensor has the shape of (N, C, d1, d2, ..., dk) where k >= 0.
The "input" tensor contains log-probabilities for input[n, :, d_1, d_2,..., d_k] being in a class of [0, C).
The operator's "target" input tensor has the shape of (N, d1, d2, ..., dk). It encodes class labels (one of C classes)
or it may contain a special value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x dk samples.
The loss value for input[n, :, d_1, d_2,...d_k] being classified as class c = target[n][d_1][d_2]...[d_k] is computed as:

```
loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k].
```

When an optional "weight" is provided, the sample loss is calculated as:

```
loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k] * weight[c].
```

loss is zero for the case when target-value equals ignore_index.

```
loss[n][d_1][d_2]...[d_k] = 0, when target[n][d_1][d_2]...[d_k] = ignore_index
```

If "reduction" attribute is set to "none", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).
If "reduction" attribute is set to "mean" (the default attribute value), the output loss is (weight) averaged:

```
mean(loss), if "weight" is not provided,
```

or if weight is provided,

```
sum(loss) / sum(weight[target[n][d_1][d_2]...[d_k]]]), for all samples.
```

If "reduction" attribute is set to "sum", the output is a scalar: `sum(loss)`.

See also https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss.

Example 1:

```
// negative log likelihood loss, "none" reduction
N, C, d1 = 2, 3, 2
input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
          [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
target = [[2, 1], [0, 2]]

loss = np.zeros((N, d1))
for n in range(N):
    for d_1 in range(d1):
        c = target[n][d_1]
        loss[n][d_1] = -input[n][c][d_1]

// print(loss)
// [[-3. -2.]
//  [-0. -2.]]
```

Example 2:

```
// weighted negative log likelihood loss, sum reduction
N, C, d1 = 2, 3, 2
input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
        [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
target = [[2, 1], [0, 2]]
weight = [0.2, 0.3, 0.1]
loss = np.zeros((N, d1))
for n in range(N):
    for d_1 in range(d1):
        c = target[n][d_1]
        loss[n][d_1] = -input[n][c][d_1] * weight[c]

loss = np.sum(loss)
// print(loss)
// -1.1
```

Example 3:

```
// weighted negative log likelihood loss, mean reduction
N, C, d1 = 2, 3, 2
input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
        [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
target = [[2, 1], [0, 2]]
weight = [0.2, 0.3, 0.1]
loss = np.zeros((N, d1))
weight_total = 0
for n in range(N):
    for d_1 in range(d1):
        c = target[n][d_1]
        loss[n][d_1] = -input[n][c][d_1] * weight[c]
        weight_total = weight_total + weight[c]

loss = np.sum(loss) / weight_total
// print(loss)
// -1.57
```
)DOC";

const char kDoc_Sin_ver7[] = R"DOC(
Calculates the sine of the given input tensor, element-wise.
)DOC";

const char kDoc_Loop_ver23[] = R"DOC(
Generic Looping construct. This loop has multiple termination conditions:

1) Trip count. Iteration count specified at runtime. Set by
   specifying the input M. Optional. Set to empty string to omit.
   Note that a static trip count (specified at graph construction time) can be
   specified by passing in a constant node for input M.
2) Loop termination condition. This is an input to the op that determines
   whether to run the first iteration and also a loop-carried dependency for
   the body graph. The body graph must yield a value for the condition variable,
   whether this input is provided or not.

This table summarizes the operating modes of this operator with equivalent
C-style code:

Operator inputs defined as (max_trip_count, condition_var).

* input ("", ""):
        for (int i=0; ; ++i) {
          cond = ... // Note this value is ignored, but is required in the body
        }

* input ("", cond) // Note this is analogous to a while loop
        bool cond = ...;
        for (int i=0; cond; ++i) {
          cond = ...;
        }

* input ("", 1) // Note this is analogous to a do-while loop
        bool cond = true
        for (int i=0; cond; ++i) {
          cond = ...;
        }

* input (trip_count, "") // Note this is analogous to a for loop
        int trip_count = ...
        for (int i=0; i < trip_count; ++i) {
          cond = ...; // ignored
        }

* input (trip_count, cond)
        int trip_count = ...;
        bool cond = ...;
        for (int i=0; i < trip_count && cond; ++i) {
          cond = ...;
        }


*Sample usage - cond as well as trip count*

    graph predict-net {
      %a = Constant[value = <Scalar Tensor [3]>]()
      %b = Constant[value = <Scalar Tensor [6]>]()
      %keepgoing = Constant[value = <Scalar Tensor [1]>]()
      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
      return
    }

    graph body-net (
      %i[INT32, scalar]           // iteration number
      %keepgoing_in[BOOL, scalar] // incoming loop-termination-condition; not used
      %b_in[INT32, scalar]        // incoming value of loop-carried-dependency b
    ) {
      %my_local = Add(%a, %b_in)
      %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b
      %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition
      %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated
      return %keepgoing_out, %b_out, %user_defined_val
    }

*Sample equivalent C code*

    {
      /* User-defined code (enclosing scope) */
      int a = 3, b = 6;
      bool keepgoing = true; // Analogous to input cond
      /* End user-defined code */

      /* Implicitly-defined code */
      const int max_trip_count = 10; // Analogous to input M
      int user_defined_vals[]; // Imagine this is resizable
      /* End implicitly-defined code */
      /* initialize loop-carried variables and scan-output variables */
      bool keepgoing_out = keepgoing
      int b_out = b

      for (int i=0; i < max_trip_count && keepgoing_out; ++i) {
        /* Implicitly-defined code: bind actual parameter values
           to formal parameter variables of loop-body */
        bool keepgoing_in = keepgoing_out;
        bool b_in = b_out;

        /* User-defined code (loop body) */
        int my_local = a + b_in; // Reading value "a" from the enclosing scope is fine
        b_out = a - b_in;
        keepgoing_out = my_local > b_out;
        user_defined_val = b_in + b_in; // b_in and b_out are different variables
        /* End user-defined code */

        /* Implicitly defined-code */
        user_defined_vals[i] = user_defined_val // accumulate scan-output values
      }
      // int t = my_local; // Can't do this. my_local is not accessible here.

      // The values below are bound to the output variables of the loop and therefore accessible
      // b_out; user_defined_vals; keepgoing_out;
    }

There are several things of note in this code snippet:

1) Values from the enclosing scope (i.e. variable "a" here) are in scope and can
   be referenced in the inputs of the loop.
2) Any values computed in the loop body that needs to be used in a subsequent
   iteration or after the loop are modeled using a pair of variables in the loop-body,
   consisting of an input variable (eg., b_in) and an output variable (eg., b_out).
   These are referred to as loop-carried dependences. The loop operation node
   supplies the input value of the input variable for the first iteration, and
   returns the output value of the output variable produced by the final
   iteration.
3) Scan_output variables are used to implicitly concatenate values computed across
   all the iterations. In the above example, the value of user_defined_val computed
   over all iterations are concatenated and returned as the value of user_defined_vals
   after the loop.
4) Values created in the body cannot be accessed in the enclosing scope,
   except using the mechanism described above.

Note that the semantics of this op support "diagonal" or "wavefront" execution.
(See Step 3 here for an example:
https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
Frontends should emit multi-layer RNNs as a series of While operators (with
time being the inner looping dimension), with each successive layer consuming
the scan_outputs from the previous layer, possibly going through several
point-wise operators (e.g. dropout, residual connections, linear layer).

The input/output of subgraph (produced by loop node) matching is based on order instead of name. The implementation will figure out the names based on this order.
)DOC";

const char kDoc_RNN_ver14[] = R"DOC(
Computes an one-layer simple RNN. This operator is usually supported
via some custom implementation such as CuDNN.

Notations:

* `X` - input tensor
* `i` - input gate
* `t` - time step (t-1 means previous time step)
* `Wi` - W parameter weight matrix for input gate
* `Ri` - R recurrence weight matrix for input gate
* `Wbi` - W parameter bias vector for input gate
* `Rbi` - R parameter bias vector for input gate
* `WBi` - W parameter weight matrix for backward input gate
* `RBi` - R recurrence weight matrix for backward input gate
* `WBbi` - WR bias vectors for backward input gate
* `RBbi` - RR bias vectors for backward input gate
* `H` - Hidden state
* `num_directions` - 2 if direction == bidirectional else 1

Activation functions:

* Relu(x)                - max(0, x)
* Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
* Sigmoid(x)             - 1/(1 + e^{-x})

NOTE: Below are optional

* Affine(x)              - alpha*x + beta
* LeakyRelu(x)           - x if x >= 0 else alpha * x
* ThresholdedRelu(x)     - x if x >= alpha else 0
* ScaledTanh(x)          - alpha*Tanh(beta*x)
* HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
* Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
* Softsign(x)            - x/(1 + |x|)
* Softplus(x)            - log(1 + e^x)

Equations (Default: f=Tanh):

* Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
)DOC";

const char kDoc_NonMaxSuppression_ver10[] = R"DOC(
Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
Boxes are suppressed if their IOU with a previously selected box is strictly greater than iou_threshold (i.e., boxes with IOU exactly equal to the threshold are kept).
Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system
result in the same boxes being selected by the algorithm.
The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.
)DOC";

const char kDoc_Log_ver6[] = R"DOC(
Calculates the natural log of the given input tensor, element-wise.
)DOC";

const char kDoc_EyeLike_ver9[] = R"DOC(
Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else. Only 2D
tensors are supported, i.e. input T1 must be of rank 2. The shape of the output tensor is the
same as the input tensor. The data type can be specified by the 'dtype' argument. If
'dtype' is not specified, then the type of input tensor is used. By default, the main diagonal
is populated with ones, but attribute 'k' can be used to populate upper or lower diagonals.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
)DOC";

const char kDoc_Reshape_ver24[] = R"DOC(
Reshape the input tensor similar to numpy.reshape.
First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor). If 'allowzero' is set, and the new shape includes 0, the
dimension will be set explicitly to zero (i.e. not taken from input tensor).
Shape (second input) could be an empty shape, which means converting to a scalar.
The input tensor's shape and the output tensor's shape are required to have the same number of elements.

If the attribute 'allowzero' is set, it is invalid for the specified shape to
contain both a zero value and -1, as the value of the dimension corresponding
to -1 cannot be determined uniquely.
)DOC";

const char kDoc_Compress_ver9[] = R"DOC(
    Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
    In case axis is not provided, input is flattened before elements are selected.
    Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html
    )DOC";

const char kDoc_PRelu_ver7[] = R"DOC(
PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
)DOC";

const char kDoc_Neg_ver6[] = R"DOC(
Neg takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where each element flipped sign, y = -x, is applied to
the tensor elementwise.
)DOC";

#else
const char kDoc_GRU_ver14[] = "";
const char kDoc_Squeeze_ver24[] = "";
const char kDoc_MaxUnpool_ver11[] = "";
const char kDoc_Size_ver24[] = "";
const char kDoc_RandomUniform_ver1[] = "";
const char kDoc_Range_ver11[] = "";
const char kDoc_DequantizeLinear_ver24[] = "";
const char kDoc_RandomNormal_ver1[] = "";
const char kDoc_Round_ver11[] = "";
const char kDoc_SpaceToDepth_ver1[] = "";
const char kDoc_InstanceNormalization_ver6[] = "";
const char kDoc_ThresholdedRelu_ver10[] = "";
const char kDoc_Acosh_ver9[] = "";
const char kDoc_Dropout_ver13[] = "";
const char kDoc_DeformConv_ver19[] = "";
const char kDoc_Softplus_ver1[] = "";
const char kDoc_Tile_ver6[] = "";
const char kDoc_Unsqueeze_ver24[] = "";
const char kDoc_ConstantOfShape_ver24[] = "";
const char kDoc_Elu_ver6[] = "";
const char kDoc_CumSum_ver11[] = "";
const char kDoc_Acos_ver7[] = "";
const char kDoc_LSTM_ver14[] = "";
const char kDoc_Bernoulli_ver15[] = "";
const char kDoc_Softsign_ver1[] = "";
const char kDoc_Selu_ver6[] = "";
const char kDoc_Shape_ver24[] = "";
const char kDoc_Upsample_ver7[] = "";
const char kDoc_Tanh_ver6[] = "";
const char kDoc_Cast_ver24[] = "";
const char kDoc_Multinomial_ver7[] = "";
const char kDoc_Constant_ver24[] = "";
const char kDoc_Where_ver9[] = "";
const char kDoc_CastLike_ver24[] = "";
const char kDoc_GridSample_ver20[] = "";
const char kDoc_Atanh_ver9[] = "";
const char kDoc_Flatten_ver24[] = "";
const char kDoc_Reciprocal_ver6[] = "";
const char kDoc_Pow_ver13[] = "";
const char kDoc_Tan_ver7[] = "";
const char kDoc_mish_ver18[] = "";
const char kDoc_RoiAlign_ver16[] = "";
const char kDoc_LeakyRelu_ver1[] = "";
const char kDoc_Det_ver11[] = "";
const char kDoc_RandomUniformLike_ver1[] = "";
const char kDoc_Relu_ver6[] = "";
const char kDoc_RandomNormalLike_ver1[] = "";
const char kDoc_Exp_ver6[] = "";
const char kDoc_Sign_ver9[] = "";
const char kDoc_Cosh_ver9[] = "";
const char kDoc_Sigmoid_ver6[] = "";
const char kDoc_HardSigmoid_ver6[] = "";
const char kDoc_LpNormalization_ver1[] = "";
const char kDoc_Erf_ver9[] = "";
const char kDoc_Asinh_ver9[] = "";
const char kDoc_Sinh_ver9[] = "";
const char kDoc_Atan_ver7[] = "";
const char kDoc_Sqrt_ver6[] = "";
const char kDoc_Asin_ver7[] = "";
const char kDoc_Expand_ver8[] = "";
const char kDoc_scan_24[] = "";
const char kDoc_Pad_ver24[] = "";
const char kDoc_Cos_ver7[] = "";
const char kDoc_HardSwish_ver14[] = "";
const char kDoc_MatMul_ver9[] = "";
const char kDoc_NegativeLogLikelihoodLoss_ver13[] = "";
const char kDoc_Sin_ver7[] = "";
const char kDoc_Loop_ver23[] = "";
const char kDoc_RNN_ver14[] = "";
const char kDoc_NonMaxSuppression_ver10[] = "";
const char kDoc_Log_ver6[] = "";
const char kDoc_EyeLike_ver9[] = "";
const char kDoc_Reshape_ver24[] = "";
const char kDoc_Compress_ver9[] = "";
const char kDoc_PRelu_ver7[] = "";
const char kDoc_Neg_ver6[] = "";
#endif
} // namespace ONNX_NAMESPACE
