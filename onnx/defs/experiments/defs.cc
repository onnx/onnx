// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
using namespace onnx;

using AttrType = onnx::OpSchema::AttrType;
using SupportType = onnx::OpSchema::SupportType;

OPERATOR_SCHEMA(Identity)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc("Identity operator")
    .Input(0, "input", "Input tensor", "T")
    .Output(
        0,
        "output",
        "Tensor to copy input into. Can be in-place",
        "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(Affine)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Affine takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the affine function, y = alpha * x + beta,
is applied to the tensor elementwise.
)DOC")
    .Attr("alpha", "Value of alpha", AttrType::FLOAT)
    .Attr("beta" , "Value of beta", AttrType::FLOAT)
    .Input(0, "X", "1D input tensor", "T")
    .Output(0, "Y", "1D output tensor", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain input and output types to float tensors.");


OPERATOR_SCHEMA(ThresholdedRelu)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > theta, y = 0 otherwise,
is applied to the tensor elementwise.
)DOC")
    .Attr("theta",
          "Threshold value",
          AttrType::FLOAT)
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(ScaledTanh)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
Calculates the scaled hyperbolic tangent of the given input tensor element-wise,
scale * tanh(x). This operation can be done in an in-place fashion too,
by providing the same input and output blobs.
    )DOC")
    .Attr("scale",
        "Scale for tanh",
        AttrType::FLOAT)
    .Input(0, "input", "1-D input tensor", "T")
    .Output(0, "output", "The scaled hyperbolic tangent values of the input tensor "
        "computed element-wise", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(ParametricSoftplus)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
ParametricSoftplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = alpha * ln(exp(beta * x) + 1), is applied to
the tensor elementwise.
)DOC")
    .Attr("alpha", "Value of alpha", AttrType::FLOAT)
    .Attr("beta", "Value of beta", AttrType::FLOAT)
    .Input(0, "X", "1D input tensor", "T")
    .Output(0, "Y", "1D input tensor", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(ConstantFill)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
The operator fills the elements of the output tensor with a constant value
specified by the 'value' attribute.

The data type is specified by the 'dtype' attribute. The 'dtype' attribute must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message. If the 'dtype' attribute is not provided, the data type of
'value' is used.

The output tensor shape is specified by the 'shape' attribute. If the number of
input is 1, the shape will be identical to that of the input at run time with
optional additional dimensions appended at the end as specified by 'extra_shape'
attribute. In that case the 'shape' attribute should not be set.

If input_as_shape is set to true, then the input should be a 1D tensor
containing the desired output shape (the dimensions specified in extra_shape
will also be appended)

NOTE: Currently, it supports data type of float, int32, int64, and bool.
)DOC")
    .Attr(
        "value",
        "The value for the elements of the output tensor.",
        AttrType::FLOAT)
    .Attr(
        "dtype",
        "The data type for the elements of the output tensor."
        "Strictly must be one of the types from DataType enum in TensorProto.",
        AttrType::INT)
    .Attr(
        "shape",
        "The shape of the output tensor."
        "Cannot set the shape argument and pass in an input at the same time.",
        AttrType::INTS)
    .Attr(
        "extra_shape",
        "The additional dimensions appended at the end of the shape indicated"
        "by the input blob."
        "Cannot set the extra_shape argument when there is no input blob.",
        AttrType::INTS)
    .Attr(
        "input_as_shape",
        "1D tensor containing the desired output shape.  First input must be in "
        "CPU context.",
        AttrType::INT)
    .Input(
        0,
        "input",
        "Input tensor (optional) to provide shape information.",
        "T1",
        true)
    .Output(
        0,
        "output",
        "Output tensor of constant values specified by 'value'"
        "argument and its type is specified by the 'dtype' argument",
        "T2")
    .TypeConstraint(
        "T1",
        {"tensor(float)", "tensor(int32)", "tensor(int64)", "tensor(bool)"},
        "Constrain input types to float, int32, int64, bool tensors.")
    .TypeConstraint(
        "T2",
        {"tensor(float)", "tensor(int32)", "tensor(int64)", "tensor(bool)"},
        "Constrain output types to float, int32, int64, bool tensors.");

OPERATOR_SCHEMA(GivenTensorFill)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .Input(0, "shape", "The shape of filled tensor", "T")
    .Output(0, "X", "The filled tensor", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.")
    .Attr("values", "", AttrType::FLOATS)
    .Attr("shape", "", AttrType::INTS)
    .Attr("input_as_shape", "", AttrType::INT)
    .Attr("extra_shape", "", AttrType::INTS)
    .AllowConsumed({{0, 0}});

OPERATOR_SCHEMA(FC)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Computes the result of passing an input vector X into a fully
connected layer with 2D weight matrix W and 1D bias vector b. That is,
the layer computes Y = X * W^T + b, where X has size (M x K),
W has size (N x K), b has size (N), and Y has size (M x N),
where M is often the batch size.
NOTE: X does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
X \in [a_0, a_1, ...,a_{k-1}, a_k, ..., a_{n-1}] where a_i \in N+ and k is
the axis provided, then X will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the X tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = M and a_1 * ... * a_{n-1} = K.
Lastly, even though b is a 1D vector of size N, it is copied/resized to
be size (M x N) implicitly and added to each vector in the batch.
Each of these dimensions must be matched correctly, or else the operator
will throw errors.
)DOC")
    .Attr(
        "axis",
        "(int32_t) default to 1; describes the axis of the inputs; "
        "defaults to one because the 0th axis most likely describes "
        "the batch_size",
        AttrType::INT)
    .Attr(
        "axis_w",
        "(int32_t) default to 1; describes the axis of the weights; "
        "defaults to one because the 0th axis most likely describes "
        "the batch_size",
        AttrType::INT)
    .Input(
        0,
        "X",
        "input tensor that's coerced into a 2D matrix of size (MxK) "
        "as described above",
        "T")
    .Input(
        1,
        "W",
        "2D blob of size (KxN) containing fully connected weight "
        "matrix",
        "T")
    .Input(2, "bias", "1D blob containing bias vector", "T")
    .Output(0, "Y", "2D output tensor", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(LpNormalization)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(1)
    .NumOutputs(1)
    .Input(0, "input", "Input matrix", "T")
    .Output(0, "output", "Matrix after normalization", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.")
    .SetDoc(R"DOC(
Given a matrix, apply Lp-normalization along the provided axis.
)DOC")
    .Attr("axis", "(int64, default -1) the axis on which to apply normalization, -1 mean last axis.", AttrType::INT)
    .Attr("P", "(float, default 2.0) the order of the normalization, only 2.0 is supported.", AttrType::FLOAT);

OPERATOR_SCHEMA(Scale)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(1)
    .NumOutputs(1)
    .Input(0, "input", "Input data to be scaled", "T")
    .Output(0, "output", "Output data after scaling", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.")
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
Scale takes one input data (Tensor<float>) and produces one output data
(Tensor<float>) whose value is the input data tensor scaled element-wise.
)DOC")
    .Attr("scale", "(float, default 1.0) the scale to apply.", AttrType::FLOAT);

OPERATOR_SCHEMA(GRUUnit)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(4)
    .NumOutputs(1)
    .SetDoc(R"DOC(
GRUUnit computes the activations of a standard GRU,
in a sequence-length aware fashion.
Concretely, given the (fused) inputs X (TxNxD), the previous hidden
state (NxD), and the sequence lengths (N), computes the GRU
activations, avoiding computation if the input is invalid (as in, the
value at X[t][n] >= seqLengths[n].
)DOC")
    .Attr(
        "drop_states",
        "Bool to determine if hidden state is zeroes or passed "
        "along for timesteps past the given sequence_length.",
        AttrType::INT)
    .Input(0, "hidden_prev", "The previous GRU hidden state.", "T")
    .Input(
        1,
        "gates",
        "Unactivated gate outputs from forget, update, "
        "and output gates, pre-activation.",
        "T")
    .Input(
        2,
        "seq_lengths",
        "Array of sequence lengths.  "
        "len(seq_lengths) should equal batch size N.",
        "T")
    .Input(3, "t", "The timestep for this operation.", "T")
    .Output(0, "hidden", "The new GRU hidden state calculated by this op.", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(ATen)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .AllowUncheckedAttributes()
    .SetDoc(R"DOC(
Experimental allowing ATen operations to be accessed directly from Caffe2
to allow for quick prototyping when ONNX is missing standard versions of
and op)DOC");


OPERATOR_SCHEMA(ImageScaler)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(Scale and bias the input image. Bias values are stored in 
the same ordering as the image pixel format.)DOC")
    .Attr("bias", "Bias applied to each channel, same size as C.", AttrType::FLOATS)
    .Attr("scale", "(float, default 1.0) the scale to apply.", AttrType::FLOAT)
    .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
    .Output(0, "output", "Result, has same shape and type as input", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(MeanVarianceNormalization)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(Perform mean variance normalization.)DOC")
    .Attr("across_channels", "If 1, mean and variance are computed across channels. Default is 0.", AttrType::INT)
    .Attr("normalize_variance", "If 0, normalize the mean only.  Default is 1.", AttrType::INT)    
    .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
    .Output(0, "output", "Result, has same shape and type as input", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    
OPERATOR_SCHEMA(Crop)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(Crop and image to the specified spatial dimensions. If scale is given, 
then optionally start the crop offset by the left/top border amounts. 
If scale is not provided, crop the borders as provided.)DOC")
    .Attr("border", "A 1-D values of (leftBorder, topBorder, rightBorder, bottomBorder).", AttrType::INTS)
    .Attr("scale", "A 1-D values of (height, width).", AttrType::INTS)
    .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
    .Output(0, "output", "Result, has same type as input, with H and W dimensions reduced.", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(Embedding)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(Turns positive integers (indexes) into dense vectors of fixed size.)DOC")
    .Attr("input_dim", "Size of the input vocabulary.", AttrType::INT)
    .Attr("output_dim", "Dimension of the embedding output vectors.", AttrType::INT)
    .Attr("W", "2-D tensor of weights [O,I].", AttrType::TENSOR)    
    .Input(0, 
           "input", 
           "1-D tensor of integers representing indices in the embedding dictionary "
           "with length [N] and values [0, input_dim -1]", "tensor(int64)")
    .Output(0, 
            "output", 
            "Output tensor of computed features [N, O].", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain output types to float tensors.");
