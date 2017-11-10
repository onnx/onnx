// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
using namespace onnx;

using AttrType = onnx::OpSchema::AttrType;
using SupportType = onnx::OpSchema::SupportType;

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

OPERATOR_SCHEMA(Normalize)
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
Given a matrix, apply L2-normalization along the last dimension.
)DOC");

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
