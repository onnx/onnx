// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
using namespace onnx;

using AttrType = onnx::OpSchema::AttrType;

OPERATOR_SCHEMA(Constant)
    .SetDoc(R"DOC(A constant tensor.)DOC")
    .Attr(
          "value",
          "The value for the elements of the output tensor.",
          AttrType::TENSOR)
    .Output(
            0,
            "output",
            "Output tensor containing the same value of the provided tensor.", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(RandomUniform)
    .SetDoc(R"DOC(
Generate a tensor with random values drawn from a uniform distribution. The shape
of the tensor is specified by the `shape` argument and the range by `low` and `high`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
)DOC")
    .Attr(
          "low",
          "Lower boundary of the output values.",
          AttrType::FLOAT)
    .Attr(
          "high",
          "Upper boundary of the output values.",
          AttrType::FLOAT)
    .Attr(
          "seed",
          "(Optional) Seed to the random generator, if not specified we will auto generate one.",
          AttrType::FLOAT)
    .Attr(
          "dtype",
          "The data type for the elements of the output tensor.",
          AttrType::INT)
    .Attr(
          "shape",
          "The shape of the output tensor.",
          AttrType::INTS)
    .Output(
            0,
            "output",
            "Output tensor of random values drawn from uniform distribution", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(RandomNormal)
    .SetDoc(R"DOC(
Generate a tensor with random values drawn from a normal distribution. The shape
of the tensor is specified by the `shape` argument and the parameter of the normal distribution
specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
)DOC")
    .Attr(
          "mean",
          "The mean of the normal distribution.",
          AttrType::FLOAT)
    .Attr(
          "scale",
          "The standard deviation of the normal distribution.",
           AttrType::FLOAT)
    .Attr(
          "seed",
          "(Optional) Seed to the random generator, if not specified we will auto generate one.",
          AttrType::FLOAT)
    .Attr(
          "dtype",
          "The data type for the elements of the output tensor.",
          AttrType::INT)
    .Attr(
          "shape",
          "The shape of the output tensor.",
          AttrType::INTS)
    .Output(
            0,
            "output",
            "Output tensor of random values drawn from normal distribution", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(RandomUniformLike)
    .SetDoc(R"DOC(
Generate a tensor with random values drawn from a uniform distribution. The shape
of the tensor is computed from the input argument and the range by `low` and `high`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
)DOC")
    .Attr(
          "low",
          "Lower boundary of the output values.",
          AttrType::FLOAT)
    .Attr(
          "high",
          "Upper boundary of the output values.",
           AttrType::FLOAT)
    .Attr(
          "seed",
          "(Optional) Seed to the random generator, if not specified we will auto generate one.",
          AttrType::FLOAT)
    .Attr(
          "dtype",
          "(Optional) The data type for the elements of the output tensor, if not specified, we will use"
          "the data type of the input tensor.",
           AttrType::INT)
    .Input(
           0,
           "input",
           "Input tensor to provide shape information.", "T")
    .Output(
            0,
            "output",
            "Output tensor of random values drawn from uniform distribution", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(RandomNormalLike)
    .SetDoc(R"DOC(
Generate a tensor with random values drawn from a normal distribution. The shape
of the tensor is computed from the input argument and the parameter of the normal distribution
specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
)DOC")
    .Attr(
          "mean",
          "The mean of the normal distribution.",
          AttrType::FLOAT)
    .Attr(
          "scale",
          "The standard deviation of the normal distribution.",
          AttrType::FLOAT)
    .Attr(
          "seed",
          "(Optional) Seed to the random generator, if not specified we will auto generate one.",
          AttrType::FLOAT)
    .Attr(
          "dtype",
          "(Optional) The data type for the elements of the output tensor, if not specified, we will use"
          "the data type of the input tensor.",
          AttrType::INT)
    .Input(
           0,
           "input",
           "Input tensor to provide shape information.", "T")
    .Output(
            0,
            "output",
            "Output tensor of random values drawn from normal distribution", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain input and output types to float tensors.");
