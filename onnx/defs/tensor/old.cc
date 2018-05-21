// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

using namespace ONNX_NAMESPACE;

ONNX_OPERATOR_SCHEMA(Cast)
    .SetDoc(R"DOC(
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.
NOTE: Casting to and from strings is not supported yet.
)DOC")
    .Attr(
        "to",
        "The data type to which the elements of the input tensor are cast."
        "Strictly must be one of the types from DataType enum in TensorProto",
        AttributeProto::STRING)
    .Input(0, "input", "Input tensor to be cast.", "T1")
    .Output(
        0,
        "output",
        "Output tensor with the same shape as input with type "
        "specified by the 'to' argument",
        "T2")
    .TypeConstraint(
        "T1",
        {"tensor(float16)",
         "tensor(float)",
         "tensor(double)",
         "tensor(int8)",
         "tensor(int16)",
         "tensor(int32)",
         "tensor(int64)",
         "tensor(uint8)",
         "tensor(uint16)",
         "tensor(uint32)",
         "tensor(uint64)",
         "tensor(bool)"},
        "Constrain input types. Casting from strings and complex are not supported.")
    .TypeConstraint(
        "T2",
        {"tensor(float16)",
         "tensor(float)",
         "tensor(double)",
         "tensor(int8)",
         "tensor(int16)",
         "tensor(int32)",
         "tensor(int64)",
         "tensor(uint8)",
         "tensor(uint16)",
         "tensor(uint32)",
         "tensor(uint64)",
         "tensor(bool)"},
        "Constrain output types. Casting to strings and complex are not supported.");

ONNX_OPERATOR_SCHEMA(Concat)
    .Attr(
        "axis",
        "Which axis to concat on.  Default value is 1.",
        AttributeProto::INT,
        OPTIONAL)
    .SetDoc("Concatenate a list of tensors into a single tensor")
    .Input(
        0,
        "inputs",
        "List of tensors for concatenation",
        "T",
        OpSchema::Variadic)
    .Output(0, "concat_result", "Concatenated tensor", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Split)
    .SinceVersion(1)
    .Input(0, "input", "The tensor to split", "T")
    .Input(
        1,
        "split",
        "Optional list of output lengths (see also arg 'split')",
        "T",
        OpSchema::Optional)
    .Output(
        0,
        "outputs...",
        "One or more outputs forming list of tensors after splitting",
        "T",
        OpSchema::Variadic)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input types to float tensors.")
    .Attr("axis", "Which axis to split on", AttributeProto::INT, OPTIONAL)
    .Attr("split", "length of each output", AttributeProto::INTS, OPTIONAL)
    .SetDoc(R"DOC(Split a tensor into a list of tensors, along the specified
'axis'. The lengths of the split can be specified using argument 'axis' or
optional second input blob to the operator. Otherwise, the tensor is split
to equal sized parts.
)DOC");

ONNX_OPERATOR_SCHEMA(Pad)
    .SinceVersion(1)
    .Attr(
        "paddings",
        "List of integers indicate the padding element count at the "
        "beginning and end of each axis, for 2D it is the number of pixel. "
        "`paddings` rank should be double of the input's rank. `paddings` format should be as follow "
        "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
        "added at the beginning of axis `i` and xi_end, the number of pixels added at "
        "the end of axis `i`.",
        AttributeProto::INTS)
    .Attr(
        "mode",
        "Three modes: constant(default), reflect, edge",
        AttributeProto::STRING,
        std::string("constant"))
    .Attr(
        "value",
        "One float, indicates the value to be filled, default is 0",
        AttributeProto::FLOAT,
        0.0f)
    .SetDoc(R"DOC(
Given `data` tensor, paddings, mode, and value.

Example:
  Insert 0 paddings to the beginning of the second dimension.

  data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  paddings = [0, 0, 2, 0]

  output = [
      [
          [0.0, 0.0, 1.0, 1.2],
          [0.0, 0.0, 2.3, 3.4],
          [0.0, 0.0, 4.5, 5.7],
      ],
  ]
)DOC")
    .Input(0, "data", "Input tensor.", "T")
    .Output(0, "output", "Tensor after padding.", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Reshape)
    .SetDoc(R"DOC(
Reshape the input tensor similar to numpy.reshape.
It takes a tensor as input and an argument `shape`. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor).)DOC")
    .Attr("shape", "New shape", AttributeProto::INTS, OPTIONAL)
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .Input(0, "data", "An input tensor.", "T")
    .Output(0, "reshaped", "Reshaped data.", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Upsample)
    .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
    .Attr(
        "width_scale",
        "The scale along width dimension. It takes value greater than or equal to 1.",
        AttributeProto::FLOAT)
    .Attr(
        "height_scale",
        "The scale along height dimension. It takes value greater than or equal to 1.",
        AttributeProto::FLOAT)
    .Attr(
        "mode",
        "Two interpolation modes: nearest(default), bilinear",
        AttributeProto::STRING,
        std::string("nearest"))
    .Input(0, "X", "4-D tensor, [N,C,H,W]", "T")
    .Output(0, "Y", "4-D tensor after resizing, [N,C,H,W]", "T")
    .TypeConstraint(
        "T",
        {"tensor(bool)",
         "tensor(int32)",
         "tensor(int64)",
         "tensor(float16)",
         "tensor(float)",
         "tensor(double)"},
        "Constrain output types to bool, int32, int64, float16, float, double tensors.")
    .SetDoc(R"DOC(
Upsample the input tensor.
The width and height of the output tensor are:
  output_width = floor(input_width * width_scale),
  output_height = floor(input_height * height_scale).
Example:
  Given `data` tensor, width_scale, height_scale, mode,
  Upsample the input 4-D tensor in nearest mode:
  data = [[[
      [1, 2],
      [3, 4]
  ]]]
  width_scale = 2
  height_scale = 2
  mode = "nearest"
  output = [[[
      [1, 1, 2, 2],
      [1, 1, 2, 2],
      [3, 3, 4, 4],
      [3, 3, 4, 4]
  ]]]
)DOC");
