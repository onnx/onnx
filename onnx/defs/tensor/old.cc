// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

using namespace onnx;

OPERATOR_SCHEMA(Split)
    .SinceVersion(1)
    .Input(0, "input", "The tensor to split", "T")
    .Input(1, "split", "Optional list of output lengths (see also arg 'split')", "T", OpSchema::Optional)
    .Output(0, "outputs...", "One or more outputs forming list of tensors after splitting", "T", OpSchema::Variadic)
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input types to float tensors.")
    .Attr("axis",
          "Which axis to split on",
          AttributeProto::INT)
    .Attr("split",
          "length of each output",
          AttributeProto::INTS)
    .SetDoc(R"DOC(Split a tensor into a list of tensors, along the specified
'axis'. The lengths of the split can be specified using argument 'axis' or
optional second input blob to the operator. Otherwise, the tensor is split
to equal sized parts.
)DOC");

OPERATOR_SCHEMA(Pad)
    .SinceVersion(1)
    .Attr("paddings",
          "List of integers indicate the padding element count at the "
          "begining and end of each axis, for 2D it is the number of pixel. "
          "`paddings` rank should be double of the input's rank. `paddings` format should be as follow "
          "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
          "added at the begining of axis `i` and xi_end, the number of pixels added at "
          "the end of axis `i`.",
          AttributeProto::INTS,
          true)
    .Attr("mode",
          "Three modes: constant(default), reflect, edge",
          AttributeProto::STRING)
    .Attr("value",
          "One float, indicates the value to be filled, default is 0",
          AttributeProto::FLOAT)
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
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");
