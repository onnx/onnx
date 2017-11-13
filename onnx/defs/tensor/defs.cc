// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

using AttrType = onnx::OpSchema::AttrType;
using namespace onnx;

OPERATOR_SCHEMA(Cast)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message. If the 'to' argument
is not provided or is not one of the enumerated types in DataType, Caffe2
throws an Enforce error.

NOTE: Casting to and from strings is not supported yet.
)DOC")
    .Attr(
          "to",
          "The data type to which the elements of the input tensor are cast."
          "Strictly must be one of the types from DataType enum in TensorProto",
          AttrType::STRING)
    .Input(0, "input", "Input tensor to be cast.", "T1")
    .Output(
        0,
        "output",
        "Output tensor with the same shape as input with type "
        "specified by the 'to' argument",
        "T2")
    .TypeConstraint("T1", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input types to float tensors.")
    .TypeConstraint("T2", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain output types to float tensors.");
            

OPERATOR_SCHEMA(Reshape)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
Reshape the input tensor similar to numpy.reshape.

It takes a tensor as input and an argument `shape`. It outputs the reshaped tensor.

At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is going to be copied
from the shape argument.)DOC")
    .Attr("shape", "New shape", AttrType::INTS)
    .Input(0, "data", "An input tensor.", "T")
    .Output(0, "reshaped", "Reshaped data.", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
	            "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(Concat)
.NumInputs(1, INT_MAX)
.NumOutputs(1)
.Attr("axis",
    "Which axis to concat on",
    AttrType::INT)
    .SetDoc("Concatenate a list of tensors into a single tensor")
    .Input(0, "inputs...", "List of tensors for concatenation", "T")
    .Output(0, "concat_result", "Concatenated tensor", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain output types to float tensors.");

OPERATOR_SCHEMA(Split)
    .NumInputs(1, 2)
    .NumOutputs(1, INT_MAX)
    .Input(0, "input", "The tensor to split", "T")
    .Input(1, "split", "Optional list of output lengths (see also arg 'split')", "T")
    .Output(0, "outputs...", "One or more outputs forming list of tensors after splitting", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input types to float tensors.")
    .Attr("axis",
          "Which axis to split on",
          AttrType::INT)
    .Attr("split",
          "length of each output",
          AttrType::INTS)
    .SetDoc(R"DOC(Split a tensor into a list of tensors, along the specified
'axis'. The lengths of the split can be specified using argument 'axis' or
optional second input blob to the operator. Otherwise, the tensor is split
to equal sized parts.
)DOC");

OPERATOR_SCHEMA(Slice)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html

Slices uses `axes`, `starts` and `ends` attributes to specify the start and end
dimension for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represent number of elements before the end of that
dimension.

Example 1:

  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]

  result = [
      [5, 6, 7],
  ]


Example 2:

  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0]
  ends = [-1]

  result = [
      [1, 2, 3, 4],
  ]

)DOC")
    .Input(0, "data", "Tensor of data to extract slices from.", "T")
    .Attr("axes",
          "Axes that `starts` and `ends` apply to. "
          "It's optional. If not present, will be treated as "
          "[0, 1, ..., len(`starts`) - 1].",
          AttrType::INTS)
    .Attr("starts",
          "Starting indices of corresponding axis in `axes`",
          AttrType::INTS,
          true)
    .Attr("ends",
          "Ending indices (exclusive) of corresponding axis in axes`",
          AttrType::INTS,
          true)
    .Output(0, "output", "Sliced data tensor.", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(Transpose)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Transpose the input tensor similar to numpy.transpose. For example, when
axes=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
will be (2, 1, 3).
)DOC")
    .Attr("perm",
          "A list of integers. By default, reverse the dimensions, "
          "otherwise permute the axes according to the values given.",
          AttrType::INTS)
    .Input(0, "data", "An input tensor.", "T")
    .Output(0, "transposed", "Transposed output.", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(Gather)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given DATA tensor of rank r >= 1, and INDICES tensor of rank q, gather
entries of the outer-most dimension of DATA indexed by INDICES, and concatenate
them in an output tensor of rank q + (r - 1).

Example:
  DATA  = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  INDICES = [
      [0, 1],
      [1, 2],
  ]
  OUTPUT = [
      [
          [1.0, 1.2],
          [2.3, 3.4],
      ],
      [
          [2.3, 3.4],
          [4.5, 5.7],
      ],
  ]
)DOC")
    .Input(0, "DATA", "Tensor of rank r >= 1.", "T")
    .Input(1, "INDICES", "Tensor of int32/int64 indices, of any rank q.", "T")
    .Output(0, "OUTPUT", "Tensor of rank q + (r - 1).", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(Squeeze)
    .NumInputs(1)
    .NumOutputs(1)
    .Attr("axes",
          "List of positive integers, indicate the dimensions to squeeze.",
          AttrType::INTS,
          true)
    .SetDoc(R"DOC(
Remove single-dimensional entries from the shape of a tensor.
Takes a  parameter `axes` with a list of axes to squeeze.
)DOC")
    .Input(0, "data", "Tensors with at least max(dims) dimensions.", "T")
    .Output(0, "squeezed", "Reshaped tensor with same data as input.", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(Pad)
    .NumInputs(1)
    .NumOutputs(1)
    .Attr("paddings",
          "List of integers indicate the padding sizes, paddings's length"
          " should be the double of input's dimension. "
          "The order should be axis_0_begin, axis_0_end, axis_1_begin, ...,"
          " axis_n_begin, axis_n_end, n is input's dimension.",
          AttrType::INTS,
          true)
    .Attr("mode",
          "Three modes: constant(default), reflect, edge",
          AttrType::STRING)
    .Attr("value",
          "One float, indicates the value to be filled, default is 0",
          AttrType::FLOAT)
    .SetDoc(R"DOC(
Given DATA tensor, paddings, mode, and value.

Example:
  Insert 0 paddings to the beginning of the second dimension.

  DATA  = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  paddings = [0, 0, 2, 0]

  OUTPUT = [
      [
          [0.0, 0.0, 1.0, 1.2],
          [0.0, 0.0, 2.3, 3.4],
          [0.0, 0.0, 4.5, 5.7],
      ],
  ]
)DOC")
    .Input(0, "DATA", "Input tensor.", "T")
    .Output(0, "OUTPUT", "Tensor after padding.", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(SpaceToDepth)
    .NumInputs(1)
    .NumOutputs(1)
    .Attr("blocksize",
          "Blocks of [blocksize, blocksize] are moved.",
          AttrType::INT)
    .SetDoc(R"DOC(SpaceToDepth rearranges blocks of spatial data into depth. More specifically, 
this op outputs a copy of the input tensor where values from the height and width dimensions 
are moved to the depth dimension.
)DOC")
    .Input(0,
           "input", 
           "Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth"
           ", H is the height and W is the width.", "T")
    .Output(0,
            "output", 
            "Output tensor of [N, C * blocksize * blocksize, H/blocksize, W/blocksize].", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input types to float tensors.");

OPERATOR_SCHEMA(DepthToSpace)
    .NumInputs(1)
    .NumOutputs(1)
    .Attr("blocksize",
          "Blocks of [blocksize, blocksize] are moved.",
          AttrType::INT)
    .SetDoc(R"DOC(DepthToSpace rearranges (permutes) data from depth into blocks of spatial data. 
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of 
the input tensor where values from the depth dimension are moved in spatial blocks to the height 
and width dimensions.
)DOC")
    .Input(0,
           "input", 
           "Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth"
           ", H is the height and W is the width.", "T")
    .Output(0,
            "output", 
            "Output tensor of [N, C/(blocksize * blocksize), H * blocksize, W * blocksize].", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input types to float tensors.");

OPERATOR_SCHEMA(Tile)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(Repeat the elements of a tensor along an axis.)DOC")
    .Input(0,
           "input",
           "Input tensor of any shape.", "T")
    .Input(1,
           "tiles",
           "Number of repeated copies to make of the input tensor.", "T") 
    .Input(2,
           "axis",
           "Axis along which to repeat.", "T")
    .Output(0,
            "output",
            "Output tensor of same shape and type as input.", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input types to float tensors.");
        