// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

using AttrType = onnx::OpSchema::AttrType;

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
    .Input(0, "input", "Input tensor to be cast.")
    .Output(
        0,
        "output",
        "Output tensor with the same shape as input with type "
        "specified by the 'to' argument");

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
    .Input(0, "data", "An input tensor.")
    .Output(0, "reshaped", "Reshaped data.");

OPERATOR_SCHEMA(Concat)
    .NumInputs(1, INT_MAX)
    .NumOutputs(2)
    .Attr("axis",
          "Which axis to concat on",
          AttrType::INT)
    .SetDoc("Concatenate a list of tensors into a single tensor")
    .Output(0, "concat_result", "Concatenated tensor");

OPERATOR_SCHEMA(Split)
    .NumInputs(1, 2)
    .NumOutputs(1, INT_MAX)
    .Input(0, "input", "The tensor to split")
    .Input(1, "split", "Optional list of output lengths (see also arg 'split')")
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
    .NumInputs(4)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html 

Slices uses `axes`, `starts` and `ends` list to specify the start and end dimension 
for each axis in the list of axes, it uses this information to slice the input `data` 
tensor. If a negative value is passed for any of the start or end indices, it represent 
number of elements before the end of that dimension.
)DOC")
    .Input(0, "data", "Tensor of data to extract slices from.")
    .Input(1, "axes", "1D Tensor contains the list of axes in which starts and ends apply to.")
    .Input(2, "starts", "1D Tensor contains the list of indices starting values corresponding to each axes in the axes input.")
    .Input(3, "ends", "1D Tensor contains the list of indices end values corresponding to each axes in the axes input.")            
    .Output(0, "output", "Sliced data tensor.");

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
    .Input(0, "data", "An input tensor.")
    .Output(0, "transposed", "Transposed output.");

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
    .Input(0, "DATA", "Tensor of rank r >= 1.")
    .Input(1, "INDICES", "Tensor of int32/int64 indices, of any rank q.")
    .Output(0, "OUTPUT", "Tensor of rank q + (r - 1).");

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
    .Input(0, "data", "Tensors with at least max(dims) dimensions.")
    .Output(0, "squeezed", "Reshaped tensor with same data as input.");
