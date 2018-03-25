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

ONNX_OPERATOR_SCHEMA(Reshape)
    .SinceVersion(6)
    .SetDoc(R"DOC(
Reshape the input tensor similar to numpy.reshape.

First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.

At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor).)DOC")
    .Input(0, "data", "An input tensor.", "T")
    .Input(1, "shape", "Specified shape for output.", "tensor(int64)")
    .SinceVersion(5)
    .Output(0, "reshaped", "Reshaped data.", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Shape)
    .SetDoc(R"DOC(
Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
)DOC")
    .Input(0, "data", "An input tensor.", "T")
    .Output(0, "shape", "Shape of the input tensor", "T1")
    .TypeConstraint(
        "T",
        {"tensor(float16)",
         "tensor(float)",
         "tensor(double)",
         "tensor(int8)",
         "tensor(int16)",
         "tensor(int32)",
         "tensor(int64)",
         "tensor(uint8)",
         "tensor(uint16)",
         "tensor(bool)"},
        "Input tensor can be of arbitrary type.")
    .TypeConstraint(
        "T1",
        {"tensor(int64)"},
        "Constrains output to int64 tensor.");

ONNX_OPERATOR_SCHEMA(Size)
    .SetDoc(R"DOC(
Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
)DOC")
    .Input(0, "data", "An input tensor.", "T")
    .Output(0, "size", "Total number of elements of the input tensor", "T1")
    .TypeConstraint(
        "T",
        {"tensor(float16)",
         "tensor(float)",
         "tensor(double)",
         "tensor(int8)",
         "tensor(int16)",
         "tensor(int32)",
         "tensor(int64)",
         "tensor(uint8)",
         "tensor(uint16)",
         "tensor(bool)"},
        "Input tensor can be of arbitrary type.")
    .TypeConstraint("T1", {"int64"}, "Constrains output to int64 scalar.");

ONNX_OPERATOR_SCHEMA(Concat)
    .SinceVersion(4)
    .Attr("axis", "Which axis to concat on", AttributeProto::INT)
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
    .SinceVersion(2)
    .Input(0, "input", "The tensor to split", "T")
    .Output(
        0,
        "outputs",
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
'axis'. Lengths of the parts can be specified using argument 'split'.
Otherwise, the tensor is split to equal sized parts.
)DOC");

ONNX_OPERATOR_SCHEMA(Slice)
    .SetDoc(R"DOC(
Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html

Slices uses `axes`, `starts` and `ends` attributes to specify the start and end
dimension for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represent number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.

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
  starts = [0, 1]
  ends = [-1, 1000]

  result = [
      [2, 3, 4],
  ]

)DOC")
    .Input(0, "data", "Tensor of data to extract slices from.", "T")
    .Attr(
        "axes",
        "Axes that `starts` and `ends` apply to. "
        "It's optional. If not present, will be treated as "
        "[0, 1, ..., len(`starts`) - 1].",
        AttributeProto::INTS,
        OPTIONAL)
    .Attr(
        "starts",
        "Starting indices of corresponding axis in `axes`",
        AttributeProto::INTS)
    .Attr(
        "ends",
        "Ending indices (exclusive) of corresponding axis in axes`",
        AttributeProto::INTS)
    .Output(0, "output", "Sliced data tensor.", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Transpose)
    .SetDoc(R"DOC(
Transpose the input tensor similar to numpy.transpose. For example, when
perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
will be (2, 1, 3).
)DOC")
    .Attr(
        "perm",
        "A list of integers. By default, reverse the dimensions, "
        "otherwise permute the axes according to the values given.",
        AttributeProto::INTS,
        OPTIONAL)
    .Input(0, "data", "An input tensor.", "T")
    .Output(0, "transposed", "Transposed output.", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.")
    .ShapeInferenceFunction([](InferenceContext& ctx) {
      if (ctx.getNumInputTypes() != 1) {
        return;
      }
      std::vector<int64_t> perm;
      {
        auto perm_attr = ctx.getAttribute("perm");
        if (perm_attr) {
          for (auto dim : perm_attr->ints()) {
            perm.push_back(dim);
          }
        } else {
          int ndims = ctx.getInputType(0)->shape().dim_size();
          for (int i = 0; i < ndims; i++) {
            perm.push_back(ndims - i - 1);
          }
        }
      }

      TypeProto_Tensor tt;
      {
        tt.set_elem_type(ctx.getInputType(0)->elem_type());
        auto shape = tt.mutable_shape();

        for (size_t i = 0; i < perm.size(); i++) {
          auto dim = shape->add_dim();
          *dim = ctx.getInputType(0)->shape().dim(perm[i]);
        }
      }

      *ctx.getOutputType(0) = tt;
    });

ONNX_OPERATOR_SCHEMA(Gather)
    .SetDoc(R"DOC(
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
them in an output tensor of rank q + (r - 1).

Example 1:
  data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  indices = [
      [0, 1],
      [1, 2],
  ]
  output = [
      [
          [1.0, 1.2],
          [2.3, 3.4],
      ],
      [
          [2.3, 3.4],
          [4.5, 5.7],
      ],
  ]

Example 2:
  data = [
      [1.0, 1.2, 1.9],
      [2.3, 3.4, 3.9],
      [4.5, 5.7, 5.9],
  ]
  indices = [
      [0, 2],
  ]
  axis = 1,
  output = [
      [
          [1.0, 1.9],
          [2.3, 3.9],
          [4.5, 5.9],
      ],
  ]
)DOC")
    .Attr(
        "axis",
        "Which axis to gather on, defaults to 0. Negative value means "
        "counting dimensions from the back. Accepted range in [-r, r-1]",
        AttributeProto::INT,
        static_cast<int64_t>(0))
    .Input(0, "data", "Tensor of rank r >= 1.", "T")
    .Input(
        1,
        "indices",
        "Tensor of int32/int64 indices, of any rank q.",
        "Tind")
    .Output(0, "output", "Tensor of rank q + (r - 1).", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.")
    .TypeConstraint(
        "Tind",
        {"tensor(int32)", "tensor(int64)"},
        "Constrain indices to integer types");

ONNX_OPERATOR_SCHEMA(Squeeze)
    .Attr(
        "axes",
        "List of positive integers, indicate the dimensions to squeeze.",
        AttributeProto::INTS)
    .SetDoc(R"DOC(
Remove single-dimensional entries from the shape of a tensor.
Takes a  parameter `axes` with a list of axes to squeeze.
)DOC")
    .Input(0, "data", "Tensors with at least max(dims) dimensions.", "T")
    .Output(0, "squeezed", "Reshaped tensor with same data as input.", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Unsqueeze)
    .Attr(
        "axes",
        "List of positive integers, indicate the dimensions to be inserted",
        AttributeProto::INTS)
    .SetDoc(R"DOC(
Insert single-dimensional entries to the shape of a tensor.
Takes one required argument `axes`, a list of dimensions that will be inserted.
Dimension indices in `axes` are as seen in the output tensor. For example:

  Given a tensor such that tensor with shape [3, 4, 5], then
  Unsqueeze(tensor, axes=[0, 4]) has shape [1, 3, 4, 5, 1]

)DOC")
    .Input(0, "data", "Original tensor", "T")
    .Output(0, "expanded", "Reshaped tensor with same data as input.", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Pad)
    .SinceVersion(2)
    .Attr(
        "pads",
        "List of integers indicate the padding element count at the "
        "beginning and end of each axis, for 2D it is the number of pixel. "
        "`pads` rank should be double of the input's rank. `pads` format should be as follow "
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
Given `data` tensor, pads, mode, and value.

Example:
  Insert 0 pads to the beginning of the second dimension.

  data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  pads = [0, 2, 0, 0]

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

ONNX_OPERATOR_SCHEMA(SpaceToDepth)
    .Attr(
        "blocksize",
        "Blocks of [blocksize, blocksize] are moved.",
        AttributeProto::INT,
        OPTIONAL)
    .SetDoc(
        R"DOC(SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
this op outputs a copy of the input tensor where values from the height and width dimensions
are moved to the depth dimension.
)DOC")
    .Input(
        0,
        "input",
        "Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth"
        ", H is the height and W is the width.",
        "T")
    .Output(
        0,
        "output",
        "Output tensor of [N, C * blocksize * blocksize, H/blocksize, W/blocksize].",
        "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input types to float tensors.");

ONNX_OPERATOR_SCHEMA(DepthToSpace)
    .Attr(
        "blocksize",
        "Blocks of [blocksize, blocksize] are moved.",
        AttributeProto::INT,
        OPTIONAL)
    .SetDoc(
        R"DOC(DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions.
)DOC")
    .Input(
        0,
        "input",
        "Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth"
        ", H is the height and W is the width.",
        "T")
    .Output(
        0,
        "output",
        "Output tensor of [N, C/(blocksize * blocksize), H * blocksize, W * blocksize].",
        "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input types to float tensors.");

ONNX_OPERATOR_SCHEMA(Tile)
    .SetDoc(R"DOC(Repeat the elements of a tensor along an axis.)DOC")
    .Input(0, "input", "Input tensor of any shape.", "T")
    .Input(
        1,
        "tiles",
        "Number of repeated copies to make of the input tensor.",
        "T")
    .Input(2, "axis", "Axis along which to repeat.", "T")
    .Output(0, "output", "Output tensor of same shape and type as input.", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input types to float tensors.");
