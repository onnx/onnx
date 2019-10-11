// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include <algorithm>
#include <cmath>
#include <numeric>
#include "onnx/defs/tensor/utils.h"

namespace ONNX_NAMESPACE {
static const char* Cast_ver1_doc = R"DOC(
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.
NOTE: Casting to and from strings is not supported yet.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    1,
    OpSchema()
        .SetDoc(Cast_ver1_doc)
        .Attr(
            "to",
            "The data type to which the elements of the input tensor are cast. "
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
            "Constrain output types. Casting to strings and complex are not supported."));

static const char* Cast_ver6_doc = R"DOC(
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.
NOTE: Casting to and from strings is not supported yet.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    6,
    OpSchema()
        .SetDoc(Cast_ver6_doc)
        .Attr(
            "to",
            "The data type to which the elements of the input tensor are cast. "
            "Strictly must be one of the types from DataType enum in TensorProto",
            AttributeProto::INT)
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
            "Constrain output types. Casting to strings and complex are not supported.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

static const char* Concat_ver1_doc =
    R"DOC(Concatenate a list of tensors into a single tensor)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Concat,
    1,
    OpSchema()
        .Attr(
            "axis",
            "Which axis to concat on.  Default value is 1.",
            AttributeProto::INT,
            OPTIONAL)
        .SetDoc(Concat_ver1_doc)
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
            "Constrain output types to float tensors."));

ONNX_OPERATOR_SET_SCHEMA(
    Concat,
    4,
    OpSchema()
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
            OpSchema::all_tensor_types(),
            "Constrain output types to any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          auto numInputs = ctx.getNumInputs();
          if (numInputs < 1 ||
              !hasNInputShapes(ctx, static_cast<int>(numInputs))) {
            return;
          }

          auto rank = ctx.getInputType(0)->tensor_type().shape().dim_size();

          auto axisAttr = ctx.getAttribute("axis");
          if (!axisAttr) {
            fail_shape_inference("Required attribute axis is missing");
          }
          int axis = static_cast<int>(axisAttr->i());
          if (rank <= axis) {
            fail_shape_inference("rank must be greater than axis");
          }
          if (axis < 0) {
            return; // TODO: check if negative axis must be supported
          }

          bool all_lengths_known = true;
          int total_length = 0;

          auto* output_shape =
              ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

          for (int64_t i = 0; i < rank; ++i) {
            output_shape->add_dim();
          }

          for (size_t i = 0; i < numInputs; i++) {
            const auto& shape = ctx.getInputType(i)->tensor_type().shape();
            if (shape.dim_size() != rank)
              fail_shape_inference("All inputs to Concat must have same rank");
            for (int j = 0; j < rank; j++) {
              if (j == axis) {
                if (shape.dim(j).has_dim_value()) {
                  total_length += static_cast<int>(shape.dim(j).dim_value());
                } else {
                  all_lengths_known = false;
                }
              } else {
                auto& output_dim = *output_shape->mutable_dim(j);
                const auto& input_dim = shape.dim(j);
                mergeInDimensionInfo(input_dim, output_dim, j);
              }
            }
          }

          if (all_lengths_known) {
            output_shape->mutable_dim(axis)->set_dim_value(total_length);
          }
        }));

static const char* Split_ver1_doc =
    R"DOC(Split a tensor into a list of tensors, along the specified
'axis'. The lengths of the split can be specified using argument 'axis' or
optional second input blob to the operator. Otherwise, the tensor is split
to equal sized parts.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Split,
    1,
    OpSchema()
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
        .SetDoc(Split_ver1_doc));

static const char* Pad_ver1_doc = R"DOC(
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
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Pad,
    1,
    OpSchema()
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
        .SetDoc(Pad_ver1_doc)
        .Input(0, "data", "Input tensor.", "T")
        .Output(0, "output", "Tensor after padding.", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Reshape_ver1_doc = R"DOC(
Reshape the input tensor similar to numpy.reshape.
It takes a tensor as input and an argument `shape`. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor).)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    1,
    OpSchema()
        .SetDoc(Reshape_ver1_doc)
        .Attr("shape", "New shape", AttributeProto::INTS, OPTIONAL)
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
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
            "Constrain input and output types to float tensors."));

static const char* Upsample_ver1_doc = R"DOC(
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
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Tile,
    1,
    OpSchema()
        .SetDoc("Repeat the elements of a tensor along an axis.")
        .Input(0, "input", "Input tensor of any shape.", "T")
        .Input(
            1,
            "tiles",
            "Number of repeated copies to make of the input tensor.",
            "T")
        .Input(2, "axis", "Axis along which to repeat.", "T")
        .Output(
            0,
            "output",
            "Output tensor of same shape and type as input.",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input types to float tensors.")
        .TypeConstraint(
            "T1",
            {"tensor(int64)"},
            "Constrain tiles and axis's type to int64 tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          // Only rank of output can be inferred. We can do better if second
          // input is a constant, but this requires extending InferenceContext
          // interface to get values of constant inputs.
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Upsample,
    1,
    OpSchema()
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
        .SetDoc(Upsample_ver1_doc));

static const char* Upsample_ver7_doc = R"DOC(
Upsample the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Upsample,
    7,
    OpSchema()
        .Attr(
            "scales",
            "The scale array along each dimension. It takes value greater than or equal to 1."
            " The number of elements of 'scales' should be the same as the rank of input 'X'.",
            AttributeProto::FLOATS)
        .Attr(
            "mode",
            "Two interpolation modes: nearest (default), and linear (including bilinear, trilinear, etc)",
            AttributeProto::STRING,
            std::string("nearest"))
        .Input(0, "X", "N-D tensor", "T")
        .Output(0, "Y", "N-D tensor after resizing", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain input and output types to all tensor types.")
        .SetDoc(Upsample_ver7_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          const auto& input_shape = getInputShape(ctx, 0);
          auto* output_shape = getOutputShape(ctx, 0);
          const auto* scales = ctx.getAttribute("scales");

          if (output_shape->dim_size() > 0) {
            if (output_shape->dim_size() != input_shape.dim_size()) {
              fail_shape_inference(
                  "Ranks inferred (",
                  input_shape.dim_size(),
                  ") is not equal to the existing rank value (",
                  output_shape->dim_size(),
                  ").");
            }
          } else { // Infer the rank of output anyway
            for (int i = 0; i < input_shape.dim_size(); ++i) {
              output_shape->add_dim();
            }
          }

          if (nullptr != scales) {
            // Infer output shape's dimension value if 'scales' is known.
            if (scales->type() == AttributeProto_AttributeType_FLOATS) {
              const std::vector<float> scales_data(
                  scales->floats().begin(), scales->floats().end());
              if (scales_data.size() !=
                  static_cast<size_t>(input_shape.dim_size())) {
                fail_shape_inference(
                    "Number of elements of attribute 'scales' must be same as rank of input 'X'");
              }
              resizeShapeInferenceHelper_opset7_to_10(
                  input_shape, scales_data, output_shape);
            } else {
              fail_shape_inference("Attribute 'scales' must have floats type.");
            } // scales->type() == float
          } else {
            fail_shape_inference("Attribute 'scales' is required.");
          } // nullptr != scales
        }));

static const char* Upsample_ver9_doc = R"DOC(
Upsample the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Upsample,
    9,
    OpSchema()
        .Attr(
            "mode",
            "Two interpolation modes: nearest (default), and linear (including bilinear, trilinear, etc)",
            AttributeProto::STRING,
            std::string("nearest"))
        .Input(0, "X", "N-D tensor", "T")
        .Input(
            1,
            "scales",
            "The scale array along each dimension. It takes value greater than or equal to 1."
            " The number of elements of 'scales' should be the same as the rank of input 'X'.",
            "tensor(float)")
        .Output(0, "Y", "N-D tensor after resizing", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain input 'X' and output 'Y' to all tensor types.")
        .SetDoc(Upsample_ver9_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          resizeShapeInference_opset7_to_10(ctx);
        }));

static const char* Resize_ver10_doc = R"DOC(
Resize the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Resize,
    10,
    OpSchema()
        .Attr(
            "mode",
            "Two interpolation modes: nearest (default), and linear (including bilinear, trilinear, etc)",
            AttributeProto::STRING,
            std::string("nearest"))
        .Input(0, "X", "N-D tensor", "T")
        .Input(
            1,
            "scales",
            "The scale array along each dimension. It takes value greater than 0. If it's less than 1,"
            " it's sampling down, otherwise, it's upsampling. The number of elements of 'scales' should"
            " be the same as the rank of input 'X'.",
            "tensor(float)")
        .Output(0, "Y", "N-D tensor after resizing", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain input 'X' and output 'Y' to all tensor types.")
        .SetDoc(Resize_ver10_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          resizeShapeInference_opset7_to_10(ctx);
        }));

static const char* Slice_ver1_doc = R"DOC(
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
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Slice,
    1,
    OpSchema()
        .SetDoc(Slice_ver1_doc)
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
            OpSchema::all_tensor_types(),
            "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          std::vector<int64_t> starts;
          std::vector<int64_t> ends;
          if (!getRepeatedAttribute(ctx, "starts", starts) ||
              !getRepeatedAttribute(ctx, "ends", ends) ||
              starts.size() != ends.size()) {
            fail_shape_inference(
                "Incorrect or missing attribute value for starts and ends");
            ;
          }
          std::vector<int64_t> axes;
          if (!getRepeatedAttribute(ctx, "axes", axes)) {
            for (int i = 0; (size_t)i < starts.size(); ++i) {
              axes.push_back(i);
            }
          } else if (axes.size() != starts.size()) {
            fail_shape_inference("Attribute axes has incorrect length");
            ;
          } else if (!std::is_sorted(axes.begin(), axes.end())) {
            // TODO support shape inference for unsorted axes
            return;
          }

          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

          for (size_t i = 0, j = 0; (int64_t)i <
               ctx.getInputType(0)->tensor_type().shape().dim_size();
               ++i) {
            auto* newdim = ctx.getOutputType(0)
                               ->mutable_tensor_type()
                               ->mutable_shape()
                               ->add_dim();
            if (j < axes.size() && static_cast<size_t>(axes[j]) == i) {
              // There's a lot of potential behaviors. For now just
              // handle some simple cases.
              if (ctx.getInputType(0)
                      ->tensor_type()
                      .shape()
                      .dim((int)i)
                      .has_dim_value() &&
                  starts[j] >= 0 && ends[j] >= 0) {
                auto newval = std::min(
                                  (int64_t)ctx.getInputType(0)
                                      ->tensor_type()
                                      .shape()
                                      .dim((int)i)
                                      .dim_value(),
                                  ends[j]) -
                    starts[j];
                if (newval >= 0) {
                  newdim->set_dim_value(newval);
                }
              }
              ++j;
            } else {
              *newdim = ctx.getInputType(0)->tensor_type().shape().dim((int)i);
            }
          }
        }));

static const char* Slice_ver10_doc = R"DOC(
Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
Slices uses `starts`, `ends`, `axes` and `steps` inputs to specify the start and end
dimension and step for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represent number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`.
If a negative value is passed for step, it represents slicing backward.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`
Example 1:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  steps = [1, 2]
  result = [
      [5, 7],
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
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Slice,
    10,
    OpSchema()
        .SetDoc(Slice_ver10_doc)
        .Input(0, "data", "Tensor of data to extract slices from.", "T")
        .Input(
            1,
            "starts",
            "1-D tensor of starting indices of corresponding axis in `axes`",
            "Tind")
        .Input(
            2,
            "ends",
            "1-D tensor of ending indices (exclusive) of corresponding axis in `axes`",
            "Tind")
        .Input(
            3,
            "axes",
            "1-D tensor of axes that `starts` and `ends` apply to.",
            "Tind",
            OpSchema::Optional)
        .Input(
            4,
            "steps",
            "1-D tensor of slice step of corresponding axis in `axes`. Default to 1. ",
            "Tind",
            OpSchema::Optional)
        .Output(0, "output", "Sliced data tensor.", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain input and output types to all tensor types.")
        .TypeConstraint(
            "Tind",
            {"tensor(int32)", "tensor(int64)"},
            "Constrain indices to integer types")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          size_t num_inputs = ctx.getNumInputs();
          if (num_inputs != 3 && num_inputs != 4 && num_inputs != 5) {
            fail_type_inference(
                "Slice op must have either three, four or five inputs.");
          }
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          // Shape Inference if
          //     1. 2nd and 3rd input data (starts, ends) are available.
          // and 2. 4th and 5th optional input (axes, steps) are either not set,
          // or set and is initializer.
          const TensorProto* startsInitializer = ctx.getInputData(1);
          const TensorProto* endsInitializer = ctx.getInputData(2);
          const TensorProto* axesInitializer =
              hasInputShape(ctx, 3) ? ctx.getInputData(3) : nullptr;
          const TensorProto* stepsInitializer =
              hasInputShape(ctx, 4) ? ctx.getInputData(4) : nullptr;

          if (!startsInitializer || !endsInitializer ||
              (hasInputShape(ctx, 3) && !ctx.getInputData(3)) ||
              (hasInputShape(ctx, 4) && !ctx.getInputData(4))) {
            return;
          }

          // don't know data_type- can't proceed
          if (!startsInitializer->has_data_type())
            return;

          auto get_initializer_data =
              [](const TensorProto* initializer) -> std::vector<int64_t> {
            std::vector<int64_t> vec;
            if (initializer->data_type() == TensorProto::INT64) {
              const auto& data = ParseData<int64_t>(initializer);
              vec.insert(vec.end(), data.begin(), data.end());
            } else if (initializer->data_type() == TensorProto::INT32) {
              const auto& data = ParseData<int32_t>(initializer);
              vec.insert(vec.end(), data.begin(), data.end());
            } else {
              // unaccepted data type
              fail_shape_inference(
                  "Only supports `int32_t` or `int64_t` inputs for starts/ends/axes/steps");
            }
            return vec;
          };

          auto clamp = [](int64_t val, int64_t low, int64_t high) -> int64_t {
            if (val < low)
              return low;
            if (val > high)
              return high;
            return val;
          };

          std::vector<int64_t> starts = get_initializer_data(startsInitializer);
          std::vector<int64_t> ends = get_initializer_data(endsInitializer);

          if (starts.size() != ends.size()) {
            fail_shape_inference(
                "Incorrect or missing input value for starts and ends");
          }

          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_rank = input_shape.dim_size();
          std::vector<int64_t> axes(starts.size());
          if (!axesInitializer) {
            std::iota(axes.begin(), axes.end(), 0);
          } else {
            axes = get_initializer_data(axesInitializer);
            if (axes.size() != starts.size()) {
              fail_shape_inference("Input axes has incorrect length");
            }
          }

          std::vector<int64_t> steps;
          if (!stepsInitializer) {
            steps = std::vector<int64_t>(starts.size(), 1);
          } else {
            steps = get_initializer_data(stepsInitializer);
            if (steps.size() != axes.size()) {
              fail_shape_inference("Input steps has incorrect length");
            }
          }

          for (size_t i = 0; (int64_t)i < input_rank; ++i) {
            // first update rank of output dim
            auto* output_dim = ctx.getOutputType(0)
                                   ->mutable_tensor_type()
                                   ->mutable_shape()
                                   ->add_dim();
            const auto& input_dim = input_shape.dim((int)i);
            if (input_dim.has_dim_value()) {
              output_dim->set_dim_value(input_dim.dim_value());
            } else if (input_dim.has_dim_param()) {
              output_dim->set_dim_param(input_dim.dim_param());
            }
          }

          std::unordered_set<int64_t> unique_axes;
          size_t axes_size = axes.size();
          for (size_t axis_index = 0; axis_index < axes_size; ++axis_index) {
            auto axis = axes[axis_index] < 0
                ? axes[axis_index] + static_cast<int64_t>(input_rank)
                : axes[axis_index];

            if (axis >= static_cast<int64_t>(input_rank) || axis < 0)
              fail_shape_inference("Input axes has invalid data");

            if (unique_axes.find(axis) != unique_axes.end())
              fail_shape_inference("'axes' has duplicates");

            unique_axes.insert(axis);

            auto input_dim =
                ctx.getInputType(0)->tensor_type().shape().dim((int)axis);

            // input dim value is missing - cannot perform shape inference for
            // this axis
            if (!input_dim.has_dim_value())
              continue;

            const auto input_dim_value = input_dim.dim_value();

            // process step
            auto step = steps[axis_index];
            if (step == 0)
              fail_shape_inference("'step' cannot be 0");

            // process start
            auto start = starts[axis_index];
            if (start < 0)
              start += input_dim_value;
            if (step < 0)
              start = clamp(start, 0, input_dim_value - 1);
            else
              start = clamp(start, 0, input_dim_value);

            // process end
            auto end = ends[axis_index];
            if (end < 0)
              end += input_dim_value;
            if (step < 0)
              end = clamp(end, -1, input_dim_value);
            else
              end = clamp(end, 0, input_dim_value);

            // find output dim value for this axis
            auto temp = static_cast<int64_t>(ceil(1.0 * (end - start) / step));
            if (temp < 0)
              temp = 0;

            // assign output value
            ctx.getOutputType(0)
                ->mutable_tensor_type()
                ->mutable_shape()
                ->mutable_dim((int)axis)
                ->set_dim_value(temp);
          }
        }));

static const char* Scatter_ver9_doc = R"DOC(
Given `data`, `updates` and `indices` input tensors of rank r >= 1, write the values provided by `updates` 
into the first input, `data`, along `axis` dimension of `data` (by default outer-most one as axis=0) at corresponding `indices`. 
For each entry in `updates`, the target index in `data` is specified by corresponding entry in `indices`
for dimension = axis, and index in source for dimension != axis. For instance, in a 2-D tensor case,
data[indices[i][j]][j] = updates[i][j] if axis = 0, or data[i][indices[i][j]] = updates[i][j] if axis = 1,
where i and j are loop counters from 0 up to the respective size in `updates` - 1.
Example 1:
  data = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
  ]
  indices = [
      [1, 0, 2],
      [0, 2, 1],
  ]
  updates = [
      [1.0, 1.1, 1.2],
      [2.0, 2.1, 2.2],
  ]
  output = [
      [2.0, 1.1, 0.0]
      [1.0, 0.0, 2.2]
      [0.0, 2.1, 1.2]
  ]
Example 2:
  data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
  indices = [[1, 3]]
  updates = [[1.1, 2.1]]
  axis = 1
  output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Scatter,
    9,
    OpSchema()
        .SetDoc(Scatter_ver9_doc)
        .Attr(
            "axis",
            "Which axis to scatter on. Negative value means "
            "counting dimensions from the back. Accepted range is [-r, r-1]",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "Tensor of rank r >= 1.", "T")
        .Input(
            1,
            "indices",
            "Tensor of int32/int64 indices, of r >= 1 (same rank as input).",
            "Tind")
        .Input(
            2,
            "updates",
            "Tensor of rank r >=1 (same rank and shape as indices)",
            "T")
        .Output(0, "output", "Tensor of rank r >= 1 (same rank as input).", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Input and output types can be of any tensor type.")
        .TypeConstraint(
            "Tind",
            {"tensor(int32)", "tensor(int64)"},
            "Constrain indices to integer types")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

static const char* DepthToSpace_ver1_doc =
    R"DOC(DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    DepthToSpace,
    1,
    OpSchema()
        .Attr(
            "blocksize",
            "Blocks of [blocksize, blocksize] are moved.",
            AttributeProto::INT)
        .SetDoc(DepthToSpace_ver1_doc)
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
            OpSchema::all_tensor_types(),
            "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          auto blocksize = getAttribute(ctx, "blocksize", 0);
          if (blocksize <= 0)
            fail_shape_inference("Blocksize must be positive");
          if (hasInputShape(ctx, 0)) {
            auto& input_shape = getInputShape(ctx, 0);
            if (input_shape.dim_size() == 4) {
              // TODO: Clarify what behavior should be if C is not a multiple of
              // blocksize*blocksize.
              updateOutputShape(
                  ctx,
                  0,
                  {input_shape.dim(0),
                   input_shape.dim(1) / (blocksize * blocksize),
                   input_shape.dim(2) * blocksize,
                   input_shape.dim(3) * blocksize});
            } else
              fail_shape_inference("Input tensor must be 4-dimensional");
          }
        }));

static const char* Gather_ver1_doc = R"DOC(
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
them in an output tensor of rank q + (r - 1).
Example 1:
```
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
```
Example 2:
```
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
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Gather,
    1,
    OpSchema()
        .SetDoc(Gather_ver1_doc)
        .Attr(
            "axis",
            "Which axis to gather on. Negative value means "
            "counting dimensions from the back. Accepted range is [-r, r-1]",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "Tensor of rank r >= 1.", "T")
        .Input(
            1,
            "indices",
            "Tensor of int32/int64 indices, of any rank q. All index values are expected to be within bounds. "
            "It is an error if any of the index values are out of bounds.",
            "Tind")
        .Output(0, "output", "Tensor of rank q + (r - 1).", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain input and output types to any tensor type.")
        .TypeConstraint(
            "Tind",
            {"tensor(int32)", "tensor(int64)"},
            "Constrain indices to integer types")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 2)) {
            return;
          }
          const TensorShapeProto& data_shape =
              ctx.getInputType(0)->tensor_type().shape();
          const TensorShapeProto& indices_shape =
              ctx.getInputType(1)->tensor_type().shape();
          int r = data_shape.dim_size();
          if (r < 1) {
            fail_shape_inference("data tensor must have rank >= 1");
          }
          int q = indices_shape.dim_size();
          int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
          if (axis < -r || axis >= r) {
            fail_shape_inference("axis must be in [-r, r-1]");
          }
          if (axis < 0) {
            axis += r;
          }
          int out_rank = q + r - 1;

          if (out_rank == 0) {
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          }
          for (int i = 0; i < out_rank; ++i) {
            *ctx.getOutputType(0)
                 ->mutable_tensor_type()
                 ->mutable_shape()
                 ->add_dim() = (i < axis) ? data_shape.dim(i) : // i < axis < r
                (i >= axis && i < axis + q) ? indices_shape.dim(i - axis)
                                            : // i - axis < q
                    data_shape.dim(i - q + 1); // i < out_rank < q + r - 1
          }
        }));

static const char* Squeeze_ver1_doc = R"DOC(
Remove single-dimensional entries from the shape of a tensor.
Takes a  parameter `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    1,
    OpSchema()
        .Attr(
            "axes",
            "List of non-negative integers, indicate the dimensions to squeeze.",
            AttributeProto::INTS,
            OPTIONAL)
        .SetDoc(Squeeze_ver1_doc)
        .Input(0, "data", "Tensors with at least max(dims) dimensions.", "T")
        .Output(0, "squeezed", "Reshaped tensor with same data as input.", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          std::vector<int64_t> axes;
          if (!getRepeatedAttribute(ctx, "axes", axes)) {
            return;
          }

          if (!ctx.getInputType(0)->tensor_type().has_shape()) {
            return;
          }

          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();

          for (int i = 0, j = 0; i < input_shape.dim_size(); ++i) {
            if (static_cast<size_t>(j) < axes.size() && axes[j] == i) {
              if (input_shape.dim(i).has_dim_value() &&
                  input_shape.dim(i).dim_value() != 1) {
                fail_shape_inference(
                    "Dimension of input ",
                    i,
                    " must be 1 instead of ",
                    input_shape.dim(i).dim_value());
              }
              ++j;
            } else {
              *ctx.getOutputType(0)
                   ->mutable_tensor_type()
                   ->mutable_shape()
                   ->add_dim() = input_shape.dim(i);
            }
          }
        }));

static const char* Unsqueeze_ver1_doc = R"DOC(
Insert single-dimensional entries to the shape of a tensor.
Takes one required argument `axes`, a list of dimensions that will be inserted.
Dimension indices in `axes` are as seen in the output tensor. For example:
  Given a tensor such that tensor with shape [3, 4, 5], then
  Unsqueeze(tensor, axes=[0, 4]) has shape [1, 3, 4, 5, 1]
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    1,
    OpSchema()
        .Attr(
            "axes",
            "List of non-negative integers, indicate the dimensions to be inserted",
            AttributeProto::INTS)
        .SetDoc(Unsqueeze_ver1_doc)
        .Input(0, "data", "Original tensor", "T")
        .Output(0, "expanded", "Reshaped tensor with same data as input.", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          std::vector<int64_t> axes;
          if (!getRepeatedAttribute(ctx, "axes", axes)) {
            return;
          }
          std::sort(axes.begin(), axes.end());

          if (!ctx.getInputType(0)->tensor_type().has_shape()) {
            return;
          }

          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

          int j = 0;
          for (int i = 0;
               i < ctx.getInputType(0)->tensor_type().shape().dim_size();
               ++i) {
            while (static_cast<size_t>(j) < axes.size() &&
                   axes[j] ==
                       ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
              ctx.getOutputType(0)
                  ->mutable_tensor_type()
                  ->mutable_shape()
                  ->add_dim()
                  ->set_dim_value(1);
              ++j;
            }
            *ctx.getOutputType(0)
                 ->mutable_tensor_type()
                 ->mutable_shape()
                 ->add_dim() =
                ctx.getInputType(0)->tensor_type().shape().dim(i);
          }
          while (static_cast<size_t>(j) < axes.size() &&
                 axes[j] ==
                     ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
            ctx.getOutputType(0)
                ->mutable_tensor_type()
                ->mutable_shape()
                ->add_dim()
                ->set_dim_value(1);
            ++j;
          }
        }));

static const char* OneHot_ver9_doc = R"DOC(
    Produces a one-hot tensor based on inputs.
    The locations represented by the index values in the 'indices' input tensor will have 'on_value'
    and the other locations will have 'off_value' in the output tensor, where 'on_value' and 'off_value'
    are specified as part of required input argument 'values', which is a two-element tensor of format
    [off_value, on_value]. The rank of the output tensor will be one greater than the rank of the
    input tensor. The additional dimension is for one-hot representation. The additional dimension will
    be inserted at the position specified by 'axis'. If 'axis' is not specified then then additional
    dimension will be inserted as the innermost dimension, i.e. axis=-1. The size of the additional
    dimension is specified by required scalar input 'depth'. The type of the output tensor is the same
    as the type of the 'values' input. Any entries in the 'indices' input tensor with values outside
    the range [0, depth) will result in one-hot representation with all 'off_value' values in the
    output tensor.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    OneHot,
    9,
    OpSchema()
        .SetDoc(OneHot_ver9_doc)
        .Attr(
            "axis",
            "(Optional) Axis along which one-hot representation in added. Default: axis=-1. "
            "axis=-1 means that the additional dimension will be inserted as the "
            "innermost/last dimension in the output tensor.",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .Input(
            0,
            "indices",
            "Input tensor containing indices. The values must be non-negative integers. "
            "Any entries in the 'indices' input tensor with values outside the range [0, depth) "
            "will result in one-hot representation with all 'off_value' values in the output tensor."
            "In case 'indices' is of non-integer type, the values will be casted to int64 before use.",
            "T1")
        .Input(
            1,
            "depth",
            "Scalar specifying the number of classes in one-hot tensor. This is also the size "
            "of the one-hot dimension (specified by 'axis' attribute) added on in the output "
            "tensor. The values in the 'indices' input tensor are expected to be "
            "in the range [0, depth). "
            "In case 'depth' is of non-integer type, it will be casted to int64 before use.",
            "T2")
        .Input(
            2,
            "values",
            "Rank 1 tensor containing exactly two elements, in the format [off_value, on_value], "
            "where 'on_value' is the value used for filling locations specified in 'indices' input "
            "tensor, and 'off_value' is the value used for filling locations other than those specified "
            "in 'indices' input tensor. ",
            "T3")
        .Output(
            0,
            "output",
            "Tensor of rank one greater than input tensor 'indices', i.e. rank(output) = rank(indices) + 1. "
            "The data type for the elements of the output tensor is the same as the type of input 'values' "
            "is used.",
            "T3")
        .TypeConstraint(
            "T1",
            OpSchema::all_numeric_types(),
            "Constrains input to only numeric types.")
        .TypeConstraint(
            "T2",
            OpSchema::all_numeric_types(),
            "Constrains input to only numeric types.")
        .TypeConstraint(
            "T3",
            OpSchema::all_tensor_types(),
            "Constrain to any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Check that the node has three inputs.
          if (ctx.getNumInputs() != 3) {
            fail_type_inference("OneHot node must have three inputs.");
          }
          // Input 'depth' must be a scalar or a single-element vector.
          // TODO: Ideally to match spec for this input only allow Scalar should
          // be allowed. Making this change now can affect backward
          // compatibility for this op. Since this does not seem like a good
          // justification to update version for this op, allowing both scalar
          // and 1 element vector for now. In future when version update for
          // this op is done we should only allow scalar or chage the spec to
          // allow both.
          if (hasInputShape(ctx, 1)) {
            auto& depth_shape = getInputShape(ctx, 1);
            if (depth_shape.dim_size() != 0 && depth_shape.dim_size() != 1) {
              fail_type_inference(
                  "Input 'depth' must be a scalar or rank 1 tensor.");
            }
            if (depth_shape.dim_size() == 1 &&
                depth_shape.dim((int)0).has_dim_value() &&
                depth_shape.dim((int)0).dim_value() != 1) {
              fail_type_inference(
                  "Input 'depth' must have exactly one element.");
            }
          }
          // Input 'values' must be a two-element vector.
          if (hasInputShape(ctx, 2)) {
            auto& values_shape = getInputShape(ctx, 2);
            if (values_shape.dim_size() != 1) {
              fail_type_inference("Input 'values' must be rank 1 tensor.");
            }
            if (values_shape.dim((int)0).has_dim_value() &&
                values_shape.dim((int)0).dim_value() != 2) {
              fail_type_inference(
                  "Input 'values' must have exactly two elements.");
            }
          }
          // Set output type to be the same as the third input, 'values'.
          propagateElemTypeFromInputToOutput(ctx, 2, 0);
          // Set the output shape, if input 0 (indices) shape is available.
          if (hasInputShape(ctx, 0)) {
            const TensorShapeProto& indices_shape =
                ctx.getInputType(0)->tensor_type().shape();
            int r = indices_shape.dim_size();
            if (r < 1) {
              fail_shape_inference("Indices tensor must have rank >= 1");
            }
            int out_rank = r + 1;
            int axis = static_cast<int>(getAttribute(ctx, "axis", -1));
            if (axis < -out_rank || axis >= out_rank) {
              fail_shape_inference(
                  "'axis' must be in [-rank(indices)-1, rank(indices)]");
            }
            if (axis < 0) {
              axis += out_rank;
            }
            auto* output_shape = getOutputShape(ctx, 0);
            for (int i = 0; i < out_rank; ++i) {
              auto* dim = output_shape->add_dim();
              if (i < axis) {
                if (indices_shape.dim(i).has_dim_value()) {
                  dim->set_dim_value(indices_shape.dim(i).dim_value());
                } else if (indices_shape.dim(i).has_dim_param()) {
                  dim->set_dim_param(indices_shape.dim(i).dim_param());
                }
              } else if (i > axis) {
                if (indices_shape.dim(i - 1).has_dim_value()) {
                  dim->set_dim_value(indices_shape.dim(i - 1).dim_value());
                } else if (indices_shape.dim(i - 1).has_dim_param()) {
                  dim->set_dim_param(indices_shape.dim(i - 1).dim_param());
                }
              }
            }
          }
        }));

static const char* Compress_ver9_doc = R"DOC(
    Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
    In case axis is not provided, input is flattened before elements are selected.
    Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html
    )DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Compress,
    9,
    OpSchema()
        .SetDoc(Compress_ver9_doc)
        .Attr(
            "axis",
            "(Optional) Axis along which to take slices. If not specified, "
            "input is flattened before elements being selected.",
            AttributeProto::INT,
            OPTIONAL)
        .Input(0, "input", "Tensor of rank r >= 1.", "T")
        .Input(
            1,
            "condition",
            "Rank 1 tensor of booleans to indicate which slices or data elements to be selected. "
            "Its length can be less than the input length alone the axis "
            "or the flattened input size if axis is not specified. "
            "In such cases data slices or elements exceeding the condition length are discarded.",
            "T1")
        .Output(
            0,
            "output",
            "Tensor of rank r if axis is specified. Otherwise output is a Tensor of rank 1.",
            "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain input and output types to all tensor types.")
        .TypeConstraint(
            "T1",
            {"tensor(bool)"},
            "Constrains to boolean tensors."));

static const char* Split_ver2_doc =
    R"DOC(Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using argument 'split'.
Otherwise, the tensor is split to equal sized parts.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Split,
    2,
    OpSchema()
        .Input(0, "input", "The tensor to split", "T")
        .Output(
            0,
            "outputs",
            "One or more outputs forming list of tensors after splitting",
            "T",
            OpSchema::Variadic)
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain input and output types to all tensor types.")
        .Attr(
            "axis",
            "Which axis to split on. ",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr("split", "length of each output", AttributeProto::INTS, OPTIONAL)
        .SetDoc(Split_ver2_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); ++i) {
            propagateElemTypeFromInputToOutput(ctx, 0, i);
          }
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          auto axisAttr = ctx.getAttribute("axis");
          int axis = axisAttr ? static_cast<int>(axisAttr->i()) : 0;
          if (axis < 0) {
            return;
          }
          std::vector<int64_t> split;
          if (!getRepeatedAttribute(ctx, "split", split)) {
            if (!ctx.getInputType(0)->tensor_type().has_shape()) {
              return;
            }
            const auto& shape = ctx.getInputType(0)->tensor_type().shape();
            if (axis >= shape.dim_size()) {
              fail_type_inference("Invalid value of attribute 'axis'");
            }
            const auto& splitDim = shape.dim(axis);
            if (!splitDim.has_dim_value()) {
              return;
            }
            int splitDimValue = static_cast<int>(splitDim.dim_value());
            int chunkSize =
                splitDimValue / static_cast<int>(ctx.getNumOutputs());
            int leftOver = splitDimValue -
                (chunkSize * static_cast<int>(ctx.getNumOutputs()));
            for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); i++) {
              split.push_back(i < leftOver ? chunkSize + 1 : chunkSize);
            }

            for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
              *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() =
                  shape;
              ctx.getOutputType(i)
                  ->mutable_tensor_type()
                  ->mutable_shape()
                  ->mutable_dim(axis)
                  ->set_dim_value(split[i]);
            }
          }
        }));

static const char* Pad_ver2_doc = R"DOC(
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
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Pad,
    2,
    OpSchema()
        .Attr(
            "pads",
            "List of integers indicating the number of padding elements to add or remove (if negative) "
            "at the beginning and end of each axis. For 2D it is the number of pixels. "
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
            "One float, indicates the value to be filled.",
            AttributeProto::FLOAT,
            0.0f)
        .SetDoc(Pad_ver2_doc)
        .Input(0, "data", "Input tensor.", "T")
        .Output(0, "output", "Tensor after padding.", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          auto& input_shape = ctx.getInputType(0)->tensor_type().shape();

          std::vector<int64_t> pads;
          if (!getRepeatedAttribute(ctx, "pads", pads))
            fail_shape_inference("Attribute value for pads is required");
          if (pads.size() != static_cast<size_t>(input_shape.dim_size() * 2)) {
            fail_shape_inference("Attribute pads has incorrect length");
            ;
          }

          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

          for (size_t i = 0; (int64_t)i < input_shape.dim_size(); ++i) {
            auto* newdim = ctx.getOutputType(0)
                               ->mutable_tensor_type()
                               ->mutable_shape()
                               ->add_dim();
            if (ctx.getInputType(0)
                    ->tensor_type()
                    .shape()
                    .dim((int)i)
                    .has_dim_value()) {
              newdim->set_dim_value(
                  ctx.getInputType(0)
                      ->tensor_type()
                      .shape()
                      .dim((int)i)
                      .dim_value() +
                  pads[i] + pads[input_shape.dim_size() + i]);
            } else if (pads[i] + pads[input_shape.dim_size() + i] == 0) {
              *newdim = input_shape.dim((int)i);
            }
          }
        }));
} // namespace ONNX_NAMESPACE