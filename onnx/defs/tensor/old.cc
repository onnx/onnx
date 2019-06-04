// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include <algorithm>
#include <cmath>
#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

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
          auto& input_shape = getInputShape(ctx, 0);
          auto* output_shape = getOutputShape(ctx, 0);
          auto* scale_attr = ctx.getAttribute("scales");
          if (input_shape.dim_size() != scale_attr->floats_size()) {
            fail_shape_inference(
                "Upsample: Input dims != attribute 'scales' dims");
          }
          for (int i = 0; i < input_shape.dim_size(); ++i) {
            float dim_value =
                static_cast<float>(input_shape.dim(i).dim_value());
            output_shape->add_dim()->set_dim_value(static_cast<int64_t>(
                std::floor(dim_value * scale_attr->floats(i))));
          }
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
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          auto& input_shape = getInputShape(ctx, 0);
          auto* output_shape = getOutputShape(ctx, 0);
          output_shape->clear_dim();
          auto scales = ctx.getInputData(1);
          if (nullptr != scales) {
            // Infer output shape's dimension value if 'scales' is known.
            if (scales->data_type() == TensorProto::FLOAT) {
              const auto& data = ParseData<float>(scales);
              if (static_cast<int>(data.size()) == input_shape.dim_size()) {
                for (int i = 0; i < input_shape.dim_size(); ++i) {
                  float dim_value =
                      static_cast<float>(input_shape.dim(i).dim_value());
                  output_shape->add_dim()->set_dim_value(
                      static_cast<int64_t>(std::floor(dim_value * data[i])));
                }
              } else {
                fail_shape_inference(
                    "Number of elements of input 'scales' must be same as rank of input 'X'.");
              }
            } else {
              fail_shape_inference(
                  "Input scales's element type must be float.");
            }
          } else {
            // Infer output shape's rank in any case.
            for (int i = 0; i < input_shape.dim_size(); ++i) {
              output_shape->add_dim();
            }
          }
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
} // namespace ONNX_NAMESPACE
