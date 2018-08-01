// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

#include <algorithm>

namespace ONNX_NAMESPACE {
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
            "The data type to which the elements of the input tensor are cast."
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

static const char* Reshape_ver5_doc = R"DOC(
Reshape the input tensor similar to numpy.reshape.
First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor).)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Reshape, 5,
    OpSchema()
        .SetDoc(Reshape_ver5_doc)
        .Input(0, "data", "An input tensor.", "T")
        .Input(1, "shape", "Specified shape for output.", "tensor(int64)")
        .Output(0, "reshaped", "Reshaped data.", "T")
        .TypeConstraint(
            "T",
			OpSchema::all_tensor_types(),
            "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          // Shape Inference if 2nd input data (the target shape) is available
          const TensorProto *targetShapeInitializer =
              ctx.getInputData(1);
          if (!targetShapeInitializer) {
            return;
          }
          // Make targetShape (0 -> same as originalShape, -1 -> inferred).
          // The targetShape vector represents the specified shape for output.
          std::vector<int64_t> targetShape;
          if (targetShapeInitializer->has_raw_data()) {
            const std::string &bytes = targetShapeInitializer->raw_data();
            targetShape.insert(targetShape.end(),
                               reinterpret_cast<const int64_t *>(bytes.c_str()),
                               reinterpret_cast<const int64_t *>(bytes.c_str() +
                                                                 bytes.size()));
          } else {
            const auto &data = targetShapeInitializer->int64_data();
            targetShape.insert(targetShape.end(), data.begin(), data.end());
          }

          // Iterate through targetShape, adding dimensions in the outputShape
          // TensorProto. If the targertShape dimension is -1, we do not set the
          // dimension value in this iteration, but we record the Dimension. If
          // targertShape dimension is 0, we attempt to propagate the dimension
          // value/param. If the value cannot be inferred, we set the flag in
          // the unresolveZeros vector. If targetShape dimension is positive, we
          // set the dimension value in the outputShape. We track the product of
          // the dimensions we are setting outputShape in the outputProduct
          // variable. The outputProduct will potentially be used for inferring
          // a dimension marked -1.
          auto *outputShape =
              ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          TensorShapeProto::Dimension *negativeOneDim = nullptr;
          const auto &dataInputTensorType = ctx.getInputType(0)->tensor_type();
          std::vector<bool> unresolvedZeros(targetShape.size(), false);
          int64_t outputProduct = 1;
          for (int i = 0; i < static_cast<int>(targetShape.size()); ++i) {
            // Add a new dimension to outputShape
            auto *new_dim = outputShape->add_dim();
            if (targetShape[i] == -1) {
              // Check if multiple -1's. If not, set negativeOneDim, marking
              // this dimension to potentially be filled in later.
              if (negativeOneDim) {
                fail_shape_inference(
                    "Target shape may not have multiple -1 dimensions");
              }
              negativeOneDim = new_dim;
            } else if (targetShape[i] == 0) {
              // Check if data input has a shape and if the index i is within
              // its bounds. If these conditions are satisfied, any dimension
              // value/param should be propogated. If dimension value cannot be
              // inferred, set the corresponding  unresolvedZeros flag to true.
              unresolvedZeros[i] = true;
              if (dataInputTensorType.has_shape()) {
                if (i >= dataInputTensorType.shape().dim_size()) {
                  fail_shape_inference("Invalid position of 0");
                }
                if (dataInputTensorType.shape().dim(i).has_dim_value()) {
                  const auto &dim_value =
                      dataInputTensorType.shape().dim(i).dim_value();
                  new_dim->set_dim_value(dim_value);
                  outputProduct *= dim_value;
                  unresolvedZeros[i] = false;
                } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
                  const auto &dim_param =
                      dataInputTensorType.shape().dim(i).dim_param();
                  new_dim->set_dim_param(dim_param);
                }
              }
            } else if (targetShape[i] > 0) {
              // Set the dimension value to targetShape[i]
              new_dim->set_dim_value(targetShape[i]);
              outputProduct *= targetShape[i];
            } else {
              // Check if value is less than -1; fail if so
              fail_shape_inference("Invalid dimension value: ", targetShape[i]);
            }
          }

          // If negativeOneDim has been set, we attempt to infer its value. This
          // can be done if all dimension values for the data input tensor shape
          // are known other than the ones corresponding to unresolvedZeros
          // flags.
          if (negativeOneDim) {
            // First, attempt to compute product of data input shape dimensions
            // that are not marked by unresolvedZeros. If not possible, set the
            // inputProductValid flag to false.
            if (!outputProduct) {
              fail_shape_inference("Invalid Target shape product of 0");
            }
            int64_t inputProduct = 1;
            bool inputProductValid = true;
            if (!dataInputTensorType.has_shape()) {
              inputProductValid = false;
            } else {
              for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
                if (dataInputTensorType.shape().dim(i).has_dim_value()) {
                  inputProduct *=
                      dataInputTensorType.shape().dim(i).dim_value();
                } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
                  inputProductValid = false;
                  break;
                }
              }
            }
            if (inputProductValid) {
              if (inputProduct % outputProduct != 0) {
                fail_shape_inference(
                    "Dimension could not be inferred: incompatible shapes");
              }
              negativeOneDim->set_dim_value(inputProduct / outputProduct);
            }
          }
        }));

static const char* Shape_ver1_doc = R"DOC(
Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    1,
    OpSchema()
        .SetDoc(Shape_ver1_doc)
        .Input(0, "data", "An input tensor.", "T")
        .Output(0, "shape", "Shape of the input tensor", "T1")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Input tensor can be of arbitrary type.")
        .TypeConstraint(
            "T1",
            {"tensor(int64)"},
            "Constrain output to int64 tensor.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(
              TensorProto::INT64);

          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          if (ctx.getInputType(0)->tensor_type().has_shape()) {
            ctx.getOutputType(0)
                ->mutable_tensor_type()
                ->mutable_shape()
                ->add_dim()
                ->set_dim_value(
                    ctx.getInputType(0)->tensor_type().shape().dim_size());
          }
        }));

static const char* Size_ver1_doc = R"DOC(
Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    1,
    OpSchema()
        .SetDoc(Size_ver1_doc)
        .Input(0, "data", "An input tensor.", "T")
        .Output(0, "size", "Total number of elements of the input tensor", "T1")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Input tensor can be of arbitrary type.")
        .TypeConstraint(
            "T1",
            {"tensor(int64)"},
            "Constrain output to int64 tensor, which should be a scalar though.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(
              TensorProto::INT64);
          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
        }));

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
              } else if (shape.dim(j).has_dim_value()) {
                if (output_shape->dim(j).has_dim_value()) {
                  if (shape.dim(j).dim_value() !=
                      output_shape->dim(j).dim_value()) {
                    fail_shape_inference("Dimension mismatch");
                    ;
                  }
                } else {
                  *output_shape->mutable_dim(j) = shape.dim(j);
                }
              }
            }
          }

          if (all_lengths_known) {
            output_shape->mutable_dim(axis)->set_dim_value(total_length);
          }
        }));

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
            "Which axis to split on (defaults to 0)",
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

static const char* Transpose_ver1_doc = R"DOC(
Transpose the input tensor similar to numpy.transpose. For example, when
perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
will be (2, 1, 3).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Transpose,
    1,
    OpSchema()
        .SetDoc(Transpose_ver1_doc)
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
            OpSchema::all_tensor_types(),
            "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          auto input_type = ctx.getInputType(0);
          const TensorShapeProto& shape = input_type->tensor_type().shape();
          std::vector<int64_t> perm;
          bool has_perm_attr = getRepeatedAttribute(ctx, "perm", perm);
          if (!has_perm_attr) {
            for (int i = shape.dim_size() - 1; i >= 0; --i)
              perm.push_back(i);
          } else if (!perm.empty()) {
            // check if every index is valid
            for (int64_t fromDimIndex : perm)
              if (!(0 <= fromDimIndex && fromDimIndex < shape.dim_size())) {
                std::ostringstream oss;
                oss << "Invalid attribute perm {" << perm[0];
                for (size_t i = 1; i != perm.size(); ++i) {
                  oss << ", " << perm[i];
                }
                oss << "}, input shape = {";
                if (shape.dim_size() > 0) {
                  oss << shape.dim(0).dim_value();
                  for (int i = 1; i != shape.dim_size(); ++i) {
                    oss << ", " << shape.dim(i).dim_value();
                  }
                  oss << "}";
                }
                fail_type_inference(oss.str());
              }
          }

          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          for (size_t i = 0; i < perm.size(); ++i) {
            appendSingleDimCopiedFromInputTypeToOutputType(
                ctx, 0, 0, static_cast<size_t>(perm[i]));
          }
        }));

static const char* Gather_ver1_doc = R"DOC(
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
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Gather,
    1,
    OpSchema()
        .SetDoc(Gather_ver1_doc)
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

          int r = ctx.getInputType(0)->tensor_type().shape().dim_size();
          int q = ctx.getInputType(1)->tensor_type().shape().dim_size();

          int out_rank = q + r - 1;

          if (out_rank == 0) {
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          }
          for (int i = 0; i < out_rank; ++i) {
            ctx.getOutputType(0)
                ->mutable_tensor_type()
                ->mutable_shape()
                ->add_dim();
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
            "List of positive integers, indicate the dimensions to squeeze.",
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

          for (int i = 0, j = 0;
               i < ctx.getInputType(0)->tensor_type().shape().dim_size();
               ++i) {
            if (static_cast<size_t>(j) < axes.size() && axes[j] == i) {
              ++j;
            } else {
              *ctx.getOutputType(0)
                   ->mutable_tensor_type()
                   ->mutable_shape()
                   ->add_dim() =
                  ctx.getInputType(0)->tensor_type().shape().dim(i);
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
            "List of positive integers, indicate the dimensions to be inserted",
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
            "One float, indicates the value to be filled, default is 0",
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

static const char* SpaceToDepth_ver1_doc =
    R"DOC(SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
this op outputs a copy of the input tensor where values from the height and width dimensions
are moved to the depth dimension.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    SpaceToDepth,
    1,
    OpSchema()
        .Attr(
            "blocksize",
            "Blocks of [blocksize, blocksize] are moved.",
            AttributeProto::INT)
        .SetDoc(SpaceToDepth_ver1_doc)
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
              // TODO: Clarify what behavior should be if H or W is not a
              // multiple of blocksize.
              updateOutputShape(
                  ctx,
                  0,
                  {input_shape.dim(0),
                   input_shape.dim(1) * (blocksize * blocksize),
                   input_shape.dim(2) / blocksize,
                   input_shape.dim(3) / blocksize});
            } else
              fail_shape_inference("Input tensor must be 4-dimensional");
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

static const char* Tile_ver6_doc =
    R"DOC(Constructs a tensor by tiling a given tensor.
This is the same as function `tile` in Numpy, but no broadcast.
For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Tile,
    6,
    OpSchema()
        .SetDoc(Tile_ver6_doc)
        .Input(0, "input", "Input tensor of any shape.", "T")
        .Input(
            1,
            "repeats",
            "1D int64 tensor of the same length as input's dimension number, "
            "includes numbers of repeated copies along input's dimensions.",
            "T1")
        .Output(
            0,
            "output",
            "Output tensor of the same dimension and type as tensor input. "
            "output_dim[i] = input_dim[i] * repeats[i]",
            "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain input and output types to all tensor types.")
        .TypeConstraint(
            "T1",
            {"tensor(int64)"},
            "Constrain repeat's type to int64 tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          // Only rank of output can be inferred. We can do better if second
          // input is a constant, but this requires extending InferenceContext
          // interface to get values of constant inputs.
        }));

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
        .SetDoc(Upsample_ver7_doc));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    1,
    OpSchema()
        .SetDoc("Identity operator")
        .Input(0, "input", "Input tensor", "T")
        .Output(0, "output", "Tensor to copy input into.", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));
} // namespace ONNX_NAMESPACE
