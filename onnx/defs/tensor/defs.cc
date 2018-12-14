// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

#include <algorithm>
#include <cmath>

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
    Reshape,
    5,
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
          const TensorProto* targetShapeInitializer = ctx.getInputData(1);
          if (!targetShapeInitializer) {
            return;
          }
          // Make targetShape (0 -> same as originalShape, -1 -> inferred).
          // The targetShape vector represents the specified shape for output.
          std::vector<int64_t> targetShape;
          if (targetShapeInitializer->has_raw_data()) {
            const std::string& bytes = targetShapeInitializer->raw_data();
            targetShape.insert(
                targetShape.end(),
                reinterpret_cast<const int64_t*>(bytes.c_str()),
                reinterpret_cast<const int64_t*>(bytes.c_str() + bytes.size()));
          } else {
            const auto& data = targetShapeInitializer->int64_data();
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
          auto* outputShape =
              ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          TensorShapeProto::Dimension* negativeOneDim = nullptr;
          const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
          std::vector<bool> unresolvedZeros(targetShape.size(), false);
          int64_t outputProduct = 1;
          for (int i = 0; i < static_cast<int>(targetShape.size()); ++i) {
            // Add a new dimension to outputShape
            auto* new_dim = outputShape->add_dim();
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
                  const auto& dim_value =
                      dataInputTensorType.shape().dim(i).dim_value();
                  new_dim->set_dim_value(dim_value);
                  outputProduct *= dim_value;
                  unresolvedZeros[i] = false;
                } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
                  const auto& dim_param =
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
                } else if (
                    i >= static_cast<int>(unresolvedZeros.size()) ||
                    !unresolvedZeros[i]) {
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
            "Which axis to split on.",
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
            "counting dimensions from the back. Accepted range in [-r, r-1]",
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
            "T"
        )
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
            "Which axis to gather on. Negative value means "
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
          const TensorShapeProto& data_shape = ctx.getInputType(0)->tensor_type().shape();
          const TensorShapeProto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
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
                ->add_dim() =
                (i < axis) ? data_shape.dim(i) :                             // i < axis < r
                (i >= axis && i < axis + q) ? indices_shape.dim(i - axis) :  // i - axis < q
                data_shape.dim(i - q + 1);                                   // i < out_rank < q + r - 1
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
            if (scales->data_type() == TensorProto::FLOAT &&
                scales->float_data_size() == input_shape.dim_size()) {
              for (int i = 0; i < input_shape.dim_size(); ++i) {
                float dim_value =
                    static_cast<float>(input_shape.dim(i).dim_value());
                output_shape->add_dim()->set_dim_value(static_cast<int64_t>(
                    std::floor(dim_value * scales->float_data(i))));
              }
            } else {
              fail_shape_inference(
                  "Number of elements of input 'scales' must be same as rank of input 'X' and element type must be float.");
            }
          } else {
            // Infer output shape's rank in any case.
            for (int i = 0; i < input_shape.dim_size(); ++i) {
              output_shape->add_dim();
            }
          }

        }));

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
        .Input(1,
            "condition",
            "Rank 1 tensor of booleans to indicate which slices or data elements to be selected. "
            "Its length can be less than the input length alone the axis "
            "or the flattened input size if axis is not specified. "
            "In such cases data slices or elements exceeding the condition length are discarded.",
            "T1")
        .Output(0, "output", "Tensor of rank r if axis is specified. Otherwise output is a Tensor of rank 1.", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain input and output types to all tensor types.")
        .TypeConstraint(
            "T1",
            {"tensor(bool)"},
            "Constrains to boolean tensors."));

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
            "tensor and the values in the 'indices' input tensor are expected to be "
            "in the range [0, depth). The"
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
        .Output(0,
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
            fail_type_inference(
                    "OneHot node must have three inputs.");
          }
          // Input 'depth' must be a single-element vector.
          if (hasInputShape(ctx, 1)) {
            auto& depth_shape = getInputShape(ctx, 1);
            if (depth_shape.dim_size() != 1) {
              fail_type_inference(
                    "Input 'depth' must be rank 1 tensor.");
            }
            if (depth_shape.dim((int)0).has_dim_value() &&
                depth_shape.dim((int)0).dim_value() != 1) {
              fail_type_inference(
                      "Input 'depth' must have exactly one element.");
            }
          }
          // Input 'values' must be a two-element vector.
          if (hasInputShape(ctx, 2)) {
            auto& values_shape = getInputShape(ctx, 2);
            if (values_shape.dim_size() != 1) {
              fail_type_inference(
                    "Input 'values' must be rank 1 tensor.");
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
            const TensorShapeProto& indices_shape = ctx.getInputType(0)->tensor_type().shape();
            int r = indices_shape.dim_size();
            if (r < 1) {
              fail_shape_inference("Indices tensor must have rank >= 1");
            }
            int out_rank = r + 1;
            int axis = static_cast<int>(getAttribute(ctx, "axis", -1));
            if (axis < -out_rank || axis >= out_rank) {
              fail_shape_inference("'axis' must be in [-rank(indices)-1, rank(indices)]");
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
              }
              else if(i > axis) {
                if (indices_shape.dim(i - 1).has_dim_value()) {
                  dim->set_dim_value(indices_shape.dim(i - 1).dim_value());
                } else if (indices_shape.dim(i - 1).has_dim_param()) {
                  dim->set_dim_param(indices_shape.dim(i - 1).dim_param());
                }
              }
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    IsNaN,
    9,
    OpSchema()
    .SetDoc(R"DOC(Returns which elements of the input are NaN.)DOC")
    .Input(0, "X", "input", "T1")
    .Output(0, "Y", "output", "T2")
    .TypeConstraint(
        "T1",
        {"tensor(float16)","tensor(float)","tensor(double)"},
        "Constrain input types to float tensors.")
    .TypeConstraint(
        "T2",
        {"tensor(bool)"},
        "Constrain output types to boolean tensors.")
    .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
                                     updateOutputElemType(ctx, 0, TensorProto::BOOL);
                                     if (hasInputShape(ctx, 0)) {
                                       propagateShapeFromInputToOutput(ctx, 0, 0);
                                     }}
      ));

} // namespace ONNX_NAMESPACE
