// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "onnx/defs/tensor/utils.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ONNX_NAMESPACE {
static const char* Cast_ver9_doc = R"DOC(
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.

Casting from string tensor in plain (e.g., "3.14" and "1000") and scientific numeric representations
(e.g., "1e-5" and "1E8") to float types is supported. For example, converting string "100.5" to an integer may
result 100. There are some string literals reserved for special floating-point values;
"+INF" (and "INF"), "-INF", and "NaN" are positive infinity, negative infinity, and not-a-number, respectively.
Any string which can exactly match "+INF" in a case-insensitive way would be mapped to positive infinite. Similarly,
this case-insensitive rule is applied to "INF" and "NaN". When casting from numeric tensors
to string tensors, plain floating-point representation (such as "314.15926") would be used. 
Converting non-numerical-literal string such as "Hello World!" is an undefined behavior. Cases 
of converting string representing floating-point arithmetic value, such as "2.718", to INT is an undefined behavior.

Conversion from a numerical type to any numerical type is always allowed.
User must be aware of precision loss and value change caused by range difference between two types.
For example, a 64-bit float 3.1415926459 may be round to a 32-bit float 3.141592. Similarly, converting
an integer 36 to Boolean may produce 1 because we truncate bits which can't be stored in the targeted type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    9,
    OpSchema()
        .SetDoc(Cast_ver9_doc)
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
             "tensor(bool)",
             "tensor(string)"},
            "Constrain input types. Casting from complex is not supported.")
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
             "tensor(bool)",
             "tensor(string)"},
            "Constrain output types. Casting to complex is not supported.")
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
    11,
    OpSchema()
        .Attr(
            "axis",
            "Which axis to concat on. A negative value means counting dimensions from the back. "
            "Accepted range is [-r, r-1] where r = rank(inputs)..",
            AttributeProto::INT)
        .SetDoc(
            "Concatenate a list of tensors into a single tensor. "
            "All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.")
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
          if (axis < -rank || axis >= rank) {
            fail_shape_inference("axis must be in [-rank, rank-1].");
          }
          if (axis < 0) {
            axis += rank;
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

static const char* Split_ver11_doc =
    R"DOC(Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using argument 'split'.
Otherwise, the tensor is split to equal sized parts.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Split,
    11,
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
            "Which axis to split on. "
            "A negative value means counting dimensions from the back. Accepted range is [-rank, rank-1] "
            "where r = rank(input).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr("split", "length of each output", AttributeProto::INTS, OPTIONAL)
        .SetDoc(Split_ver11_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); ++i) {
            propagateElemTypeFromInputToOutput(ctx, 0, i);
          }
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          std::vector<int64_t> split;
          if (!getRepeatedAttribute(ctx, "split", split)) {
            if (!ctx.getInputType(0)->tensor_type().has_shape()) {
              return;
            }
            const auto& shape = ctx.getInputType(0)->tensor_type().shape();
            int rank = shape.dim_size();
            int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
            if (axis < -rank || axis >= rank) {
              fail_type_inference(
                  "Invalid value of attribute 'axis'. Rank=",
                  rank,
                  " Value=",
                  axis);
            }
            if (axis < 0) {
              axis += rank;
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

static const char* Slice_ver11_doc = R"DOC(
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
    11,
    OpSchema()
        .SetDoc(Slice_ver11_doc)
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
            "1-D tensor of axes that `starts` and `ends` apply to. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(data).",
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

static const char* Scatter_ver11_doc = R"DOC(
This operator is deprecated. Please use ScatterElements, which provides the same functionality.

Scatter takes three inputs `data`, `updates`, and `indices` of the same
rank r >= 1 and an optional attribute axis that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). The output of the operation
is produced by creating a copy of the input `data`, and then updating its value
to values specified by `updates` at specific index positions specified by
`indices`. Its output shape is the same as the shape of `data`.

For each entry in `updates`, the target index in `data` is obtained by combining
the corresponding entry in `indices` with the index of the entry itself: the
index-value for dimension = axis is obtained from the value of the corresponding
entry in `indices` and the index-value for dimension != axis is obtained from the
index of the entry itself.

For instance, in a 2-D tensor case, the update corresponding to the [i][j] entry
is performed as below:
```
  output[indices[i][j]][j] = updates[i][j] if axis = 0, 
  output[i][indices[i][j]] = updates[i][j] if axis = 1,
```

This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.

Example 1:
```
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
```
Example 2:
```
  data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
  indices = [[1, 3]]
  updates = [[1.1, 2.1]]
  axis = 1
  output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Scatter,
    11,
    OpSchema()
        .Deprecate()
        .SetDoc(Scatter_ver11_doc)
        .Attr(
            "axis",
            "Which axis to scatter on. Negative value means "
            "counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "Tensor of rank r >= 1.", "T")
        .Input(
            1,
            "indices",
            "Tensor of int32/int64 indices, of r >= 1 (same rank as input). All index values are expected to be "
            "within bounds [-s, s-1] along axis of size s. It is an error if any of the index values are out of bounds.",
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

static const char* ScatterND_ver11_doc = R"DOC(
ScatterND takes three inputs `data` tensor of rank r >= 1, `indices` tensor of rank q >= 1,
and `updates` tensor of rank q + r - indices.shape[-1] - 1. The output of the operation
is produced by creating a copy of the input `data`, and then updating its value to values
specified by `updates` at specific index positions specified by `indices`. Its output shape
is the same as the shape of `data`. Note that `indices` should not have duplicate entries.
That is, two or more `updates` for the same index-location is not supported.

`indices` is an integer tensor. Let k denote indices.shape[-1], the last dimension in the shape of `indices`.
 `indices` is treated as a (q-1)-dimensional tensor of k-tuples, where each k-tuple is a partial-index into `data`.
Hence, k can be a value at most the rank of `data`. When k equals rank(data), each update entry specifies an
update to a single element of the tensor. When k is less than rank(data) each update entry specifies an
update to a slice of the tensor.

`updates` is treated as a (q-1)-dimensional tensor of replacement-slice-values. Thus, the
first (q-1) dimensions of updates.shape must match the first (q-1) dimensions of indices.shape.
The remaining dimensions of `updates` correspond to the dimensions of the
replacement-slice-values. Each replacement-slice-value is a (r-k) dimensional tensor,
corresponding to the trailing (r-k) dimensions of `data`.  Thus, the shape of `updates`
must equal indices.shape[0:q-1] ++ data.shape[k:r-1], where ++ denotes the concatenation
of shapes.

The `output` is calculated via the following equation:

    output = np.copy(data)
    update_indices = indices.shape[:-1]
    for idx in np.ndindex(update_indices):
        output[indices[idx]] = updates[idx]

The order of iteration in the above loop is not specified.
In particular, indices should not have duplicate entries: that is, if idx1 != idx2, then indices[idx1] != indices[idx2].
This ensures that the output value does not depend on the iteration order.

This operator is the inverse of GatherND.

Example 1:
```
  data    = [1, 2, 3, 4, 5, 6, 7, 8]
  indices = [[4], [3], [1], [7]]
  updates = [9, 10, 11, 12]
  output  = [1, 11, 3, 10, 9, 6, 7, 12]
```

Example 2:
```
  data    = [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
             [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
  indices = [[0], [2]]
  updates = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
             [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]]
  output  = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
             [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
             [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ScatterND,
    11,
    OpSchema()
        .SetDoc(ScatterND_ver11_doc)
        .Input(0, "data", "Tensor of rank r >= 1.", "T")
        .Input(1, "indices", "Tensor of rank q >= 1.", "tensor(int64)")
        .Input(
            2,
            "updates",
            "Tensor of rank q + r - indices_shape[-1] - 1.",
            "T")
        .Output(0, "output", "Tensor of rank r >= 1.", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain input and output types to any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          propagateShapeFromInputToOutput(ctx, 0, 0);
        }));

static const char* ScatterElements_ver11_doc = R"DOC(
ScatterElements takes three inputs `data`, `updates`, and `indices` of the same
rank r >= 1 and an optional attribute axis that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). The output of the operation
is produced by creating a copy of the input `data`, and then updating its value
to values specified by `updates` at specific index positions specified by
`indices`. Its output shape is the same as the shape of `data`.

For each entry in `updates`, the target index in `data` is obtained by combining
the corresponding entry in `indices` with the index of the entry itself: the
index-value for dimension = axis is obtained from the value of the corresponding
entry in `indices` and the index-value for dimension != axis is obtained from the
index of the entry itself.

For instance, in a 2-D tensor case, the update corresponding to the [i][j] entry
is performed as below:
```
  output[indices[i][j]][j] = updates[i][j] if axis = 0, 
  output[i][indices[i][j]] = updates[i][j] if axis = 1,
```

This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.

Example 1:
```
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
```
Example 2:
```
  data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
  indices = [[1, 3]]
  updates = [[1.1, 2.1]]
  axis = 1
  output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ScatterElements,
    11,
    OpSchema()
        .SetDoc(ScatterElements_ver11_doc)
        .Attr(
            "axis",
            "Which axis to scatter on. Negative value means "
            "counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "Tensor of rank r >= 1.", "T")
        .Input(
            1,
            "indices",
            "Tensor of int32/int64 indices, of r >= 1 (same rank as input). All index values are expected to be "
            "within bounds [-s, s-1] along axis of size s. It is an error if any of the index values are out of bounds.",
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

static const char* Gather_ver11_doc = R"DOC(
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
them in an output tensor of rank q + (r - 1).

axis = 0 :

Let
k = indices[i_{0}, ..., i_{q-1}]
Then
output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[k , j_{0}, ..., j_{r-2}]

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
axis = 1 :

Let
k = indices[i_{0}, ..., i_{q-1}]
Then
output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[j_{0}, k, j_{1}, ..., j_{r-2}]

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
    11,
    OpSchema()
        .SetDoc(Gather_ver11_doc)
        .Attr(
            "axis",
            "Which axis to gather on. Negative value means "
            "counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "Tensor of rank r >= 1.", "T")
        .Input(
            1,
            "indices",
            "Tensor of int32/int64 indices, of any rank q. All index values are expected to be within bounds [-s, s-1] "
            "along axis of size s. It is an error if any of the index values are out of bounds.",
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

static const char* GatherElements_ver11_doc = R"DOC(

GatherElements takes two inputs `data` and `indices` of the same rank r >= 1
and an optional attribute `axis` that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). It is an indexing operation
that produces its output by indexing into the input data tensor at index
positions determined by elements of the `indices` tensor.
Its output shape is the same as the shape of `indices` and consists of one value
(gathered from the `data`) for each element in `indices`.

For instance, in the 3-D case (r = 3), the output produced is determined
by the following equations: 
```
  out[i][j][k] = input[index[i][j][k]][j][k] if axis = 0,
  out[i][j][k] = input[i][index[i][j][k]][k] if axis = 1,
  out[i][j][k] = input[i][j][index[i][j][k]] if axis = 2,
```

This operator is also the inverse of ScatterElements. It is similar to Torch's gather operation.

Example 1:
```
  data = [
      [1, 2],
      [3, 4],
  ]
  indices = [
      [0, 0],
      [1, 0],
  ]
  axis = 1
  output = [
      [
        [1, 1],
        [4, 3],
      ],
  ]
```
Example 2:
```
  data = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
  ]
  indices = [
      [1, 2, 0],
      [2, 0, 0],
  ]
  axis = 0
  output = [
      [
        [4, 8, 3],
        [7, 2, 3],
      ],
  ]
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    GatherElements,
    11,
    OpSchema()
        .SetDoc(GatherElements_ver11_doc)
        .Attr(
            "axis",
            "Which axis to gather on. Negative value means "
            "counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "Tensor of rank r >= 1.", "T")
        .Input(
            1,
            "indices",
            "Tensor of int32/int64 indices, with the same rank r as the input. All index values are expected to be "
            "within bounds [-s, s-1] along axis of size s. It is an error if any of the index values are out of bounds.",
            "Tind")
        .Output(0, "output", "Tensor of the same shape as indices.", "T")
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
          propagateShapeFromInputToOutput(ctx, 1, 0);
        }));

static const char* Squeeze_ver11_doc = R"DOC(
Remove single-dimensional entries from the shape of a tensor.
Takes a  parameter `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    11,
    OpSchema()
        .Attr(
            "axes",
            "List of integers indicating the dimensions to squeeze. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(data).",
            AttributeProto::INTS,
            OPTIONAL)
        .SetDoc(Squeeze_ver11_doc)
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
          const auto input_ndim = input_shape.dim_size();
          std::transform(
              axes.begin(),
              axes.end(),
              axes.begin(),
              [&](int64_t axis) -> int64_t {
                return axis < 0 ? axis + input_ndim : axis;
              });

          for (int i = 0, j = 0; i < input_ndim; ++i) {
            if (std::find(axes.begin(), axes.end(), i) != axes.end()) {
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

static const char* Unsqueeze_ver11_doc = R"DOC(
Insert single-dimensional entries to the shape of an input tensor (`data`).
Takes one required argument `axes` - which contains a list of dimension indices and this operator will insert a dimension of value `1` into the corresponding index of the output tensor (`expanded`).

For example:
  Given an input tensor (`data`) of shape [3, 4, 5], then
  Unsqueeze(data, axes=[0, 4]) outputs a tensor (`expanded`) containing same data as `data` but with shape [1, 3, 4, 5, 1].

The attribute `axes` should not contain any duplicate entries. It is an error if it contains duplicates.
The rank of the output tensor (`output_rank`) is the rank of the input tensor (`data`) plus the number of values in `axes`.
Each value in `axes` should be within the (inclusive) range [-output_rank , output_rank - 1]. 
The order of values in `axes` does not matter and can come in any order. 

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    11,
    OpSchema()
        .Attr(
            "axes",
            "List of integers indicating the dimensions to be inserted. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(expanded).",
            AttributeProto::INTS)
        .SetDoc(Unsqueeze_ver11_doc)
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

          // validate 'axes' for duplicate entries
          std::unordered_set<int64_t> unique_values;
          for (const auto val : axes) {
            if (unique_values.find(val) != unique_values.end()) {
              fail_shape_inference(
                  "'axes' attribute must not contain any duplicates");
            }
            unique_values.insert(val);
          }

          if (!ctx.getInputType(0)->tensor_type().has_shape()) {
            return;
          }

          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_ndim = input_shape.dim_size();
          const auto output_ndim = input_ndim + static_cast<int>(axes.size());
          for (size_t(i) = 0; i < axes.size(); ++i) {
            if (axes[i] < -output_ndim || axes[i] >= output_ndim) {
              fail_shape_inference(
                  "values in 'axes' are beyond the bounds of the computed output shape");
            }
            if (axes[i] < 0) {
              axes[i] += output_ndim;
            }
          }

          // sort after correcting negative axes values (if any) in the previous step
          std::sort(axes.begin(), axes.end());

          int j = 0;
          for (int i = 0; i < input_ndim; ++i) {
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

static const char* DepthToSpace_ver11_doc =
    R"DOC(DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions. By default, `mode` = `DCR`.
In the DCR mode, elements along the depth dimension from the input tensor are rearranged in the
following order: depth, column, and then row. The output y is computed from the input x as below:

b, c, h, w = x.shape

tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])

tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])

y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])


In the CRD mode, elements along the depth dimension from the input tensor are rearranged in the
following order: column, row, and the depth. The output y is computed from the input x as below:

b, c, h, w = x.shape

tmp = np.reshape(x, [b, c // (blocksize ** 2), blocksize, blocksize, h, w])

tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])

y = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    DepthToSpace,
    11,
    OpSchema()
        .Attr(
            "blocksize",
            "Blocks of [blocksize, blocksize] are moved.",
            AttributeProto::INT)
        .Attr(
            "mode",
            "DCR (default) for depth-column-row order re-arrangement. Use CRD for column-row-depth order.",
            AttributeProto::STRING,
            std::string("DCR"))
        .SetDoc(DepthToSpace_ver11_doc)
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
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          // Shape inference

          // Needs atleast the first input to proceed
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_rank = input_shape.dim_size();

          const auto* repeats_inputs = ctx.getInputData(1);

          auto* output_shape =
              ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

          if (nullptr != repeats_inputs && hasNInputShapes(ctx, 2)) {
            // shape inference is possible only when 'repeats' is an initializer
            const auto& repeats_shape =
                ctx.getInputType(1)->tensor_type().shape();
            if (repeats_shape.dim_size() != 1 ||
                repeats_inputs->data_type() != TensorProto::INT64)
              fail_shape_inference(
                  "'Repeats' input must be 1D tensor of type int64");

            const auto& repeats_data = ParseData<int64_t>(repeats_inputs);

            if (repeats_data.size() != static_cast<size_t>(input_rank))
              fail_shape_inference(
                  "'Repeats' input has incorrect number of values. "
                  "The number of values in 'repeats' must be equal "
                  "to the number of input dimensions.");

            for (size_t i = 0; (int64_t)i < input_rank; ++i) {
              const auto& input_dim = input_shape.dim((int)i);
              auto* output_dim = output_shape->add_dim();
              if (input_dim.has_dim_value()) {
                output_dim->set_dim_value(
                    input_dim.dim_value() * repeats_data[i]);
              }
            }
          } else {
            // Infer output shape's rank in any case (if repeats data is not
            // available)
            auto* output_shape_0 = getOutputShape(ctx, 0);
            for (size_t i = 0; (int64_t)i < input_rank; ++i) {
              output_shape_0->add_dim();
            }
          }
          return;
        }));

static const char* Upsample_ver10_doc = R"DOC(
Upsample the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Upsample,
    10,
    OpSchema()
        .Deprecate()
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
        .SetDoc(Upsample_ver10_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          resizeShapeInference_opset7_to_10(ctx);
        }));

static const char* Resize_ver11_doc = R"DOC(
Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * (roi_end - roi_start) * scale) if input \"sizes\" is not specified.
)DOC";

static const char* Resize_attr_coordinate_transformation_mode_doc = R"DOC(
This attribute describes how to transform the coordinate in the resized tensor to the coordinate in the original tensor. <br/>

The coordinate of each dimension is transformed individually. Let's describe a case using axis x as an example. 
Denote x_resized as the coordinate of axis x in the resized tensor, x_original as the coordinate of axis x in the original tensor, length_original as the length of the original tensor in axis x, length_resized as the length of the resized tensor in axis x, roi_x = (start_x, end_x) of the axis x in input "roi", scale = length_resized / length_original, <br/>

if coordinate_transformation_mode is "half_pixel", <br/>
x_original = (x_resized + 0.5) / scale - 0.5, <br/>

if coordinate_transformation_mode is "pytorch_half_pixel", <br/>
x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0, <br/>

if coordinate_transformation_mode is "align_corners", <br/>
x_original = x_resized * (length_original - 1) / (length_resized - 1), <br/>

if coordinate_transformation_mode is "asymmetric", <br/>
x_original = x_resized / scale, <br/>

if coordinate_transformation_mode is "tf_half_pixel_for_nn", <br/>
x_original = (x_resized + 0.5) / scale, <br/>

if coordinate_transformation_mode is "tf_crop_and_resize", <br/>
x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized * (end_x - start_x) * (length_original - 1) / (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original - 1).)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Resize,
    11,
    OpSchema()
        .Attr(
            "mode",
            "Three interpolation modes: nearest (default), linear and cubic. "
            "The \"linear\" mode includes linear interpolation for 1D tensor and N-linear interpolation for N-D tensor (for example, bilinear interpolation for 2D tensor). "
            "The \"cubic\" mode includes cubic interpolation for 1D tensor and N-cubic interpolation for N-D tensor (for example, bicubic interpolation for 2D tensor).",
            AttributeProto::STRING,
            std::string("nearest"))
        .Attr(
            "cubic_coeff_a",
            "The coefficient 'a' used in cubic interpolation. Two common choice are -0.5 (in some cases of TensorFlow) and -0.75"
            " (in PyTorch). Check out Equation (4) in https://ieeexplore.ieee.org/document/1163711 for the details. "
            "This attribute is valid only if \"mode\" is \"cubic\".",
            AttributeProto::FLOAT,
            static_cast<float>(-0.75))
        .Attr(
            "exclude_outside",
            "If set to 1, the weight of sampling locations outside the tensor will be set to 0"
            " and the weight will be renormalized so that their sum is 1.0. The default value is 0.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "coordinate_transformation_mode",
            Resize_attr_coordinate_transformation_mode_doc,
            AttributeProto::STRING,
            std::string("half_pixel"))
        .Attr(
            "nearest_mode",
            "Four modes: round_prefer_floor (default, as known as round half down), round_prefer_ceil (as known as round half up), floor, ceil. Only used by nearest interpolation. It indicates how to get \"nearest\" pixel in input tensor from x_original, so this attribute is valid only if \"mode\" is \"nearest\".",
            AttributeProto::STRING,
            std::string("round_prefer_floor"))
        .Attr(
            "extrapolation_value",
            "When coordinate_transformation_mode is \"tf_crop_and_resize\" and x_original is outside the range [0, length_original - 1], this value is used as the corresponding output value. Default is 0.0f.",
            AttributeProto::FLOAT,
            static_cast<float>(0))
        .Input(0, "X", "N-D tensor", "T1")
        .Input(
            1,
            "roi",
            "1-D tensor given as [start1, ..., startN, end1, ..., endN], where N is the rank of X. The RoIs' coordinates are normalized in the coordinate system of the input image. It only takes effect when coordinate_transformation_mode is \"tf_crop_and_resize\"",
            "T2")
        .Input(
            2,
            "scales",
            "The scale array along each dimension. It takes value greater than 0. If it's less than 1,"
            " it's sampling down, otherwise, it's upsampling. The number of elements of 'scales' should"
            " be the same as the rank of input 'X'. Only one of 'scales' and 'sizes' can be specified. If 'size' is needed, the user can use an empty string as the name of 'scales' in this operator's input list.",
            "tensor(float)")
        .Input(
            3,
            "sizes",
            "The size of the output tensor. The number of elements of 'sizes' should be the same as the"
            " rank of input 'X'. Only one of 'scales' and 'sizes' can be specified.",
            "tensor(int64)",
            OpSchema::Optional)
        .Output(0, "Y", "N-D tensor after resizing", "T1")
        .TypeConstraint(
            "T1",
            OpSchema::all_tensor_types(),
            "Constrain input 'X' and output 'Y' to all tensor types.")
        .TypeConstraint(
            "T2",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain roi type to float or double.")
        .SetDoc(Resize_ver11_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          resizeShapeInference(ctx, true);
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

static const char* Compress_ver11_doc = R"DOC(
    Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
    In case axis is not provided, input is flattened before elements are selected.
    Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html
    )DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Compress,
    11,
    OpSchema()
        .SetDoc(Compress_ver11_doc)
        .Attr(
            "axis",
            "(Optional) Axis along which to take slices. If not specified, "
            "input is flattened before elements being selected. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(input).",
            AttributeProto::INT,
            OPTIONAL)
        .Input(0, "input", "Tensor of rank r >= 1.", "T")
        .Input(
            1,
            "condition",
            "Rank 1 tensor of booleans to indicate which slices or data elements to be selected. "
            "Its length can be less than the input length along the axis "
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
            "Constrains to boolean tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasInputShape(ctx, 0)) {
            const TensorShapeProto& indices_shape =
                ctx.getInputType(0)->tensor_type().shape();
            int r = indices_shape.dim_size();
            if (r < 1) {
              fail_shape_inference("Indices tensor must have rank >= 1");
            }
            auto axisAttr = ctx.getAttribute("axis");
            if (axisAttr) {
              int axis = static_cast<int>(axisAttr->i());
              if (axis < -r || axis >= r) {
                fail_shape_inference(
                    "'axis' must be in [-rank(indices), rank(indices)-1]");
              }
              if (axis < 0) {
                axis += r;
              }
            }
          }
        }));

static const char* OneHot_ver11_doc = R"DOC(
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
    the range [-depth, depth-1] will result in one-hot representation with all 'off_value' values in the
    output tensor.

    when axis = 0:
    output[input[i, j, k], i, j, k] = 1 for all i, j, k and 0 otherwise.

    when axis = -1:
    output[i, j, k, input[i, j, k]] = 1 for all i, j, k and 0 otherwise.

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    OneHot,
    11,
    OpSchema()
        .SetDoc(OneHot_ver11_doc)
        .Attr(
            "axis",
            "(Optional) Axis along which one-hot representation in added. Default: axis=-1. "
            "axis=-1 means that the additional dimension will be inserted as the "
            "innermost/last dimension in the output tensor. Negative value means counting dimensions "
            "from the back. Accepted range is [-r-1, r] where r = rank(indices).",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .Input(
            0,
            "indices",
            "Input tensor containing indices. Any entries in the 'indices' input tensor with "
            "values outside the range [-depth, depth-1] will result in one-hot representation with all "
            "'off_value' values in the output tensor."
            "In case 'indices' is of non-integer type, the values will be casted to int64 before use.",
            "T1")
        .Input(
            1,
            "depth",
            "Scalar specifying the number of classes in one-hot tensor. This is also the size "
            "of the one-hot dimension (specified by 'axis' attribute) added on in the output "
            "tensor. The values in the 'indices' input tensor are expected to be "
            "in the range [-depth, depth-1]. "
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
          // TODO: Ideally to match spec for this input only Scalar should
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
                  "'axis' must be in [-rank(indices), rank(indices)-1]");
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

ONNX_OPERATOR_SET_SCHEMA(
    IsNaN,
    9,
    OpSchema()
        .SetDoc(R"DOC(Returns which elements of the input are NaN.)DOC")
        .Input(0, "X", "input", "T1")
        .Output(0, "Y", "output", "T2")
        .TypeConstraint(
            "T1",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input types to float tensors.")
        .TypeConstraint(
            "T2",
            {"tensor(bool)"},
            "Constrain output types to boolean tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          updateOutputElemType(ctx, 0, TensorProto::BOOL);
          if (hasInputShape(ctx, 0)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    IsInf,
    10,
    OpSchema()
        .SetDoc(R"DOC(Map infinity to true and other values to false.)DOC")
        .Input(0, "X", "input", "T1")
        .Output(0, "Y", "output", "T2")
        .Attr(
            "detect_positive",
            "(Optional) Whether map positive infinity to true. Default to 1 "
            "so that positive infinity induces true. Set this attribute to 0 "
            "if positive infinity should be mapped to false.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "detect_negative",
            "(Optional) Whether map negative infinity to true. Default to 1 "
            "so that negative infinity induces true. Set this attribute to 0 "
            "if negative infinity should be mapped to false.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeConstraint(
            "T1",
            {"tensor(float)", "tensor(double)"},
            "Constrain input types to float tensors.")
        .TypeConstraint(
            "T2",
            {"tensor(bool)"},
            "Constrain output types to boolean tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          updateOutputElemType(ctx, 0, TensorProto::BOOL);
          if (hasInputShape(ctx, 0)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

static const char* Where_ver9_doc = R"DOC(
    Return elements, either from X or Y, depending on condition
    (with Numpy-style broadcasting support).
    Where behaves like numpy.where with three parameters:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Where,
    9,
    OpSchema()
        .SetDoc(Where_ver9_doc)
        .Input(
            0,
            "condition",
            "When True (nonzero), yield X, otherwise yield Y",
            "B")
        .Input(
            1,
            "X",
            "values selected at indices where condition is True",
            "T")
        .Input(
            2,
            "Y",
            "values selected at indices where condition is False",
            "T")
        .Output(
            0,
            "output",
            "Tensor of shape equal to the broadcasted shape of condition, X, and Y.",
            "T")
        .TypeConstraint("B", {"tensor(bool)"}, "Constrain to boolean tensors.")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 1, 0);
          if (hasNInputShapes(ctx, 3)) {
            std::vector<const TensorShapeProto*> shapes;
            shapes.push_back(&ctx.getInputType(0)->tensor_type().shape());
            shapes.push_back(&ctx.getInputType(1)->tensor_type().shape());
            shapes.push_back(&ctx.getInputType(2)->tensor_type().shape());
            multidirectionalBroadcastShapeInference(
                shapes,
                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
          }
        }));

static const char* NonZero_ver9_doc = R"DOC(
    Returns the indices of the elements that are non-zero
    (in row-major order - by dimension).
    NonZero behaves similar to numpy.nonzero:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    NonZero,
    9,
    OpSchema()
        .SetDoc(NonZero_ver9_doc)
        .Input(0, "X", "input", "T")
        .Output(0, "Y", "output (always 2D tensor)", "tensor(int64)")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          updateOutputElemType(ctx, 0, TensorProto::INT64);
        }));

static const char* ReverseSequence_ver10_doc = R"DOC(
Reverse batch of sequences having different lengths specified by `sequence_lens`.

For each slice i iterating on batch axis, the operator reverses the first sequence_lens[i] elements on time axis,
and copies elements whose index's beyond sequence_lens[i] to the output. So the output slice i contains reversed
sequences on the first sequence_lens[i] elements, then have original values copied for the other elements.

Example 1:
  input = [[0.0, 4.0, 8.0,  12.0],
           [1.0, 5.0, 9.0,  13.0],
           [2.0, 6.0, 10.0, 14.0],
           [3.0, 7.0, 11.0, 15.0]]
  sequence_lens = [4, 3, 2, 1]
  time_axis = 0
  batch_axis = 1

  output = [[3.0, 6.0, 9.0,  12.0],
            [2.0, 5.0, 8.0,  13.0],
            [1.0, 4.0, 10.0, 14.0],
            [0.0, 7.0, 11.0, 15.0]]

Example 2:
  input = [[0.0,  1.0,  2.0,  3.0 ],
           [4.0,  5.0,  6.0,  7.0 ],
           [8.0,  9.0,  10.0, 11.0],
           [12.0, 13.0, 14.0, 15.0]]
  sequence_lens = [1, 2, 3, 4]
  time_axis = 1
  batch_axis = 0

  output = [[0.0,  1.0,  2.0,  3.0 ],
            [5.0,  4.0,  6.0,  7.0 ],
            [10.0, 9.0,  8.0,  11.0],
            [15.0, 14.0, 13.0, 12.0]]
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ReverseSequence,
    10,
    OpSchema()
        .SetDoc(ReverseSequence_ver10_doc)
        .Attr(
            "time_axis",
            "(Optional) Specify which axis is time axis. Must be one of 0 (default), or 1.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "batch_axis",
            "(Optional) Specify which axis is batch axis. Must be one of 1 (default), or 0.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Input(0, "input", "Tensor of rank r >= 2.", "T")
        .Input(
            1,
            "sequence_lens",
            "Tensor specifying lengths of the sequences in a batch. It has shape `[batch_size]`.",
            "tensor(int64)")
        .Output(0, "Y", "Tensor with same shape of input.", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Input and output types can be of any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 2)) {
            return;
          }

          auto& first_input_shape = getInputShape(ctx, 0);
          if (first_input_shape.dim_size() < 2) {
            fail_shape_inference("'input' must have rank >= 2");
          }
          auto& seq_len_input_shape = getInputShape(ctx, 1);
          if (seq_len_input_shape.dim_size() != 1) {
            fail_shape_inference("'sequence_lens' must have rank of 1");
          }

          propagateShapeFromInputToOutput(ctx, 0, 0);
        }));

static const char* Unique_ver11_doc = R"DOC(
Find the unique elements of a tensor. When an optional attribute 'axis' is provided, unique subtensors sliced along the 'axis' are returned. 
Otherwise the input tensor is flattened and unique values of the flattened tensor are returned. 

This operator returns the unique values or sliced unique subtensors of the input tensor and three optional outputs. 
The first output tensor 'Y' contains all unique values or subtensors of the input. 
The second optional output tensor 'indices' contains indices of 'Y' elements' first occurance in 'X'.. 
The third optional output tensor 'inverse_indices' contains, for elements of 'X', its corresponding indices in 'Y'. ". 
The fourth optional output tensor 'counts' contains the count of each element of 'Y' in the input. 

Outputs are either sorted in ascending order or optionally in the order of the first occurrence of the values in the input. 

https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html

Example 1:
  input_X = [2, 1, 1, 3, 4, 3]
  attribute_sorted = 0
  attribute_axis = None
  output_Y = [2, 1, 3, 4]
  output_indices = [0, 1, 3, 4]
  output_inverse_indices = [0, 1, 1, 2, 3, 2]
  output_counts = [1, 2, 2, 1]

Example 2:
  input_X = [[1, 3], [2, 3]]
  attribute_sorted = 1
  attribute_axis = None
  output_Y = [1, 2, 3]
  output_indices = [0, 2, 1]
  output_inverse_indices = [0, 2, 1, 2]
  output_counts = [1, 1, 2]

Example 3:
  input_X = [[1, 0, 0], [1, 0, 0], [2, 3, 4]]
  attribute_sorted = 1
  attribute_axis = 0
  output_Y = [[1, 0, 0], [2, 3, 4]]
  output_indices = [0, 2]
  output_inverse_indices = [0, 0, 1]
  output_counts = [2, 1]

Example 4:
  input_x = [[[1., 1.], [0., 1.], [2., 1.], [0., 1.]], 
             [[1., 1.], [0., 1.], [2., 1.], [0., 1.]]]
  attribute_sorted = 1
  attribute_axis = 1

  intermediate data are presented below for better understanding: 
  
  there are 4 subtensors sliced along axis 1 of input_x (shape = (2, 4, 2)):
  A: [[1, 1], [1, 1]], 
     [[0, 1], [0, 1]], 
     [[2, 1], [2, 1]], 
     [[0, 1], [0, 1]].
  
  there are 3 unique subtensors: 
  [[1, 1], [1, 1]], 
  [[0, 1], [0, 1]], 
  [[2, 1], [2, 1]].
  
  sorted unique subtensors:
  B: [[0, 1], [0, 1]], 
     [[1, 1], [1, 1]], 
     [[2, 1], [2, 1]].
  
  output_Y is constructed from B:
  [[[0. 1.], [1. 1.], [2. 1.]], 
   [[0. 1.], [1. 1.], [2. 1.]]]

  output_indices is to map from B to A:
  [1, 0, 2]
  
  output_inverse_indices is to map from A to B:
  [1, 0, 2, 0]

  output_counts = [2 1 1]
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Unique,
    11,
    OpSchema()
        .SetDoc(Unique_ver11_doc)
        .Attr(
            "sorted",
            "(Optional) Whether to sort the unique elements in ascending order before returning as output. "
            "Must be one of 0, or 1 (default).",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "axis",
            "(Optional) The dimension to apply unique. If not specified, the unique elements of the "
            "flattened input are returned. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(input).",
            AttributeProto::INT,
            OPTIONAL)
        .Input(0, "X", "A N-D input tensor that is to be processed.", "T")
        .Output(
            0,
            "Y",
            "A tensor of the same type as 'X' "
            "containing all the unique values or subtensors sliced along a provided 'axis' in 'X', either sorted "
            "or maintained in the same order they occur in input 'X'",
            "T")
        .Output(
            1,
            "indices",
            "A 1-D INT64 tensor "
            "containing indices of 'Y' elements' first occurance in 'X'. "
            "When 'axis' is provided, it contains indices to subtensors in input 'X' on the 'axis'. "
            "When 'axis' is not provided, it contains indices to values in the flattened input tensor. ",
            "tensor(int64)",
            OpSchema::Optional)
        .Output(
            2,
            "inverse_indices",
            "A 1-D INT64 tensor "
            "containing, for elements of 'X', its corresponding indices in 'Y'. "
            "When 'axis' is provided, it contains indices to subtensors in output 'Y' on the 'axis'. "
            "When 'axis' is not provided, it contains indices to values in output 'Y'. ",
            "tensor(int64)",
            OpSchema::Optional)
        .Output(
            3,
            "counts",
            "A 1-D INT64 tensor containing "
            "the count of each element "
            "of 'Y' in input 'X'",
            "tensor(int64)",
            OpSchema::Optional)
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Input can be of any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          const TypeProto* xTensorProto = ctx.getInputType(0);
          TypeProto* yTensorProto = ctx.getOutputType(0);
          TypeProto* indicesTensorProto = nullptr;
          TypeProto* inverseIndicesTensorProto = nullptr;
          TypeProto* countsTensorProto = nullptr;

          // 'indices', 'inverse_indices', and 'counts' are 1-D tensors of
          // unknown dimension.
          auto num_outputs = ctx.getNumOutputs();
          if (num_outputs >= 2) {
            indicesTensorProto = ctx.getOutputType(1);
            updateOutputElemType(ctx, 1, TensorProto::INT64);
            indicesTensorProto->mutable_tensor_type()
                ->mutable_shape()
                ->add_dim();
          }

          if (num_outputs >= 3) {
            inverseIndicesTensorProto = ctx.getOutputType(2);
            updateOutputElemType(ctx, 2, TensorProto::INT64);
            inverseIndicesTensorProto->mutable_tensor_type()
                ->mutable_shape()
                ->add_dim();
          }

          if (num_outputs >= 4) {
            countsTensorProto = ctx.getOutputType(3);
            updateOutputElemType(ctx, 3, TensorProto::INT64);
            countsTensorProto->mutable_tensor_type()
                ->mutable_shape()
                ->add_dim();
          }

          auto axisAttr = ctx.getAttribute("axis");
          if (!axisAttr) {
            // 'axis' is not provided. Input 'X' is flattened.
            // 'Y' is a 1-D tensor of unknown dimension.
            yTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
          } else {
            // 'axis' is provided.
            int axis = static_cast<int>(axisAttr->i());
            const TensorShapeProto& input_shape =
                xTensorProto->tensor_type().shape();
            int rank = input_shape.dim_size();
            if (axis < 0)
              axis += rank;
            if (axis < 0 || axis >= rank)
              fail_shape_inference("Invalid value for attribute axis");

            // 'Y' has the same shape as 'X' except in the 'axis' dimension
            // which is unknown.
            for (int i = 0; i < input_shape.dim_size(); i++) {
              auto* dim = yTensorProto->mutable_tensor_type()
                              ->mutable_shape()
                              ->add_dim();
              if (i != axis) {
                *dim = input_shape.dim(i);
              }
            }
          }
        }));

static const char* GatherND_ver11_doc = R"DOC(
Given `data` tensor of rank `r` >= 1, and `indices` tensor of rank `q` >= 1, this operator gathers 
slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1`.

`indices` is an q-dimensional integer tensor, best thought of as a `(q-1)`-dimensional tensor of index-tuples into `data`, 
where each element defines a slice of `data`

Some salient points about the inputs' rank and shape:
 
1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks `r` and `q`

2) The `indices_shape[-1]` should have a value between 1 (inclusive) and rank `r` (inclusive) 

3) All values in `indices` are expected to be within bounds [-s, s-1] along axis of size `s` (i.e.) `-data_shape[i] <= indices[...,i] <= data_shape[i] - 1`.
   It is an error if any of the index values are out of bounds.

The output is computed as follows:

The output tensor is obtained by mapping each index-tuple in the `indices` tensor to the corresponding slice of the input `data`.
 
1) If `indices_shape[-1] > r` => error condition

2) If `indices_shape[-1] == r`, since the rank of `indices` is `q`, `indices` can be thought of as a `(q-1)`-dimensional tensor
   containing 1-D tensors of dimension `r`. Let us think of each such `r` ranked tensor as `indices_slice`. 
   Each *scalar value* corresponding to `data[indices_slice]` is filled into the corresponding location of the `(q-1)`-dimensional tensor 
   to form the `output` tensor (Example 1 below)

3) If `indices_shape[-1] < r`, since the rank of `indices` is `q`, `indices` can be thought of as a `(q-1)`-dimensional tensor
   containing 1-D tensors of dimension `< r`. Let us think of each such tensors as `indices_slice`. 
   Each *tensor slice* corresponding to `data[indices_slice , :]` is filled into the corresponding location of the `(q-1)`-dimensional tensor 
   to form the `output` tensor (Examples 2, 3, and 4 below)

This operator is the inverse of `ScatterND`.

`Example 1`

  data    = [[0,1],[2,3]]   # data_shape = [2, 2]

  indices = [[0,0],[1,1]]   # indices_shape = [2, 2]

  output  = [0,3]           # output_shape = [2]

`Example 2`

  data    = [[0,1],[2,3]]  # data_shape = [2, 2]

  indices = [[1],[0]]      # indices_shape = [2, 1]

  output  = [[2,3],[0,1]]  # output_shape = [2, 2]

`Example 3`

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]

  output  = [[2,3],[4,5]]                 # output_shape = [2, 2]   

`Example 4`

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]

  output  = [[[2,3]],[[4,5]]]             # output_shape = [2, 1, 2] 

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    GatherND,
    11,
    OpSchema()
        .SetDoc(GatherND_ver11_doc)
        .Input(0, "data", "Tensor of rank r >= 1.", "T")
        .Input(
            1,
            "indices",
            "Tensor of rank q >= 1. All index values are expected to be within bounds [-s, s-1] "
            "along axis of size s. It is an error if any of the index values are out of bounds.",
            "tensor(int64)")
        .Output(
            0,
            "output",
            "Tensor of rank q + r - indices_shape[-1] - 1.",
            "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain input and output types to any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Shape inference
          if (!hasNInputShapes(ctx, 2)) {
            // cannot proceed with shape or rank inference
            return;
          }

          const auto& data_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto data_rank = data_shape.dim_size();

          const auto& indices_shape =
              ctx.getInputType(1)->tensor_type().shape();
          const auto indices_rank = indices_shape.dim_size();

          if (data_rank < 1 || indices_rank < 1) {
            fail_shape_inference(
                "Both `data` and `indices` input tensors in GatherND op "
                "need to have rank larger than 0.");
          }

          // cannot ascertain if the input shapes are valid if shape of
          // `indices` is missing last dimension value so return at this point
          if (!indices_shape.dim(indices_rank - 1).has_dim_value()) {
            return;
          }

          const auto last_index_dimension =
              indices_shape.dim(indices_rank - 1).dim_value();

          if (last_index_dimension > data_rank) {
            fail_shape_inference(
                "Last dimension of `indices` input tensor in GatherND op "
                "must not be larger than the rank of `data` tensor");
          }

          for (int i = 0; i < indices_rank - 1; ++i) {
            *ctx.getOutputType(0)
                 ->mutable_tensor_type()
                 ->mutable_shape()
                 ->add_dim() = indices_shape.dim(i);
          }

          for (int i = static_cast<int>(last_index_dimension); i < data_rank;
               ++i) {
            *ctx.getOutputType(0)
                 ->mutable_tensor_type()
                 ->mutable_shape()
                 ->add_dim() = data_shape.dim(i);
          }
        }));

static const char* Pad_ver11_doc = R"DOC(
Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`, 
a padded tensor (`output`) is generated.

The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):

1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0)

2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

3) `edge` - pads with the edge values of array


Example 1 (`constant` mode):
  Insert 0 pads to the beginning of the second dimension.

  data = 
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ] 

  pads = [0, 2, 0, 0]

  mode = 'constant'

  constant_value = 0.0

  output = 
  [
      [
          [0.0, 0.0, 1.0, 1.2],
          [0.0, 0.0, 2.3, 3.4],
          [0.0, 0.0, 4.5, 5.7],
      ],
  ]


Example 2 (`reflect` mode):
  data = 
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ] 

  pads = [0, 2, 0, 0]

  mode = 'reflect'

  output = 
  [
      [
          [1.0, 1.2, 1.0, 1.2],
          [2.3, 3.4, 2.3, 3.4],
          [4.5, 5.7, 4.5, 5.7],
      ],
  ]


Example 3 (`edge` mode):
  data = 
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ] 

  pads = [0, 2, 0, 0]

  mode = 'edge'

  output = 
  [
      [
          [1.0, 1.0, 1.0, 1.2],
          [2.3, 2.3, 2.3, 3.4],
          [4.5, 4.5, 4.5, 5.7],
      ],
  ]

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Pad,
    11,
    OpSchema()
        .Attr(
            "mode",
            "Supported modes: `constant`(default), `reflect`, `edge`",
            AttributeProto::STRING,
            std::string("constant"))
        .SetDoc(Pad_ver11_doc)
        .Input(0, "data", "Input tensor.", "T")
        .Input(
            1,
            "pads",
            "Tensor of integers indicating the number of padding elements to add or remove (if negative) "
            "at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. "
            "`pads` should be a 1D tensor of shape [2 * input_rank]. "
            "`pads` format should be: [x1_begin, x2_begin,...,x1_end, x2_end,...], "
            "where xi_begin is the number of pad values added at the beginning of axis `i` and "
            "xi_end, the number of pad values added at the end of axis `i`.",
            "tensor(int64)")
        .Input(
            2,
            "constant_value",
            "(Optional) A scalar value to be used if the mode chosen is `constant` (by default it is 0).",
            "T",
            OpSchema::Optional)
        .Output(0, "output", "Tensor after padding.", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types(),
            "Constrains input and output to only numeric types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          // Shape inference needs the input data shape
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_rank = input_shape.dim_size();

          // Infer output shape if 'pads' tensor is available
          const auto* pads_initializer = ctx.getInputData(1);
          if (nullptr != pads_initializer) {
            if (pads_initializer->dims_size() != 1 ||
                pads_initializer->data_type() != TensorProto::INT64)
              fail_shape_inference(
                  "'pads' input must be a 1D (shape: [2 * input_rank]) tensor of type int64");

            const auto& pads_data = ParseData<int64_t>(pads_initializer);

            if (pads_data.size() != static_cast<size_t>(2 * input_rank))
              fail_shape_inference("Pads has incorrect number of values");

            auto* output_shape =
                ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
            for (size_t i = 0; static_cast<int64_t>(i) < input_rank; ++i) {
              const auto& input_dim = input_shape.dim((int)i);
              auto* output_dim = output_shape->add_dim();
              if (input_dim.has_dim_value()) {
                output_dim->set_dim_value(
                    input_dim.dim_value() + pads_data[i] +
                    pads_data[i + input_rank]);
              } else if (pads_data[i] + pads_data[i + input_rank] == 0) {
                *output_dim = input_dim;
              }
            }
          } else {
            // Infer output shapes' rank in any case
            auto* output_shape_0 = getOutputShape(ctx, 0);
            for (size_t i = 0; static_cast<int64_t>(i) < input_rank; ++i) {
              output_shape_0->add_dim();
            }
          }
          return;
        }));

} // namespace ONNX_NAMESPACE