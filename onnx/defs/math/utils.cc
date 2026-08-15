// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/math/utils.h"

#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx/common/safe_math.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE::defs::math::utils {

static constexpr const char* TopK_ver11_doc = R"DOC(
Retrieve the top-K largest or smallest elements along a specified axis. Given an input tensor of
shape [a_0, a_1, ..., a_{n-1}] and integer argument k, return two outputs:

* Value tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}]
  which contains the values of the top k elements along the specified axis
* Index tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] which
  contains the indices of the top k elements (original indices from the input
  tensor).

* If "largest" is 1 (the default value) then the k largest elements are returned.
* If "sorted" is 1 (the default value) then the resulting k elements will be sorted.
* If "sorted" is 0, order of returned 'Values' and 'Indices' are undefined.

Given two equivalent values, this operator uses the indices along the axis as
a tiebreaker. That is, the element with the lower index will appear first.
)DOC";

std::function<void(OpSchema&)> TopKOpGenerator(std::vector<std::string> allowed_types) {
  return [allowed_types = std::move(allowed_types)](OpSchema& schema) {
    schema.SetDoc(TopK_ver11_doc)
        .Input(
            0,
            "X",
            "Tensor of shape [a_0, a_1, ..., a_{n-1}]",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "K",
            "A 1-D tensor containing a single positive value corresponding to the number of top elements to retrieve",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "Values",
            "Tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] "
            "containing top K values from the input tensor",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(
            1,
            "Indices",
            "Tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] "
            "containing the corresponding input tensor indices for the top K "
            "values.",
            "I",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint("T", allowed_types, "Constrain input and output types to numeric tensors.")
        .TypeConstraint("I", {types::Int64}, "Constrain index tensor to int64")
        .Attr(
            "axis",
            "Dimension on which to do the sort. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(input).",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .Attr(
            "largest",
            "Whether to return the top-K largest or smallest elements.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr("sorted", "Whether to return the elements in sorted order.", AttributeProto::INT, static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference:
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          updateOutputElemType(ctx, 1, TensorProto::INT64);
          // Shape inference:
          if (!hasInputShape(ctx, 0))
            return;
          const auto& input_shape = getInputShape(ctx, 0);
          int64_t rank = input_shape.dim_size();
          int64_t axis = getAttribute(ctx, "axis", -1);
          if (axis < 0)
            axis += rank;
          if (axis < 0 || axis >= rank) {
            fail_shape_inference("Invalid value for attribute axis");
          }

          const auto& axis_dim = input_shape.dim(static_cast<int>(axis));
          const auto* const k = ctx.getInputData(1);

          // Infer output shape if:
          // (1) 'K' is available
          // (2) axis_dim has dim value
          // Otherwise cannot reliably compute output shape as axis dim value is
          // unknown and hence cannot determine if axis dim value >= k (which
          // should be enforced)
          if (nullptr != k && axis_dim.has_dim_value()) {
            int64_t k_value = 0;
            if (k->dims_size() != 1 || k->dims(0) != 1) {
              fail_shape_inference("K input must be a one-dimensional tensor of size 1.");
            }
            if (k->data_type() == TensorProto::INT64) {
              const auto data = ParseData<int64_t>(k);
              k_value = data[0];
            } else {
              fail_shape_inference("K input must be of type int64.");
            }
            if (axis_dim.dim_value() < k_value) {
              fail_shape_inference("Axis has less than the requested k elements.");
            }

            TensorShapeProto result_shape = input_shape;
            result_shape.mutable_dim(static_cast<int>(axis))->set_dim_value(k_value);

            updateOutputShape(ctx, 0, result_shape);
            updateOutputShape(ctx, 1, result_shape);

            return;
          }

          // Infer output shapes' rank in any case
          auto* output_shape_0 = getOutputShape(ctx, 0);
          auto* output_shape_1 = getOutputShape(ctx, 1);
          for (int i = 0; i < input_shape.dim_size(); ++i) {
            output_shape_0->add_dim();
            output_shape_1->add_dim();
          }

          return;
        });
  };
}

std::function<void(OpSchema&)>
UnaryFloatMathOpGenerator(const char* doc, const char* output_description, std::vector<std::string> allowed_types) {
  return [doc, output_description, allowed_types = std::move(allowed_types)](OpSchema& schema) {
    schema.SetDoc(doc)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "output", output_description, "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", allowed_types, "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput);
  };
}

int64_t MathOpTwoIntegers(const std::string& op_type, int64_t a, int64_t b) {
  bool (*checked_op)(int64_t, int64_t, int64_t*) = nullptr;
  if (op_type == "Add") {
    checked_op = checked_add_overflow;
  } else if (op_type == "Sub") {
    checked_op = checked_sub_overflow;
  } else if (op_type == "Mul") {
    checked_op = checked_mul_overflow;
  } else {
    fail_shape_inference("Wrong op_type name for running propagation: ", op_type);
  }

  int64_t result = 0;
  if (checked_op(a, b, &result)) {
    fail_shape_inference("Integer overflow in ", op_type, " during data propagation");
  }
  return result;
}

void MatMulShapeInference(ONNX_NAMESPACE::InferenceContext& ctx, int input1Idx, int input2Idx) {
  if (!hasInputShape(ctx, input1Idx) || !hasInputShape(ctx, input2Idx)) {
    return;
  }

  const auto shape0 = ctx.getInputType(input1Idx)->tensor_type().shape();
  const auto shape1 = ctx.getInputType(input2Idx)->tensor_type().shape();

  if (shape0.dim_size() == 0 || shape1.dim_size() == 0) {
    fail_shape_inference("Input tensors of wrong rank (0).");
  }

  ONNX_NAMESPACE::TensorShapeProto shapeL, shapeR;

  // First promote each shape to at least rank-2. This logic is
  // specific to matmul, not generic broadcasting.
  {
    if (shape0.dim_size() == 1) {
      shapeL.add_dim()->set_dim_value(1);
      *shapeL.add_dim() = shape0.dim(0);
    } else {
      *shapeL.mutable_dim() = shape0.dim();
    }
    if (shape1.dim_size() == 1) {
      *shapeR.add_dim() = shape1.dim(0);
      shapeR.add_dim()->set_dim_value(1);
    } else {
      *shapeR.mutable_dim() = shape1.dim();
    }
  }

  // Check for compatible matrix multiply dimensions
  {
    const auto& dimL = shapeL.dim(shapeL.dim_size() - 1);
    const auto& dimR = shapeR.dim(shapeR.dim_size() - 2);
    if (dimL.has_dim_value() && dimR.has_dim_value() && dimL.dim_value() != dimR.dim_value()) {
      fail_shape_inference("Incompatible dimensions for matrix multiplication");
    }
  }

  ONNX_NAMESPACE::TensorShapeProto resultShape;

  // Now call out to generic multidimensional broadcasting for
  // the broadcastable prefixes.
  {
    ONNX_NAMESPACE::TensorShapeProto prefixShapeL, prefixShapeR;
    for (int i = 0; i < shapeL.dim_size() - 2; ++i) {
      *prefixShapeL.add_dim() = shapeL.dim(i);
    }
    for (int i = 0; i < shapeR.dim_size() - 2; ++i) {
      *prefixShapeR.add_dim() = shapeR.dim(i);
    }
    bidirectionalBroadcastShapeInference(prefixShapeL, prefixShapeR, resultShape);
  }

  // Back to matmul-specific. Add the trailing dimensions back in.
  {
    if (shape0.dim_size() != 1) {
      *resultShape.add_dim() = shapeL.dim(shapeL.dim_size() - 2);
    }
    if (shape1.dim_size() != 1) {
      *resultShape.add_dim() = shapeR.dim(shapeR.dim_size() - 1);
    }
  }

  *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = resultShape;
}

void QLinearMatMulShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  const auto* const a_type = ctx.getInputType(0);
  const auto* const b_type = ctx.getInputType(3);
  if (nullptr == a_type || nullptr == b_type || a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
      b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
    fail_type_inference("inputs are expected to have tensor type.");
  }

  const auto* const a_zero_point_type = ctx.getInputType(2);
  if (nullptr == a_zero_point_type ||
      a_zero_point_type->tensor_type().elem_type() != a_type->tensor_type().elem_type()) {
    fail_type_inference("input and zero_point pair is expected to have be same type.");
  }

  const auto* const b_zero_point_type = ctx.getInputType(5);
  if (nullptr == b_zero_point_type ||
      b_zero_point_type->tensor_type().elem_type() != b_type->tensor_type().elem_type()) {
    fail_type_inference("input and zero_point pair is expected to have same type.");
  }

  propagateElemTypeFromInputToOutput(ctx, 7, 0);

  MatMulShapeInference(ctx, 0, 3);
}

static constexpr const char* Einsum_ver12_doc = R"DOC(
An einsum of the form `term1, term2 -> output-term` produces an output tensor using the following equation

```
output[output-term] = reduce-sum( input1[term1] * input2[term2] )
```

where the reduce-sum performs a summation over all the indices occurring in the input terms (term1, term2)
that do not occur in the output-term.

The Einsum operator evaluates algebraic tensor operations on a sequence of tensors, using the Einstein summation
convention. The equation string contains a comma-separated sequence of lower case letters. Each term corresponds to
an operand tensor, and the characters within the terms correspond to operands dimensions.

This sequence may be followed by "->" to separate the left and right hand side of the equation.
If the equation contains "->" followed by the right-hand side, the explicit (not classical) form of the Einstein
summation is performed, and the right-hand side indices indicate output tensor dimensions. In other cases,
output indices are (implicitly) set to the alphabetically sorted sequence of indices appearing exactly once in the
equation.

When a dimension character is repeated in the left-hand side, it represents summation along the dimension.

The equation may contain ellipsis ("...") to enable broadcasting. Ellipsis must indicate a fixed number of dimensions.
Specifically, every occurrence of ellipsis in the equation must represent the same number of dimensions.
The right-hand side may contain exactly one ellipsis. In implicit mode, the ellipsis dimensions are set to the
beginning of the output. The equation string may contain space (U+0020) character.
)DOC";

static void einsumShapeInference(ONNX_NAMESPACE::InferenceContext& ctx, std::string const& equation) {
  // Only accept letters for indices
  auto is_letter = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };

  const size_t num_inputs = ctx.getNumInputs();
  if (num_inputs < 1 || !hasNInputShapes(ctx, num_inputs)) {
    return;
  }
  ONNX_NAMESPACE::TensorShapeProto output_shape;
  std::string left_equation;

  auto mid_index = equation.find("->");
  if (mid_index != std::string::npos) {
    // Separate right and left hand sides of the equation
    left_equation = equation.substr(0, mid_index);
  } else {
    // No right hand side
    left_equation = equation;
  }

  std::string term;
  size_t num_operands = 0;
  size_t num_ellipsis = 0;
  size_t num_ellipsis_indices = 0;

  // Parse the left-hand side
  std::stringstream str(left_equation);
  std::map<char, int> label_maps;
  std::unordered_set<char> repeated_labels;
  ONNX_NAMESPACE::TensorShapeProto dims_value, ellipsis_dims_value;
  int num_labels = 0;
  bool ellipsis_flag = true;

  while (!str.eof()) {
    std::getline(str, term, ',');
    auto ellipsis_index = term.find("...");
    if (num_inputs <= num_operands) {
      fail_shape_inference("Number of input tensors does not match the operands in the equation.");
    }
    const auto& shape = ctx.getInputType(num_operands)->tensor_type().shape();
    size_t rank = shape.dim_size();
    size_t ellipsis_dims = 0;

    size_t term_size = 0; // number of legal indices for the current term
    size_t num_illegal_char = 0; // number of illegal char before the current 'index' in the current term

    for (char index : term) {
      if (is_letter(index)) {
        term_size += 1;
      }
    }

    // Validate that term_size is compatible with rank before accessing dimensions
    if (ellipsis_index != std::string::npos) {
      // For ellipsis case, rank must be at least term_size
      if (rank < term_size) {
        fail_shape_inference(
            "Ellipsis represents incompatible dimensions for input ",
            num_operands,
            ". Rank ",
            rank,
            " is less than term size ",
            term_size,
            ".");
      }
    } else {
      // For non-ellipsis case, rank must equal term_size
      if (rank != term_size) {
        fail_shape_inference(
            "Rank of input ", num_operands, " (", rank, ") does not match the equation indices (", term_size, ").");
      }
    }

    for (size_t index = 0; index < term.size(); ++index) {
      if (index == ellipsis_index) {
        // find ellipsis and record the dims represented by ellipsis
        ellipsis_dims = rank - term_size;
        if (ellipsis_flag) {
          ellipsis_flag = false;
          for (size_t i = 0; i < ellipsis_dims; i++) {
            *ellipsis_dims_value.add_dim() = shape.dim(static_cast<int>(index + i - num_illegal_char));
          }
        } else {
          for (size_t i = 0; i < ellipsis_dims; i++) {
            const auto shape_dim = shape.dim(static_cast<int>(index + i - num_illegal_char));
            auto* const current_dim = ellipsis_dims_value.mutable_dim(static_cast<int>(i));
            if (shape_dim.has_dim_value() && current_dim->has_dim_value() &&
                shape_dim.dim_value() > current_dim->dim_value() && current_dim->dim_value() == 1) {
              current_dim->set_dim_value(shape_dim.dim_value());
            }
          }
        }
        index += 2; // skip the rest of dots
        num_illegal_char += 3;
        continue;

      } else if (!is_letter(term[index])) {
        num_illegal_char += 1;
        continue;
      }

      const auto inserted = label_maps.emplace(term[index], num_labels).second;
      if (inserted) {
        *dims_value.add_dim() = shape.dim(static_cast<int>(index + ellipsis_dims - num_illegal_char));
        ++num_labels;
      } else {
        repeated_labels.insert(term[index]);
      }
    }

    if (ellipsis_index != std::string::npos) {
      // If there is an ellipsis, the number of dimensions it represents
      // must be total dim - letter dimensions
      if (num_ellipsis == 0) {
        num_ellipsis_indices = rank - term_size;
      } else { // ellipsis has been seen before. Check that if dimensions
               // are compatible
        if (num_ellipsis_indices != rank - term_size) {
          fail_shape_inference("Ellipsis represents incompatible dimensions.");
        }
      }
      num_ellipsis++;
    }
    num_operands++;
  }

  if (num_inputs != num_operands) {
    fail_shape_inference("Number of input tensors does not match the operands in the equation.");
  }

  // Parse the provided right-hand side
  if (mid_index != std::string::npos) {
    std::string right_equation = equation.substr(mid_index + 2);
    auto right_ellipsis_index = right_equation.find("...");

    for (size_t index = 0; index < right_equation.size(); ++index) {
      // If there's an ellipsis, add its corresponding dimensions
      if (index == right_ellipsis_index) {
        for (size_t i = 0; i < num_ellipsis_indices; i++) {
          *output_shape.add_dim() = ellipsis_dims_value.dim(static_cast<int>(i));
        }
        index += 2; // skip the rest of dots
        continue;
      }

      if (is_letter(right_equation[index])) {
        auto it = label_maps.find(right_equation[index]);
        if (it == label_maps.end()) {
          fail_shape_inference("Equation output contains a label missing from the inputs");
        }
        *output_shape.add_dim() = dims_value.dim(it->second);
      }
    }
  } else { // Infer the dimension for right-hand side
    // If there's an ellipsis, add its corresponding dimensions
    for (size_t i = 0; i < num_ellipsis_indices; i++) {
      *output_shape.add_dim() = ellipsis_dims_value.dim(static_cast<int>(i));
    }
    // If no explicit output was given, generate an implicit output by ordering all the
    // labels in alphabetic order (by ASCII value consistent with numpy, so Z < a).
    // Exclude any labels that occurred more than once, as these cancel out.
    for (const auto& [label, dim_idx] : label_maps) {
      if (repeated_labels.count(label) == 0) {
        *output_shape.add_dim() = dims_value.dim(dim_idx);
      }
    }
  }

  updateOutputShape(ctx, 0, output_shape);
}

std::function<void(OpSchema&)> EinsumOpGenerator(std::vector<std::string> allowed_types) {
  return [allowed_types = std::move(allowed_types)](OpSchema& schema) {
    schema.SetDoc(Einsum_ver12_doc)
        .Attr("equation", "Einsum expression string.", AttributeProto::STRING)
        .Input(0, "Inputs", "Operands", "T", OpSchema::Variadic, true, 1, OpSchema::Differentiable)
        .Output(0, "Output", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", allowed_types, "Constrain input and output types to all numerical tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          std::string equation = getAttribute(ctx, "equation", "");
          if (equation.empty()) {
            return;
          }

          equation.erase(std::remove(equation.begin(), equation.end(), ' '),
                         equation.end()); // Remove space char
          einsumShapeInference(ctx, equation);
        });
  };
}

const char* QLinearMatMulDoc() {
  static constexpr const char* QLinearMatMul_doc = R"DOC(
Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
It consumes two quantized input tensors, their scales and zero points, scale and zero point of output,
and computes the quantized output. The quantization formula is y = saturate((x / y_scale) + y_zero_point).
For (x / y_scale), it is rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
Scale and zero point must have same shape. They must be either scalar (per tensor) or N-D tensor
(per row for 'a' and per column for 'b'). Scalar refers to per tensor quantization whereas N-D refers to per row
or per column quantization. If the input is 2D of shape [M, K] then zero point and scale tensor may be
an M element vector [v_1, v_2, ..., v_M] for per row quantization and K element vector of shape [v_1, v_2, ..., v_K]
for per column quantization. If the input is N-D tensor with shape [D1, D2, M, K] then zero point and scale tensor may
have shape [D1, D2, M, 1] for per row quantization and shape [D1, D2, 1, K] for per column quantization.
Production must never overflow, and accumulation may overflow if and only if in 32 bits.
)DOC";
  return QLinearMatMul_doc;
}

} // namespace ONNX_NAMESPACE::defs::math::utils
