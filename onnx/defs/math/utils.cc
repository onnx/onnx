// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/math/utils.h"

#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnx/common/safe_math.h"
#include "onnx/defs/function.h"
#include "onnx/defs/printer.h"
#include "onnx/defs/tensor_proto_util.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE::defs::math::utils {

void TopKShapeInference(InferenceContext& ctx) {
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

  int64_t k_value = 0;
  if (nullptr != k) {
    if (k->dims_size() != 1 || k->dims(0) != 1) {
      fail_shape_inference("K input must be a one-dimensional tensor of size 1.");
    }
    if (k->data_type() == TensorProto::INT64) {
      const auto data = ParseData<int64_t>(k);
      k_value = data[0];
    } else {
      fail_shape_inference("K input must be of type int64.");
    }
    if (k_value <= 0) {
      fail_shape_inference("K input must contain a single positive value.");
    }
  }

  // Infer output shape if:
  // (1) 'K' is available
  // (2) axis_dim has dim value
  // Otherwise cannot reliably compute output shape as axis dim value is
  // unknown and hence cannot determine if axis dim value >= k (which
  // should be enforced)
  if (nullptr != k && axis_dim.has_dim_value()) {
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
}

std::function<void(OpSchema&)> TopKOpGenerator() {
  return [](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(TopKShapeInference); };
}

std::function<void(OpSchema&)> UnaryFloatMathOpGenerator() {
  return [](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput); };
}

int64_t MathOpTwoIntegers(std::string_view op_type, int64_t a, int64_t b) {
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

  *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = std::move(resultShape);
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

static void einsumShapeInference(ONNX_NAMESPACE::InferenceContext& ctx, std::string_view equation) {
  // Only accept letters for indices
  auto is_letter = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };

  const size_t num_inputs = ctx.getNumInputs();
  if (num_inputs < 1 || !hasNInputShapes(ctx, num_inputs)) {
    return;
  }
  ONNX_NAMESPACE::TensorShapeProto output_shape;
  const auto mid_index = equation.find("->");
  auto left_equation = equation.substr(0, mid_index);

  size_t num_operands = 0;
  size_t num_ellipsis = 0;
  size_t num_ellipsis_indices = 0;
  size_t first_ellipsis_operand = 0;

  // Parse the left-hand side
  std::map<char, int> label_maps;
  std::unordered_set<char> repeated_labels;
  ONNX_NAMESPACE::TensorShapeProto dims_value, ellipsis_dims_value;
  int num_labels = 0;
  bool ellipsis_flag = true;

  while (true) {
    const auto comma = left_equation.find(',');
    const auto term = left_equation.substr(0, comma);
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
    if (ellipsis_index != std::string_view::npos) {
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
      // Every occurrence of an ellipsis must cover the same number of dimensions.
      // This has to be validated before the loop below records or unifies the
      // ellipsis dimensions, so its indexing remains in bounds.
      if (num_ellipsis == 0) {
        num_ellipsis_indices = rank - term_size;
        first_ellipsis_operand = num_operands;
      } else if (num_ellipsis_indices != rank - term_size) {
        fail_shape_inference(
            "Ellipsis for input ",
            num_operands,
            " represents ",
            rank - term_size,
            " dimensions, but ellipsis for input ",
            first_ellipsis_operand,
            " represents ",
            num_ellipsis_indices,
            " dimensions.");
      }
      num_ellipsis++;
    } else {
      // For non-ellipsis case, rank must equal term_size
      if (rank != term_size) {
        fail_shape_inference(
            "Rank of input ", num_operands, " (", rank, ") does not match the equation indices (", term_size, ").");
      }
    }

    for (size_t index = 0; index < term.size();) {
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
        index += 3; // skip the ellipsis
        num_illegal_char += 3;
        continue;

      } else if (!is_letter(term[index])) {
        fail_shape_inference(
            "Einsum equation has an unsupported character '",
            term[index],
            "' in the term for input ",
            num_operands,
            ". Only letters and one ellipsis are allowed.");
      }

      const auto inserted = label_maps.emplace(term[index], num_labels).second;
      if (inserted) {
        *dims_value.add_dim() = shape.dim(static_cast<int>(index + ellipsis_dims - num_illegal_char));
        ++num_labels;
      } else {
        repeated_labels.insert(term[index]);
      }
      ++index;
    }

    num_operands++;
    if (comma == std::string_view::npos)
      break;
    // Keep empty terms, including a trailing scalar operand.
    left_equation.remove_prefix(comma + 1);
  }

  if (num_inputs != num_operands) {
    fail_shape_inference("Number of input tensors does not match the operands in the equation.");
  }

  // Parse the provided right-hand side
  if (mid_index != std::string_view::npos) {
    const auto right_equation = equation.substr(mid_index + 2);
    auto right_ellipsis_index = right_equation.find("...");

    for (size_t index = 0; index < right_equation.size();) {
      // If there's an ellipsis, add its corresponding dimensions
      if (index == right_ellipsis_index) {
        for (size_t i = 0; i < num_ellipsis_indices; i++) {
          *output_shape.add_dim() = ellipsis_dims_value.dim(static_cast<int>(i));
        }
        index += 3; // skip the ellipsis
        continue;
      }

      if (is_letter(right_equation[index])) {
        auto it = label_maps.find(right_equation[index]);
        if (it == label_maps.end()) {
          fail_shape_inference("Equation output contains a label missing from the inputs");
        }
        *output_shape.add_dim() = dims_value.dim(it->second);
      }
      ++index;
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

void EinsumShapeInference(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  std::string equation = getAttribute(ctx, "equation", "");
  if (equation.empty()) {
    return;
  }

  equation.erase(std::remove(equation.begin(), equation.end(), ' '),
                 equation.end()); // Remove space char
  einsumShapeInference(ctx, equation);
}

std::function<void(OpSchema&)> EinsumOpGenerator() {
  return [](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(EinsumShapeInference); };
}

static void MathOpDataPropagator(DataPropagationContext& ctx, std::string_view op_type) {
  const auto* const input_0 = ctx.getInputData(0);
  const auto* const input_1 = ctx.getInputData(1);
  if (input_0 == nullptr || input_1 == nullptr) {
    return;
  }
  int size_0 = input_0->dim_size();
  int size_1 = input_1->dim_size();
  // Fails to broadcast if the ranks are different and no any rank is 1
  if (size_0 != size_1 && size_0 != 1 && size_1 != 1) {
    fail_shape_inference("Invalid rank for ", op_type, " broadcasting: (", size_0, ") vs (", size_1, ").");
  }
  TensorShapeProto tsp;
  int size_out = size_0 == 1 ? size_1 : size_0;
  for (int i = 0; i < size_out; ++i) {
    const auto& input_dim_0 = input_0->dim(size_0 == 1 ? 0 : i);
    const auto& input_dim_1 = input_1->dim(size_1 == 1 ? 0 : i);
    if (input_dim_0.has_dim_value() && input_dim_1.has_dim_value()) {
      tsp.mutable_dim()->Add()->set_dim_value(
          defs::math::utils::MathOpTwoIntegers(op_type, input_dim_0.dim_value(), input_dim_1.dim_value()));
    } else {
      // Cannot compute the value; simply add an empty dim without value and param
      tsp.mutable_dim()->Add();
    }
  }
  ctx.addOutputData(0, std::move(tsp));
}

void MathShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void AddDataPropagation_opset14(DataPropagationContext& ctx) {
  MathOpDataPropagator(ctx, "Add");
}

void SubDataPropagation_opset14(DataPropagationContext& ctx) {
  MathOpDataPropagator(ctx, "Sub");
}

void ModShapeInference_opset28(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void MulDataPropagation_opset14(DataPropagationContext& ctx) {
  MathOpDataPropagator(ctx, "Mul");
}

static constexpr const char* gelu_default_approx = "none";

bool BuildContextDependentFunctionBodyGelu(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  const auto* const approx_attr_proto = ctx.getAttribute("approximate");
  std::string approximate =
      approx_attr_proto != nullptr && approx_attr_proto->has_s() ? approx_attr_proto->s() : gelu_default_approx;
  FunctionBuilder builder(functionProto);

  if (approximate == "tanh") {
    builder.Add(R"(
              Half = Constant <value = float {0.5}>()
              HalfCast = CastLike (Half, X)
              One = Constant <value = float {1.0}>()
              OneCast = CastLike (One, X)
              TwoOverPi = Constant <value = float {0.63661977236}>()
              TwoOverPiCast = CastLike (TwoOverPi, X)
              C0 = Constant <value = float {0.044715}>()
              C0Cast = CastLike (C0, X)
              SqrtTwoOverPi = Sqrt (TwoOverPiCast)
              Three = Constant <value = float {3.0}>()
              ThreeCast = CastLike (Three, X)
              XCubed = Pow (X, ThreeCast)
              XCubedC0 = Mul (C0Cast, XCubed)
              XC0XCubed = Sum (X, XCubedC0)
              TanhInput = Mul (SqrtTwoOverPi, XC0XCubed)
              ErfApprox = Tanh (TanhInput)
              PhiApprox = Sum (OneCast, ErfApprox)
              MultX = Mul (HalfCast, X)
              Y = Mul (MultX, PhiApprox)
              )");
  } else {
    builder.Add(R"(
              Half = Constant <value = float {0.5}>()
              HalfCast = CastLike (Half, X)
              One = Constant <value = float {1.0}>()
              OneCast = CastLike (One, X)
              Two = Constant <value = float {2.0}>()
              TwoCast = CastLike (Two, X)
              SqrtTwo = Sqrt (TwoCast)
              XSqrt = Div (X, SqrtTwo)
              ErfXSqrt = Erf(XSqrt)
              Phi = Sum (OneCast, ErfXSqrt)
              MultX = Mul (HalfCast, X)
              Y = Mul (MultX, Phi)
              )");
  }
  schema.BuildFunction(functionProto);
  return true;
}

void SwiGLUShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }
  // A and B must have identical shapes: no broadcasting, no rank expansion, no
  // size-1 stretching. Seed the output with A's shape and merge B into it, which
  // fails on any rank mismatch or conflicting static dimension while merging
  // symbolic/unknown dimensions toward the more-specified one.
  auto* output_type = ctx.getOutputType(0)->mutable_tensor_type();
  *output_type->mutable_shape() = ctx.getInputType(0)->tensor_type().shape();
  mergeInShapeInfo(ctx.getInputType(1)->tensor_type(), *output_type);
}

void PowShapeInference_opset15(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void ElementwiseMultiOpShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  int num_inputs = static_cast<int>(ctx.getNumInputs());
  std::vector<const TensorShapeProto*> shapes;
  shapes.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    const auto* const input_type = ctx.getInputType(i);
    if (nullptr == input_type || !input_type->has_tensor_type() || !input_type->tensor_type().has_shape()) {
      return;
    }
    shapes.push_back(&input_type->tensor_type().shape());
  }

  multidirectionalBroadcastShapeInference(shapes, *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

bool BuildContextDependentFunctionBodyClip(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  bool has_min = ctx.hasInput(1);
  bool has_max = ctx.hasInput(2);

  FunctionBuilder builder(functionProto);
  if (!has_min && !has_max) {
    builder.Add("output = Identity (input)");
  } else if (has_min && !has_max) {
    builder.Add("input_less_than_min = Less (input, min)");
    builder.Add("output = Where (input_less_than_min, min, input)");
  } else if (!has_min && has_max) {
    builder.Add("input_large_than_max = Less (max, input)");
    builder.Add("output = Where (input_large_than_max, max, input)");
  } else {
    builder.Add("input_less_than_min = Less (input, min)");
    builder.Add("tmp = Where (input_less_than_min, min, input)");
    builder.Add("output_large_than_max = Less (max, tmp)");
    builder.Add("output = Where (output_large_than_max, max, tmp)");
  }

  schema.BuildFunction(functionProto);
  return true;
}

static void SoftmaxFamilyShapeInference(InferenceContext& ctx, const char* name) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference starts
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  // Validate the value of 'axis'
  const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  int r = input_shape.dim_size();
  if (r == 0) {
    fail_shape_inference("Input rank must be >= 1 for ", name, ".");
  }
  int axis = static_cast<int>(getAttribute(ctx, "axis", -1));
  if (axis < -r || axis >= r) {
    fail_shape_inference("'axis' must be in [", -r, " , ", (r - 1), "]. Its actual value is: ", axis);
  }

  // Shape inference
  propagateShapeFromInputToOutput(ctx, 0, 0);
}

bool SoftmaxFunctionBody_opset18(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  int64_t axis = ctx.getAttribute("axis") != nullptr ? ctx.getAttribute("axis")->i() : -1;
  FunctionBuilder builder(functionProto);
  builder.Const1D("axes", axis).Add("X_ReduceMax = ReduceMax <keepdims = 1> (input, axes)").Add(R"(
                    X_Sub = Sub (input, X_ReduceMax)
                    X_Exp = Exp (X_Sub)
                    X_ReduceSum = ReduceSum <keepdims = 1> (X_Exp, axes)
                    output = Div (X_Exp, X_ReduceSum)
                )");

  schema.BuildFunction(functionProto);
  return true;
}

bool SoftmaxFunctionBody_opset13(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  int64_t axis = ctx.getAttribute("axis") != nullptr ? ctx.getAttribute("axis")->i() : -1;
  FunctionBuilder builder(functionProto);
  builder.Const1D("axes", axis)
      .Add("X_ReduceMax = ReduceMax <keepdims = 1> (input)", "axes", std::vector<int64_t>({axis}))
      .Add(R"(
                    X_Sub = Sub (input, X_ReduceMax)
                    X_Exp = Exp (X_Sub)
                    X_ReduceSum = ReduceSum <keepdims = 1> (X_Exp, axes)
                    output = Div (X_Exp, X_ReduceSum)
                )");

  schema.BuildFunction(functionProto);
  return true;
}

void SoftmaxShapeInference_opset13(InferenceContext& ctx) {
  SoftmaxFamilyShapeInference(ctx, "Softmax");
}

bool LogSoftmaxFunctionBody_opset18(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  const int64_t axis = ctx.getAttribute("axis") != nullptr ? ctx.getAttribute("axis")->i() : -1;
  FunctionBuilder builder(functionProto);
  builder.Const1D("axes", axis).Add("X_ReduceMax = ReduceMax <keepdims = 1> (input, axes)").Add(R"(
                    X_Sub = Sub (input, X_ReduceMax)
                    X_Exp = Exp (X_Sub)
                    X_ReduceSum = ReduceSum <keepdims = 1> (X_Exp, axes)
                    X_Log = Log (X_ReduceSum)
                    output = Sub (X_Sub, X_Log)
                )");

  schema.BuildFunction(functionProto);
  return true;
}

bool LogSoftmaxFunctionBody_opset13(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  const int64_t axis = ctx.getAttribute("axis") != nullptr ? ctx.getAttribute("axis")->i() : -1;
  FunctionBuilder builder(functionProto);
  builder.Const1D("axes", axis)
      .Add("X_ReduceMax = ReduceMax <keepdims = 1> (input)", "axes", std::vector<int64_t>({axis}))
      .Add(R"(
                    X_Sub = Sub (input, X_ReduceMax)
                    X_Exp = Exp (X_Sub)
                    X_ReduceSum = ReduceSum <keepdims = 1> (X_Exp, axes)
                    X_Log = Log (X_ReduceSum)
                    output = Sub (X_Sub, X_Log)
                )");

  schema.BuildFunction(functionProto);
  return true;
}

void LogSoftmaxShapeInference_opset13(InferenceContext& ctx) {
  SoftmaxFamilyShapeInference(ctx, "LogSoftmax");
}

void HardmaxShapeInference_opset13(InferenceContext& ctx) {
  SoftmaxFamilyShapeInference(ctx, "Hardmax");
}

void GemmShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  const bool transA = getAttribute(ctx, "transA", int64_t{0}) != 0;
  const bool transB = getAttribute(ctx, "transB", int64_t{0}) != 0;
  if (hasInputShape(ctx, 1) && getInputShape(ctx, 1).dim_size() != 2) {
    const auto& b_shape = getInputShape(ctx, 1);
    fail_shape_inference(
        "Gemm input B must have rank 2, but has rank ", b_shape.dim_size(), " and shape ", b_shape, ".");
  }
  Dim K, N;
  if (transB) {
    ctx.unifyInputShape(1, {N, K});
  } else {
    ctx.unifyInputShape(1, {K, N});
  }
  if (!hasInputShape(ctx, 0)) {
    return;
  }
  auto shape = getInputShape(ctx, 0);
  const int rank = shape.dim_size();
  if (rank == 0) {
    fail_shape_inference("Gemm input A must have rank at least 1, but has rank 0 and shape ", shape, ".");
  }
  if (transA && rank > 1) {
    shape.mutable_dim()->SwapElements(rank - 2, rank - 1);
  }
  const auto& a_k = shape.dim(rank - 1);
  if (a_k.has_dim_value() && K.has_dim_value() && a_k.dim_value() != K.dim_value()) {
    fail_shape_inference(
        "Gemm inputs have incompatible contraction dimensions: A has shape ",
        getInputShape(ctx, 0),
        " with transA=",
        transA,
        ", B has shape ",
        getInputShape(ctx, 1),
        " with transB=",
        transB,
        ". After transposition, A's last dimension is ",
        a_k.dim_value(),
        " but B's first dimension is ",
        K.dim_value(),
        "; these dimensions must match.");
  }
  *shape.mutable_dim(rank - 1) = N;
  if (hasInputShape(ctx, 2)) {
    const auto& bias = getInputShape(ctx, 2);
    if (bias.dim_size() > rank) {
      fail_shape_inference(
          "Gemm input C with shape ",
          bias,
          " must be unidirectionally broadcastable to output shape ",
          shape,
          ", but C has rank ",
          bias.dim_size(),
          " greater than output rank ",
          rank,
          ".");
    }
    for (int i = 0; i < bias.dim_size(); ++i) {
      const auto& c_dim = bias.dim(i);
      const auto& y_dim = shape.dim(rank - bias.dim_size() + i);
      if (c_dim.has_dim_value() && y_dim.has_dim_value() && c_dim.dim_value() != 1 &&
          c_dim.dim_value() != y_dim.dim_value()) {
        fail_shape_inference(
            "Gemm input C with shape ",
            bias,
            " must be unidirectionally broadcastable to output shape ",
            shape,
            ", but C dimension ",
            i,
            " has size ",
            c_dim.dim_value(),
            " instead of 1 or ",
            y_dim.dim_value(),
            " (output dimension ",
            rank - bias.dim_size() + i,
            ").");
      }
    }
  }
  updateOutputShape(ctx, 0, shape);
}

bool BuildContextDependentFunctionBodyGemm(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  const auto* a_type = ctx.getInputType(0);
  if (a_type == nullptr) {
    return false;
  }
  const auto elem_type = a_type->tensor_type().elem_type();
  if (elem_type != TensorProto::FLOAT && elem_type != TensorProto::DOUBLE && elem_type != TensorProto::FLOAT16 &&
      elem_type != TensorProto::BFLOAT16) {
    return false;
  }
  const auto* transA = ctx.getAttribute("transA");
  const auto* transB = ctx.getAttribute("transB");
  FunctionBuilder builder(functionProto);
  if (transA != nullptr && transA->i() != 0) {
    if (!a_type->tensor_type().has_shape()) {
      return false;
    }
    const int rank = a_type->tensor_type().shape().dim_size();
    std::vector<int64_t> perm;
    perm.reserve(rank);
    for (int i = 0; i < rank; ++i) {
      perm.push_back(i);
    }
    if (rank > 1) {
      std::swap(perm[rank - 2], perm[rank - 1]);
    }
    builder.Add("AT = Transpose (A)", "perm", perm);
  } else {
    builder.Add("AT = Identity (A)");
  }
  builder.Add(transB != nullptr && transB->i() != 0 ? "BT = Transpose <perm = [1, 0]> (B)" : "BT = Identity (B)");
  builder.Add(R"ONNX(
    Product = MatMul (AT, BT)
    Alpha = Constant <value_float: float = @alpha> ()
    AlphaCast = CastLike (Alpha, A)
    ScaledProduct = Mul (Product, AlphaCast)
  )ONNX");
  const auto* beta = ctx.getAttribute("beta");
  if (ctx.hasInput(2) && (beta == nullptr || beta->f() != 0.0f)) {
    builder.Add(R"ONNX(
      Beta = Constant <value_float: float = @beta> ()
      BetaCast = CastLike (Beta, A)
      ScaledBias = Mul (C, BetaCast)
      Y = Add (ScaledProduct, ScaledBias)
    )ONNX");
  } else {
    builder.Add("Y = Identity (ScaledProduct)");
  }
  schema.BuildFunction(functionProto);
  return true;
}

void MatMulShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  defs::math::utils::MatMulShapeInference(ctx, 0, 1);
}

void GroupedMatMulShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  Dim M, K, G, N, S;
  ctx.unifyInputShape(0, {M, K});
  ctx.unifyInputShape(1, {G, K, N});
  ctx.unifyInputShape(2, {M, S});
  if (ctx.hasInput(3)) {
    ctx.unifyInputShape(3, {G, N});
  }
  updateOutputShape(ctx, 0, {M, S, N});
}

bool BuildContextDependentFunctionBodyGroupedMatMul(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  FunctionBuilder builder(functionProto);
  builder.Const1D("index_zero", int64_t{0})
      .Const1D("index_one", int64_t{1})
      .Const1D("index_two", int64_t{2})
      .Add("input_shape = Shape (input)")
      .Add("weights_shape = Shape (weights)")
      .Add("indices_shape = Shape (group_indices)")
      .Add("M = Gather <axis = 0> (input_shape, index_zero)")
      .Add("K = Gather <axis = 0> (input_shape, index_one)")
      .Add("N = Gather <axis = 0> (weights_shape, index_two)")
      .Add("S = Gather <axis = 0> (indices_shape, index_one)")
      .Add("MS = Mul (M, S)")
      .Add("indices_flat = Reshape <allowzero = 1> (group_indices, MS)")
      .Add("selected_weights = Gather <axis = 0> (weights, indices_flat)")
      .Add("input_unsqueezed = Unsqueeze (input, index_one)")
      .Add("expanded_shape = Concat <axis = 0> (M, S, K)")
      .Add("input_expanded = Expand (input_unsqueezed, expanded_shape)")
      .Add("matmul_input_shape = Concat <axis = 0> (MS, index_one, K)")
      .Add("matmul_input = Reshape <allowzero = 1> (input_expanded, matmul_input_shape)")
      .Add("matmul_result = MatMul (matmul_input, selected_weights)")
      .Add("output_shape = Concat <axis = 0> (M, S, N)")
      .Add("result = Reshape <allowzero = 1> (matmul_result, output_shape)");

  if (ctx.hasInput(3)) {
    builder.Add("selected_bias = Gather <axis = 0> (bias, indices_flat)")
        .Add("reshaped_bias = Reshape <allowzero = 1> (selected_bias, output_shape)")
        .Add("output = Add (result, reshaped_bias)");
  } else {
    builder.Add("output = Identity (result)");
  }

  schema.BuildFunction(functionProto);
  return true;
}

void ExpandShapeInference_opset13(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference
  // For shape inference, we need both input shape
  if (hasNInputShapes(ctx, 2)) {
    const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    bool found = false;
    TensorShapeProto second_shape = getShapeInput(ctx, 1, found);
    if (found) {
      bidirectionalBroadcastShapeInference(input_shape, second_shape, *getOutputShape(ctx, 0));
    }
  }
}

void MatMulIntegerShapeInference_opset10(ONNX_NAMESPACE::InferenceContext& ctx) {
  auto a_type = ctx.getInputType(0);
  auto b_type = ctx.getInputType(1);
  auto y_type = ctx.getOutputType(0);
  if (nullptr == a_type || nullptr == b_type || nullptr == y_type ||
      a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
      b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
    fail_type_inference("inputs are expected to have tensor type and output type should not be null.");
  }

  // Right now we only support int32
  y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::INT32);

  defs::math::utils::MatMulShapeInference(ctx, 0, 1);
}

void DetShapeInference_opset22(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference: input [..., M, M] -> output [...]
  if (hasInputShape(ctx, 0)) {
    const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    const int rank = static_cast<int>(input_shape.dim_size());

    if (rank < 2) {
      fail_shape_inference("Input rank must be >= 2.");
    }

    // Validate the inner-most 2 dims are square
    Dim M;
    unifyDim(input_shape.dim(rank - 2), M);
    unifyDim(input_shape.dim(rank - 1), M);

    // Output is the batch prefix (all dims except the last 2)
    TensorShapeProto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
    for (int i = 0; i < rank - 2; ++i) {
      *output_shape->add_dim() = input_shape.dim(i);
    }
  }
}

bool BuildContextDependentFunctionBody(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  if (ctx.getInputType(0) == nullptr) {
    // we cannot create a correct function body without knowing the input type
    return false;
  }
  auto input_type = ctx.getInputType(0)->tensor_type().elem_type();
  bool float_input = input_type == TensorProto_DataType_FLOAT;
  const auto* const reduction_attr_proto = ctx.getAttribute("reduction");
  std::string reduction_attr =
      reduction_attr_proto != nullptr && reduction_attr_proto->has_s() ? reduction_attr_proto->s() : "mean";

  FunctionBuilder builder(functionProto);
  builder.Const1D("const_zero", static_cast<int64_t>(0))
      .Const1D("const_one", static_cast<int64_t>(1))
      .Const1D("axes", static_cast<int64_t>(1))
      .Add("expanded_target = Unsqueeze (target, axes)");

  if (ctx.getAttribute("ignore_index") == nullptr) {
    builder.Add(R"(
      input_gather_element = GatherElements <axis = 1> (input, expanded_target)
      loss_NCdd = Neg (input_gather_element)
      loss_N1dd = Slice (loss_NCdd, const_zero, const_one, const_one)
    )");

    if (!ctx.hasInput(2)) {
      if (reduction_attr == "none") {
        builder.Add("loss = Squeeze (loss_N1dd, axes)");
      } else {
        builder.Add("loss_Ndd = Squeeze (loss_N1dd, axes)");
        if (reduction_attr == "mean") {
          builder.Add("loss = ReduceMean <keepdims = 0> (loss_Ndd)");
        } else {
          builder.Add("loss = ReduceSum <keepdims = 0> (loss_Ndd)");
        }
      }
    } else {
      builder.Add("weight_gather = Gather (weight, target)");
      builder.Add("loss_unweighted = Squeeze (loss_N1dd, axes)");
      if (reduction_attr == "none") {
        builder.Add("loss = Mul (loss_unweighted, weight_gather)");
      } else {
        builder.Add("loss_Ndd = Mul (loss_unweighted, weight_gather)");
        if (reduction_attr == "mean") {
          builder.Add(R"(
            loss_sum = ReduceSum <keepdims = 0> (loss_Ndd)
            weight_gather_sum = ReduceSum <keepdims = 0> (weight_gather)
            loss = Div (loss_sum, weight_gather_sum)
          )");
        } else {
          builder.Add("loss = ReduceSum <keepdims = 0> (loss_Ndd)");
        }
      }
    }
  } else {
    builder.Const1D("const_ignore_index", ctx.getAttribute("ignore_index")->i());
    builder.Add(R"(
      const_zero_target_typed = Sub (expanded_target, expanded_target)
      expanded_target_int64 = Cast <to = 7> (expanded_target)
      mask = Equal (expanded_target_int64, const_ignore_index)
      transform_targets = Where (mask, const_zero_target_typed, expanded_target)
    )");
    builder.Add("input_gather_element = GatherElements <axis = 1> (input, transform_targets)");
    builder.Const1D("const_zero_float", 0.0f);
    if (!float_input) {
      builder.Add("const_zero_casted = Cast (const_zero_float)", "to", static_cast<int64_t>(input_type))
          .Add("input_gather_element_transform = Where (mask, const_zero_casted, input_gather_element)");
    } else {
      builder.Add("input_gather_element_transform = Where (mask, const_zero_float, input_gather_element)");
    }
    builder.Add("loss_NCdd = Neg (input_gather_element_transform)");
    builder.Add("loss_N1dd = Slice (loss_NCdd, const_zero, const_one, const_one)");

    if (!ctx.hasInput(2)) {
      builder.Add("squeeze_mask = Squeeze (mask, axes)");
      builder.Const1D("const_one_float", 1.0f);
      if (!float_input) {
        builder.Add("const_one_casted = Cast (const_one_float)", "to", static_cast<int64_t>(input_type))
            .Add("weight_gather = Where (squeeze_mask, const_zero_casted, const_one_casted)");
      } else {
        builder.Add("weight_gather = Where (squeeze_mask, const_zero_float, const_one_float)");
      }

    } else {
      builder.Add("weight_gather_temp = Gather (weight, transform_targets)");
      builder.Add(
          float_input ? "weight_gather_temp_1 = Where (mask, const_zero_float, weight_gather_temp)"
                      : "weight_gather_temp_1 = Where (mask, const_zero_casted, weight_gather_temp)");
      builder.Add("weight_gather = Squeeze (weight_gather_temp_1, axes)");
    }

    builder.Add("loss_unweighted = Squeeze (loss_N1dd, axes)");
    if (reduction_attr == "none") {
      builder.Add("loss = Mul (loss_unweighted, weight_gather)");
    } else {
      builder.Add("loss_Ndd = Mul (loss_unweighted, weight_gather)");
      if (reduction_attr == "mean") {
        builder.Add(R"(
            loss_sum = ReduceSum <keepdims = 0> (loss_Ndd)
            weight_gather_sum = ReduceSum <keepdims = 0> (weight_gather)
            loss = Div (loss_sum, weight_gather_sum)
        )");
      } else {
        builder.Add("loss = ReduceSum <keepdims = 0> (loss_Ndd)");
      }
    }
  }

  schema.BuildFunction(functionProto);
  return true;
}

void NegativeLogLikelihoodLossShapeInference_opset22(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference
  // input: [N, C, d1, ..., dk], target: [N, d1, ..., dk], weight (optional): [C]
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();

  const int input_rank = static_cast<int>(input_shape.dim_size());
  const int target_rank = input_rank - 1;

  if (input_rank < 2) {
    fail_shape_inference("Input rank must be >= 2. input_rank=", input_rank);
  }
  checkInputRank(ctx, 1, target_rank);

  // input: [N, C, d1, ..., dk], target: [N, d1, ..., dk]
  // Build dims array for target shape: [N, d1, ..., dk]
  Dim C;
  std::vector<Dim> target_dims(target_rank);
  // Unify target dims against both inputs
  for (int i = 0; i < target_rank; i++) {
    unifyInputDim(ctx, 1, i, target_dims[i]);
    // input[0]=N matches target[0], input[2..]=d1..dk matches target[1..]
    int input_i = (i == 0) ? 0 : i + 1;
    unifyInputDim(ctx, 0, input_i, target_dims[i]);
  }
  unifyInputDim(ctx, 0, 1, C);

  // Optional weight input: shape [C]
  if (ctx.getNumInputs() == 3 && hasInputShape(ctx, 2)) {
    ctx.unifyInputShape(2, {C});
  }

  if (getAttribute(ctx, "reduction", "mean") == "none") {
    // output tensor is of shape (N, d1, d2, ..., dk)
    TensorShapeProto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
    for (int i = 0; i < target_rank; i++) {
      *output_shape->add_dim() = target_dims[i];
    }
  } else {
    // output is a scalar (empty shape)
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  }
}

bool BuildContextDependentFunctionBodySCE(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  FunctionBuilder builder(functionProto);
  // Using stable implementation of LogSoftmax
  builder //
      .Const("Shape3D", std::vector<int64_t>({0, 0, -1})) //
      .Add(R"(
        X_NCD = Reshape (scores, Shape3D)
        X_NDC = Transpose <perm = [0, 2, 1]> (X_NCD)
        X_LogSM = LogSoftmax <axis = 2> (X_NDC)
        X_LogSM_NCD = Transpose <perm = [0, 2, 1]> (X_LogSM)
        X_shape = Shape (scores)
        X_Log = Reshape (X_LogSM_NCD, X_shape)
      )");

  // Review(mzs): Ideally we want to reuse the output from Log for sub-graph
  // output as well but looking at the graph resolve code it does not include
  // graph outputs as intermediate outputs, hence if intermediate X_log is
  // renamed as log_prob then it will be treated as graph output and will not be
  // available to NegativeLogLikelihoodLoss. May be my understanding is
  // incorrect or there is a bug in function population code in ORTbut I will
  // dig further to be 100%. In the meantime we just replicate the log.
  if (ctx.hasOutput(1)) {
    builder.Add("log_prob = Identity (X_Log)");
  }

  builder.Add(
      ctx.hasInput(2)
          ? "output = NegativeLogLikelihoodLoss <reduction : string = @reduction, ignore_index : int = @ignore_index> (X_Log, labels, weights)"
          : "output = NegativeLogLikelihoodLoss <reduction : string = @reduction, ignore_index : int = @ignore_index> (X_Log, labels)");

  schema.BuildFunction(functionProto);
  return true;
}

void SoftmaxCrossEntropyLossShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  std::string reduction = getAttribute(ctx, "reduction", "mean");
  if (reduction == "none") {
    if (hasInputShape(ctx, 1)) {
      propagateShapeFromInputToOutput(ctx, 1, 0);
    }
  } else {
    updateOutputShape(ctx, 0, TensorShapeProto());
  }

  if (ctx.getNumOutputs() == 2) {
    propagateElemTypeFromInputToOutput(ctx, 0, 1);
    propagateShapeFromInputToOutput(ctx, 0, 1);
  }
}

void DFTShapeInference_opset20(ONNX_NAMESPACE::InferenceContext& ctx) {
  bool is_onesided = static_cast<bool>(getAttribute(ctx, "onesided", 0));
  bool inverse = static_cast<bool>(getAttribute(ctx, "inverse", 0));

  const size_t input_arg_index = 0;
  const size_t dft_length_arg_index = 1;
  const size_t axis_arg_index = 2;
  const size_t output_index = 0;

  propagateElemTypeFromInputToOutput(ctx, input_arg_index, output_index);
  if (!hasInputShape(ctx, input_arg_index)) {
    // If no shape is available for the input, skip shape inference...
    return;
  }

  auto& input_shape = getInputShape(ctx, input_arg_index);

  // The last dimension is the real and imaginary parts of the value.
  const int64_t rank = input_shape.dim_size();
  if (rank < 2) {
    fail_shape_inference("input tensor must have rank >= 2, including the complex dimension.");
  }

  // check the inputs are correct types for one-sided DFT
  if (is_onesided) {
    auto last_dim = input_shape.dim(rank - 1);
    if (inverse) {
      // Check last dimension is 2 (complex input required)
      if (last_dim.has_dim_value() && last_dim.dim_value() != 2) {
        fail_shape_inference("inverse one-sided DFT requires complex input (last dimension must be 2)");
      }
    } else {
      // Check last dimension is 1 (real input required)
      if (last_dim.has_dim_value() && last_dim.dim_value() != 1) {
        fail_shape_inference("one-sided DFT requires real input (last dimension must be 1)");
      }
    }
  }

  // In general the output shape will match the input shape exactly
  // So initialize the output shape with the input shape
  TensorShapeProto result_shape_proto = input_shape;

  // Handle when axis is an input but is not statically known
  if (ctx.hasInput(axis_arg_index) && ctx.getInputData(axis_arg_index) == nullptr) {
    // Axis is an input but is not statically known
    if (is_onesided || ctx.hasInput(dft_length_arg_index)) {
      // We don't know which axis is the signal dimension, so we cannot infer shape
      // when onesided is enabled or when dft_length_arg_index is provided
      TensorShapeProto new_shape_proto{};
      for (int i = 0; i < rank; ++i) {
        new_shape_proto.add_dim();
      }
      // Set last dimension based on operation type
      ONNX_ASSERTM(
          rank == static_cast<int64_t>(new_shape_proto.dim_size()),
          "rank should be equal to new_shape_proto.dim_size()")
      if (inverse && is_onesided) {
        // IRFFT: output is real-valued
        new_shape_proto.mutable_dim(rank - 1)->set_dim_value(1);
      } else {
        // Complex output
        new_shape_proto.mutable_dim(rank - 1)->set_dim_value(2);
      }
      updateOutputShape(ctx, output_index, new_shape_proto);
      return;
    } else {
      // Set last dimension based on operation type
      int dim_size = result_shape_proto.dim_size();
      if (inverse && is_onesided) {
        // IRFFT: output is real-valued
        result_shape_proto.mutable_dim(dim_size - 1)->set_dim_value(1);
      } else {
        // Complex output
        result_shape_proto.mutable_dim(dim_size - 1)->set_dim_value(2);
      }
      updateOutputShape(ctx, output_index, result_shape_proto);
      return;
    }
  }

  // Get the axis where the DFT will be performed
  int64_t axis;
  if (!ctx.hasInput(axis_arg_index)) {
    // axis is not an input. We use -2 by default
    axis = -2;
  } else {
    const TensorProto* axis_tensor = ctx.getInputData(axis_arg_index);
    ONNX_ASSERTM(axis_tensor != nullptr, "axis should not be nullptr at this point")
    // TODO(justinchuby): Create invariance checking functions to ensure shapes and sizes
    // to abstract the following logic out.
    if (axis_tensor->dims_size() != 0) {
      fail_shape_inference("axis input must be a scalar.");
    }
    axis = defs::math::utils::GetScalarValueFromTensor<int64_t>(axis_tensor);
  }

  // NOLINTNEXTLINE(readability-simplify-boolean-expr)
  if (!(-rank <= axis && axis != -1 && axis < rank - 1)) {
    fail_shape_inference(
        "axis attribute value ",
        axis,
        " is invalid for a tensor of rank ",
        rank,
        ". Valid values are '-rank <= axis && axis != -1 && axis < rank - 1'");
  }

  auto axis_idx = (axis >= 0 ? axis : axis + rank);

  // If dft_length is specified, then we should honor the shape.
  // Set the output dimension to match the dft_length on the axis.
  if (ctx.hasInput(dft_length_arg_index)) {
    // dft_length is provided
    const TensorProto* dft_length = ctx.getInputData(dft_length_arg_index);
    if (dft_length == nullptr) {
      // If we cannot read the dft_length, we cannot infer shape on the signal axis
      result_shape_proto.mutable_dim(axis_idx)->clear_dim_value();
    } else {
      if (dft_length->dims_size() != 0) {
        fail_shape_inference("dft_length input must be a scalar.");
      }
      auto dft_length_value = defs::math::utils::GetScalarValueFromTensor<int64_t>(dft_length);

      // For RFFT, output size on signal axis is floor(dft_length/2) + 1
      if (is_onesided && !inverse) {
        // RFFT: one-sided output
        auto half_signal_size = (dft_length_value / 2) + 1;
        result_shape_proto.mutable_dim(axis_idx)->set_dim_value(half_signal_size);
      } else {
        // Standard FFT/IFFT and IRFFT: full length
        result_shape_proto.mutable_dim(axis_idx)->set_dim_value(dft_length_value);
      }
    }
  } else if (is_onesided) {
    auto axis_dimension = result_shape_proto.dim(axis_idx);
    if (axis_dimension.has_dim_value()) {
      auto axis_dimension_value = axis_dimension.dim_value();
      if (inverse) {
        // IRFFT without explicit dft_length: cannot reliably infer full signal length
        // Default to even length: N = 2 * (input_size - 1)
        auto full_signal_size = checkedMultiply(2, checkedSubtract(axis_dimension_value, 1));
        result_shape_proto.mutable_dim(axis_idx)->set_dim_value(full_signal_size);
      } else {
        // RFFT without explicit dft_length: infer one-sided output size from input
        auto half_signal_size = (axis_dimension_value / 2) + 1;
        result_shape_proto.mutable_dim(axis_idx)->set_dim_value(half_signal_size);
      }
    } else {
      result_shape_proto.mutable_dim(axis_idx)->clear_dim_value();
      result_shape_proto.mutable_dim(axis_idx)->clear_dim_param();
    }
  }

  // Set the last dimension based on whether output is real or complex
  if (is_onesided && inverse) {
    // IRFFT: complex input -> real output (last dim = 1)
    result_shape_proto.mutable_dim(static_cast<int>(rank - 1))->set_dim_value(1);
  } else {
    // All other cases: complex output (last dim = 2)
    result_shape_proto.mutable_dim(static_cast<int>(rank - 1))->set_dim_value(2);
  }

  updateOutputShape(ctx, output_index, result_shape_proto);
}

void CosineSumWindowOpShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  // Update the output data type to the output_datatype
  auto output_datatype = getAttribute(ctx, "output_datatype", static_cast<int64_t>(TensorProto_DataType_FLOAT));
  updateOutputElemType(ctx, 0, narrow<int32_t>(output_datatype));

  if (!hasInputShape(ctx, 0)) {
    // If no shape is available for the input, skip shape inference.
    return;
  }

  const auto* const size = ctx.getInputData(0);
  if (size == nullptr) {
    // Size is not available, so return early
    return;
  }

  if (size->dims_size() != 0) {
    fail_shape_inference("size input must be a scalar.");
  }

  auto size_value = defs::math::utils::GetScalarValueFromTensor<int64_t>(size);
  if (size_value <= 0) {
    fail_shape_inference("size input must be greater than 0.");
  }

  ONNX_NAMESPACE::TensorShapeProto result_shape;
  result_shape.add_dim()->set_dim_value(size_value);
  updateOutputShape(ctx, 0, result_shape);
}

void MelWeightMatrixShapeInference_opset17(ONNX_NAMESPACE::InferenceContext& ctx) {
  auto output_datatype = getAttribute(ctx, "output_datatype", static_cast<int64_t>(TensorProto_DataType_FLOAT));
  updateOutputElemType(ctx, 0, narrow<int32_t>(output_datatype));

  if (!hasInputShape(ctx, 0) || !hasInputShape(ctx, 1)) {
    return;
  }

  const auto num_mel_bins = ctx.getInputData(0);
  const auto dft_length = ctx.getInputData(1);
  if (nullptr == num_mel_bins || nullptr == dft_length) {
    return;
  }

  int64_t num_mel_bins_value = -1;
  int64_t dft_length_value = -1;
  if (num_mel_bins->dims_size() != 0) {
    fail_shape_inference("num_mel_bins input must be scalar.");
  }
  num_mel_bins_value = defs::math::utils::GetScalarValueFromTensor<int64_t>(num_mel_bins);

  if (dft_length->dims_size() != 0) {
    fail_shape_inference("dft_length input must be scalar.");
  }
  dft_length_value = defs::math::utils::GetScalarValueFromTensor<int64_t>(dft_length);

  if (num_mel_bins_value > 0 && dft_length_value > 0) {
    ONNX_NAMESPACE::TensorShapeProto result_shape;
    result_shape.add_dim()->set_dim_value(static_cast<int64_t>((dft_length_value / 2) + 1));
    result_shape.add_dim()->set_dim_value(num_mel_bins_value);
    updateOutputShape(ctx, 0, result_shape);
  }
}

void STFTShapeInference_opset17(ONNX_NAMESPACE::InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasInputShape(ctx, 0)) {
    return;
  }

  auto& input_shape = getInputShape(ctx, 0);
  if (input_shape.dim_size() != 3) {
    fail_shape_inference("Input 0 (signal) must have rank 3.");
  }

  const auto& complex_dim = input_shape.dim(2);
  if (complex_dim.has_dim_value() && complex_dim.dim_value() != 1 && complex_dim.dim_value() != 2) {
    fail_shape_inference("The last dimension of signal must have size 1 (real) or 2 (complex).");
  }

  const int64_t onesided = getAttribute(ctx, "onesided", 1);
  if (onesided != 0 && onesided != 1) {
    fail_shape_inference("Attribute onesided must be 0 or 1.");
  }
  if (onesided == 1 && complex_dim.has_dim_value() && complex_dim.dim_value() != 1) {
    fail_shape_inference("One-sided STFT requires real input (signal's last dimension must be 1).");
  }

  const auto& signal_dim = input_shape.dim(1);
  const int64_t signal_size = signal_dim.has_dim_value() ? signal_dim.dim_value() : -1;

  // Preserve compatibility with existing models that use a single-element vector.
  if (hasInputShape(ctx, 1)) {
    const auto& frame_step_shape = getInputShape(ctx, 1);
    if (frame_step_shape.dim_size() > 1 ||
        (frame_step_shape.dim_size() == 1 && frame_step_shape.dim(0).has_dim_value() &&
         frame_step_shape.dim(0).dim_value() != 1)) {
      fail_shape_inference("Input 1 (frame_step) must be a scalar or a single-element vector.");
    }
  }

  const auto* frame_step = ctx.getInputData(1);
  int64_t frame_step_value = -1;
  if (frame_step != nullptr) {
    if (frame_step->dims_size() > 1 || (frame_step->dims_size() == 1 && frame_step->dims(0) != 1)) {
      fail_shape_inference("Input 1 (frame_step) must be a scalar or a single-element vector.");
    }
    frame_step_value = defs::math::utils::GetScalarValueFromTensor<int64_t>(frame_step);
    if (frame_step_value <= 0) {
      fail_shape_inference("frame_step must be greater than 0.");
    }
  }

  const bool has_window = ctx.hasInput(2);
  const auto* window_shape = has_window ? getOptionalInputShape(ctx, 2) : nullptr;
  int64_t window_length = -1;
  if (window_shape != nullptr) {
    if (window_shape->dim_size() != 1) {
      fail_shape_inference("Input 2 (window) must have rank 1.");
    }
    if (window_shape->dim(0).has_dim_value()) {
      window_length = window_shape->dim(0).dim_value();
      if (window_length <= 0) {
        fail_shape_inference("window must have a positive length.");
      }
    }
  }

  const bool has_frame_length = ctx.hasInput(3);
  if (has_frame_length && hasInputShape(ctx, 3) && getInputShape(ctx, 3).dim_size() != 0) {
    fail_shape_inference("Input 3 (frame_length) must be a scalar.");
  }

  const auto* frame_length = has_frame_length ? ctx.getInputData(3) : nullptr;
  int64_t frame_length_value = -1;
  if (frame_length != nullptr) {
    if (frame_length->dims_size() != 0) {
      fail_shape_inference("Input 3 (frame_length) must be a scalar.");
    }
    frame_length_value = defs::math::utils::GetScalarValueFromTensor<int64_t>(frame_length);
    if (frame_length_value <= 0) {
      fail_shape_inference("frame_length must be greater than 0.");
    }
    if (window_length >= 0 && window_length != frame_length_value) {
      fail_type_inference(
          "If STFT has both a window input and frame_length specified, the dimension of the window "
          "must match the frame_length specified.");
    }
  }

  const bool frame_length_defaults_to_signal = !has_window && !has_frame_length;
  int64_t dft_size = frame_length_value;
  if (dft_size < 0 && window_length >= 0) {
    dft_size = window_length;
  } else if (dft_size < 0 && frame_length_defaults_to_signal) {
    dft_size = signal_size;
  }

  // The output has the following shape: [batch_size][frames][dft_unique_bins][2]
  ONNX_NAMESPACE::TensorShapeProto result_shape_proto;
  *result_shape_proto.add_dim() = input_shape.dim(0);

  if (frame_length_defaults_to_signal) {
    result_shape_proto.add_dim()->set_dim_value(1);
  } else if (frame_step_value > 0 && signal_size >= dft_size && dft_size > 0) {
    result_shape_proto.add_dim()->set_dim_value(((signal_size - dft_size) / frame_step_value) + 1);
  } else {
    result_shape_proto.add_dim();
  }

  if (dft_size > 0) {
    result_shape_proto.add_dim()->set_dim_value(onesided == 1 ? ((dft_size / 2) + 1) : dft_size);
  } else {
    result_shape_proto.add_dim();
  }
  result_shape_proto.add_dim()->set_dim_value(2);
  updateOutputShape(ctx, 0, result_shape_proto);
}

bool BuildContextDependentFunctionBody_opset13(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  if (ctx.getInputType(0) == nullptr) {
    // we cannot create a correct function body without knowing the input type
    return false;
  }
  auto input_type = ctx.getInputType(0)->tensor_type().elem_type();
  bool float_input = input_type == TensorProto_DataType_FLOAT;
  const auto* const reduction_attr_proto = ctx.getAttribute("reduction");
  std::string reduction_attr =
      reduction_attr_proto != nullptr && reduction_attr_proto->has_s() ? reduction_attr_proto->s() : "mean";

  FunctionBuilder builder(functionProto);
  builder.Const1D("const_zero", static_cast<int64_t>(0))
      .Const1D("const_one", static_cast<int64_t>(1))
      .Const1D("axes", static_cast<int64_t>(1))
      .Add("expanded_target = Unsqueeze (target, axes)");

  if (ctx.getAttribute("ignore_index") == nullptr) {
    builder.Add(R"(
      input_gather_element = GatherElements <axis = 1> (input, expanded_target)
      loss_NCdd = Neg (input_gather_element)
      loss_N1dd = Slice (loss_NCdd, const_zero, const_one, const_one)
    )");

    if (!ctx.hasInput(2)) {
      if (reduction_attr == "none") {
        builder.Add("loss = Squeeze (loss_N1dd, axes)");
      } else {
        builder.Add("loss_Ndd = Squeeze (loss_N1dd, axes)");
        if (reduction_attr == "mean") {
          builder.Add("loss = ReduceMean <keepdims = 0> (loss_Ndd)");
        } else {
          builder.Add("loss = ReduceSum <keepdims = 0> (loss_Ndd)");
        }
      }
    } else {
      builder.Add("weight_gather = Gather (weight, target)");
      builder.Add("loss_unweighted = Squeeze (loss_N1dd, axes)");
      if (reduction_attr == "none") {
        builder.Add("loss = Mul (loss_unweighted, weight_gather)");
      } else {
        builder.Add("loss_Ndd = Mul (loss_unweighted, weight_gather)");
        if (reduction_attr == "mean") {
          builder.Add(R"(
            loss_sum = ReduceSum <keepdims = 0> (loss_Ndd)
            weight_gather_sum = ReduceSum <keepdims = 0> (weight_gather)
            loss = Div (loss_sum, weight_gather_sum)
          )");
        } else {
          builder.Add("loss = ReduceSum <keepdims = 0> (loss_Ndd)");
        }
      }
    }
  } else {
    builder.Const1D("const_ignore_index", ctx.getAttribute("ignore_index")->i());
    builder.Add(R"(
      const_zero_target_typed = Sub (expanded_target, expanded_target)
      expanded_target_int64 = Cast <to = 7> (expanded_target)
      mask = Equal (expanded_target_int64, const_ignore_index)
      transform_targets = Where (mask, const_zero_target_typed, expanded_target)
    )");
    builder.Add("input_gather_element = GatherElements <axis = 1> (input, transform_targets)");
    builder.Const1D("const_zero_float", 0.0f);
    if (!float_input) {
      builder.Add("const_zero_casted = Cast (const_zero_float)", "to", static_cast<int64_t>(input_type))
          .Add("input_gather_element_transform = Where (mask, const_zero_casted, input_gather_element)");
    } else {
      builder.Add("input_gather_element_transform = Where (mask, const_zero_float, input_gather_element)");
    }
    builder.Add("loss_NCdd = Neg (input_gather_element_transform)");
    builder.Add("loss_N1dd = Slice (loss_NCdd, const_zero, const_one, const_one)");

    if (!ctx.hasInput(2)) {
      builder.Add("squeeze_mask = Squeeze (mask, axes)");
      builder.Const1D("const_one_float", 1.0f);
      if (!float_input) {
        builder.Add("const_one_casted = Cast (const_one_float)", "to", static_cast<int64_t>(input_type))
            .Add("weight_gather = Where (squeeze_mask, const_zero_casted, const_one_casted)");
      } else {
        builder.Add("weight_gather = Where (squeeze_mask, const_zero_float, const_one_float)");
      }

    } else {
      builder.Add("weight_gather_temp = Gather (weight, transform_targets)");
      builder.Add(
          float_input ? "weight_gather_temp_1 = Where (mask, const_zero_float, weight_gather_temp)"
                      : "weight_gather_temp_1 = Where (mask, const_zero_casted, weight_gather_temp)");
      builder.Add("weight_gather = Squeeze (weight_gather_temp_1, axes)");
    }

    builder.Add("loss_unweighted = Squeeze (loss_N1dd, axes)");
    if (reduction_attr == "none") {
      builder.Add("loss = Mul (loss_unweighted, weight_gather)");
    } else {
      builder.Add("loss_Ndd = Mul (loss_unweighted, weight_gather)");
      if (reduction_attr == "mean") {
        builder.Add(R"(
            loss_sum = ReduceSum <keepdims = 0> (loss_Ndd)
            weight_gather_sum = ReduceSum <keepdims = 0> (weight_gather)
            loss = Div (loss_sum, weight_gather_sum)
        )");
      } else {
        builder.Add("loss = ReduceSum <keepdims = 0> (loss_Ndd)");
      }
    }
  }

  schema.BuildFunction(functionProto);
  return true;
}

void NegativeLogLikelihoodLossShapeInference_opset13(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference
  if (hasNInputShapes(ctx, 2)) {
    const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    const TensorShapeProto& target_shape = ctx.getInputType(1)->tensor_type().shape();

    const int input_rank = static_cast<int>(input_shape.dim_size());
    const int target_rank = static_cast<int>(target_shape.dim_size());

    if (input_rank < 2) {
      fail_shape_inference("Input rank must be >= 2.");
    }
    if (target_rank != input_rank - 1) {
      fail_shape_inference("Target rank must be 1 less than the input rank.");
    }

    // match input dimensions (N, C, d1, ..., dk) with target
    // dimensions of (C, d1, ..., dk)
    for (int dim = 0; dim < target_rank; dim++) {
      const auto input_dim = dim == 0 ? input_shape.dim(dim) : input_shape.dim(dim + 1);
      const auto target_dim = target_shape.dim(dim);
      if (input_dim.has_dim_value() && target_dim.has_dim_value() && input_dim.dim_value() != target_dim.dim_value())
        fail_shape_inference("Input and target dimension value mismatch.");
    }

    if (ctx.getNumInputs() == 3 && hasInputShape(ctx, 2)) {
      const TensorShapeProto& weight_shape = ctx.getInputType(2)->tensor_type().shape();
      if (weight_shape.dim_size() != 1) {
        fail_shape_inference("Weight rank must be 1.");
      }
    }

    TensorShapeProto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

    if (getAttribute(ctx, "reduction", "mean") == "none") {
      // output tensor is of shape (N, d1, d2, ..., dk) if
      // reduction attribute is "none".
      for (int i = 0; i < input_rank - 1; i++) {
        auto dim = output_shape->add_dim();
        if (i == 0)
          *dim = input_shape.dim(i);
        else
          *dim = input_shape.dim(i + 1);
      }
    }
    // otherwise output is a scalar.
  }
}

void DetShapeInference_opset11(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference
  if (hasInputShape(ctx, 0)) {
    const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    TensorShapeProto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
    const int rank = static_cast<int>(input_shape.dim_size());

    if (rank < 2) {
      fail_shape_inference("Input rank must be >= 2.");
    }

    const auto mat_w = input_shape.dim(rank - 1);
    const auto mat_h = input_shape.dim(rank - 2);
    if (mat_w.has_dim_value() && mat_h.has_dim_value() && (mat_w.dim_value() != mat_h.dim_value())) {
      fail_shape_inference(
          "The inner-most 2 dimensions must have the same size (mat_w:",
          mat_w.dim_value(),
          " != mat_h:",
          mat_h.dim_value(),
          ").");
    }

    for (int i = 0; i < rank - 2; ++i) {
      auto dim = output_shape->add_dim();
      *dim = input_shape.dim(i);
    }
  }
}

void MathShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void MathShapeInference_opset_7(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

static void SoftmaxFamilyShapeInference_opset11(InferenceContext& ctx, const char* name) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference starts
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  // Validate the value of 'axis'
  const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  int r = input_shape.dim_size();
  if (r == 0) {
    fail_shape_inference("Input rank must be >= 1 for ", name, ".");
  }
  int axis = static_cast<int>(getAttribute(ctx, "axis", 1));
  if (axis < -r || axis >= r) {
    fail_shape_inference("'axis' must be in [", -r, " , ", (r - 1), "]. Its actual value is: ", axis);
  }

  // Shape inference
  propagateShapeFromInputToOutput(ctx, 0, 0);
}

void SoftmaxShapeInference_opset11(InferenceContext& ctx) {
  SoftmaxFamilyShapeInference_opset11(ctx, "softmax");
}

void LogSoftmaxShapeInference_opset11(InferenceContext& ctx) {
  SoftmaxFamilyShapeInference_opset11(ctx, "logsoftmax");
}

void HardmaxShapeInference_opset11(InferenceContext& ctx) {
  SoftmaxFamilyShapeInference_opset11(ctx, "hardmax");
}

void ModShapeInference_opset10(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void PowShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void PowShapeInference_opset12(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void ElementwiseMultiOpShapeInference_opset8(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto num_inputs = ctx.getNumInputs();
  std::vector<const TensorShapeProto*> shapes;
  for (size_t i = 0; i < num_inputs; ++i) {
    const auto* const input_type = ctx.getInputType(i);
    if (nullptr == input_type || !input_type->has_tensor_type() || !input_type->tensor_type().has_shape()) {
      return;
    }
    shapes.push_back(&input_type->tensor_type().shape());
  }

  multidirectionalBroadcastShapeInference(shapes, *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void GemmShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto transAAttr = ctx.getAttribute("transA");
  bool transA = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
  auto transBAttr = ctx.getAttribute("transB");
  bool transB = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
  // Gemm computes Y = alpha * A' * B' + beta * C
  // A' is A or A^T depending on transA; B' is B or B^T depending on transB
  Dim M, N, K;
  if (transA) {
    ctx.unifyInputShape(0, {K, M});
  } else {
    ctx.unifyInputShape(0, {M, K});
  }
  if (transB) {
    ctx.unifyInputShape(1, {N, K});
  } else {
    ctx.unifyInputShape(1, {K, N});
  }
  // Output shape is (M, N)
  updateOutputShape(ctx, 0, {M, N});
}

void GemmShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 2)) {
    auto transAAttr = ctx.getAttribute("transA");
    bool transA = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
    auto transBAttr = ctx.getAttribute("transB");
    bool transB = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
    auto& first_input_shape = getInputShape(ctx, 0);
    auto& second_input_shape = getInputShape(ctx, 1);
    if (first_input_shape.dim_size() != 2) {
      fail_shape_inference("First input does not have rank 2");
    }
    if (second_input_shape.dim_size() != 2) {
      fail_shape_inference("Second input does not have rank 2");
    }
    updateOutputShape(ctx, 0, {first_input_shape.dim(transA ? 1 : 0), second_input_shape.dim(transB ? 0 : 1)});
  }
}

static void matmulShapeInference_opset_9(ONNX_NAMESPACE::InferenceContext& ctx, size_t input1Idx, size_t input2Idx) {
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

  *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = std::move(resultShape);
}

void MatMulShapeInference_opset9(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  matmulShapeInference_opset_9(ctx, 0, 1);
}

void ExpandShapeInference_opset8(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference
  // For shape inference, we need both input shape
  const auto shape_initializer = ctx.getInputData(1);
  if (hasNInputShapes(ctx, 2)) {
    const auto& shape_input_shape = ctx.getInputType(1)->tensor_type().shape();
    if (shape_input_shape.dim_size() != 1) {
      fail_shape_inference("'shape' input must be 1D tensor");
    }

    const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    TensorShapeProto second_shape;
    if (nullptr != shape_initializer) {
      const auto shape_data = ParseData<int64_t>(shape_initializer);

      for (const auto& e : shape_data) {
        auto dim = second_shape.add_dim();
        dim->set_dim_value(e);
      }
    } else if (shape_input_shape.dim(0).has_dim_value()) {
      // Attempt rank inference using shape of shape input
      int64_t dim_value = shape_input_shape.dim(0).dim_value();
      for (int64_t i = 0; i < dim_value; ++i) {
        second_shape.add_dim();
      }
    } else {
      return;
    }
    bidirectionalBroadcastShapeInference(input_shape, second_shape, *getOutputShape(ctx, 0));
  }
}

static TensorProto ToDimensionOneFloatTensor_old(float value) {
  auto t = ToTensor(std::vector<float>({value}));
  t.add_dims(1);
  return t;
}

static TensorProto ToDimensionOneTensor_old(int32_t value) {
  auto t = ToTensor(std::vector<int32_t>({value}));
  t.add_dims(1);
  return t;
}

static TensorProto ToDimensionOneInt64Tensor_old(int64_t value) {
  auto t = ToTensor(std::vector<int64_t>({value}));
  t.add_dims(1);
  return t;
}

static TensorProto ToDimensionOneInt64Tensor_old(const std::vector<int64_t>& value) {
  auto t = ToTensor(value);
  t.add_dims(static_cast<int64_t>(value.size()));
  return t;
}

bool BuildContextDependentFunctionBody_opset12(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  if (ctx.getInputType(0) == nullptr) {
    // we cannot create a correct function body without knowing the input type
    return false;
  }
  auto input_type = ctx.getInputType(0)->tensor_type().elem_type();
  bool float_input = input_type == TensorProto_DataType_FLOAT;
  const auto* const reduction_attr_proto = ctx.getAttribute("reduction");
  std::string reduction_attr =
      reduction_attr_proto != nullptr && reduction_attr_proto->has_s() ? reduction_attr_proto->s() : "mean";
  std::vector<FunctionBodyHelper::NodeDef> body;
  body.reserve(23);
  body.push_back({{"const_zero"}, "Constant", {}, {MakeAttribute("value", ToDimensionOneTensor_old(0))}});

  body.push_back({{"const_one"}, "Constant", {}, {MakeAttribute("value", ToDimensionOneTensor_old(1))}});

  body.push_back({{"expanded_target"}, "Unsqueeze", {"target"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});

  if (ctx.getAttribute("ignore_index") == nullptr) {
    body.push_back(
        {{"input_gather_element"},
         "GatherElements",
         {"input", "expanded_target"},
         {MakeAttribute("axis", static_cast<int64_t>(1))}});

    body.push_back({{"loss_NCdd"}, "Neg", {"input_gather_element"}});

    body.push_back({{"loss_N1dd"}, "Slice", {"loss_NCdd", "const_zero", "const_one", "const_one"}});

    if (!ctx.hasInput(2)) {
      if (reduction_attr == "none") {
        body.push_back({{"loss"}, "Squeeze", {"loss_N1dd"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});
      } else {
        body.push_back({{"loss_Ndd"}, "Squeeze", {"loss_N1dd"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});
        if (reduction_attr == "mean") {
          body.push_back({{"loss"}, "ReduceMean", {"loss_Ndd"}, {MakeAttribute("keepdims", static_cast<int64_t>(0))}});
        } else {
          body.push_back({{"loss"}, "ReduceSum", {"loss_Ndd"}, {MakeAttribute("keepdims", static_cast<int64_t>(0))}});
        }
      }
    } else {
      body.push_back({{"weight_gather"}, "Gather", {"weight", "target"}});
      body.push_back(
          {{"loss_unweighted"}, "Squeeze", {"loss_N1dd"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});
      if (reduction_attr == "none") {
        body.push_back({{"loss"}, "Mul", {"loss_unweighted", "weight_gather"}});
      } else {
        body.push_back({{"loss_Ndd"}, "Mul", {"loss_unweighted", "weight_gather"}});
        if (reduction_attr == "mean") {
          body.push_back(
              {{"loss_sum"}, "ReduceSum", {"loss_Ndd"}, {MakeAttribute("keepdims", static_cast<int64_t>(0))}});
          body.push_back(
              {{"weight_gather_sum"},
               "ReduceSum",
               {"weight_gather"},
               {MakeAttribute("keepdims", static_cast<int64_t>(0))}});
          body.push_back({{"loss"}, "Div", {"loss_sum", "weight_gather_sum"}});
        } else {
          body.push_back({{"loss"}, "ReduceSum", {"loss_Ndd"}, {MakeAttribute("keepdims", static_cast<int64_t>(0))}});
        }
      }
    }
  } else {
    body.push_back(
        {{"const_ignore_index"},
         "Constant",
         {},
         {MakeAttribute("value", ToDimensionOneInt64Tensor_old(ctx.getAttribute("ignore_index")->i()))}});

    body.push_back({{"const_zero_target_typed"}, "Sub", {"expanded_target", "expanded_target"}});
    body.push_back(
        {{"expanded_target_int64"},
         "Cast",
         {"expanded_target"},
         {MakeAttribute("to", static_cast<int64_t>(TensorProto_DataType::TensorProto_DataType_INT64))}});

    body.push_back({{"mask"}, "Equal", {"expanded_target_int64", "const_ignore_index"}});
    body.push_back({{"transform_targets"}, "Where", {"mask", "const_zero_target_typed", "expanded_target"}});
    body.push_back(
        {{"input_gather_element"},
         "GatherElements",
         {"input", "transform_targets"},
         {MakeAttribute("axis", static_cast<int64_t>(1))}});
    body.push_back(
        {{"const_zero_float"}, "Constant", {}, {MakeAttribute("value", ToDimensionOneFloatTensor_old(0.0f))}});
    if (!float_input) {
      body.push_back(
          {{"const_zero_casted"},
           "Cast",
           {"const_zero_float"},
           {MakeAttribute("to", static_cast<int64_t>(input_type))}});
    }
    body.push_back(
        {{"input_gather_element_transform"},
         "Where",
         {"mask", float_input ? "const_zero_float" : "const_zero_casted", "input_gather_element"}});
    body.push_back({{"loss_NCdd"}, "Neg", {"input_gather_element_transform"}});
    body.push_back({{"loss_N1dd"}, "Slice", {"loss_NCdd", "const_zero", "const_one", "const_one"}});

    if (!ctx.hasInput(2)) {
      body.push_back({{"squeeze_mask"}, "Squeeze", {"mask"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});

      body.push_back(
          {{"const_one_float"}, "Constant", {}, {MakeAttribute("value", ToDimensionOneFloatTensor_old(1.0f))}});
      if (!float_input) {
        body.push_back(
            {{"const_one_casted"},
             "Cast",
             {"const_one_float"},
             {MakeAttribute("to", static_cast<int64_t>(input_type))}});
      }
      body.push_back(
          {{"weight_gather"},
           "Where",
           {"squeeze_mask",
            float_input ? "const_zero_float" : "const_zero_casted",
            float_input ? "const_one_float" : "const_one_casted"}});

    } else {
      body.push_back({{"weight_gather_temp"}, "Gather", {"weight", "transform_targets"}});

      body.push_back(
          {{"weight_gather_temp_1"},
           "Where",
           {"mask", float_input ? "const_zero_float" : "const_zero_casted", "weight_gather_temp"}});

      body.push_back(
          {{"weight_gather"}, "Squeeze", {"weight_gather_temp_1"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});
    }

    body.push_back({{"loss_unweighted"}, "Squeeze", {"loss_N1dd"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});
    if (reduction_attr == "none") {
      body.push_back({{"loss"}, "Mul", {"loss_unweighted", "weight_gather"}});
    } else {
      body.push_back({{"loss_Ndd"}, "Mul", {"loss_unweighted", "weight_gather"}});
      if (reduction_attr == "mean") {
        body.push_back({{"loss_sum"}, "ReduceSum", {"loss_Ndd"}, {MakeAttribute("keepdims", static_cast<int64_t>(0))}});
        body.push_back(
            {{"weight_gather_sum"},
             "ReduceSum",
             {"weight_gather"},
             {MakeAttribute("keepdims", static_cast<int64_t>(0))}});
        body.push_back({{"loss"}, "Div", {"loss_sum", "weight_gather_sum"}});
      } else {
        body.push_back({{"loss"}, "ReduceSum", {"loss_Ndd"}, {MakeAttribute("keepdims", static_cast<int64_t>(0))}});
      }
    }
  }

  auto func_nodes = FunctionBodyHelper::BuildNodes(body);
  for (const auto& node : func_nodes) {
    auto* new_node = functionProto.add_node();
    new_node->CopyFrom(node);
  }

  schema.BuildFunction(functionProto);
  return true;
}

void NegativeLogLikelihoodLossShapeInference_opset12(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference
  if (hasNInputShapes(ctx, 2)) {
    const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    const TensorShapeProto& target_shape = ctx.getInputType(1)->tensor_type().shape();

    const int input_rank = static_cast<int>(input_shape.dim_size());
    const int target_rank = static_cast<int>(target_shape.dim_size());

    if (input_rank < 2) {
      fail_shape_inference("Input rank must be >= 2. input_rank=", input_rank);
    }
    if (target_rank != input_rank - 1) {
      fail_shape_inference(
          "Target rank must be 1 less than the input rank. input_rank=", input_rank, ", target_rank=", target_rank);
    }

    // match input dimensions (N, C, d1, ..., dk) with target
    // dimensions of (C, d1, ..., dk)
    for (int dim = 0; dim < target_rank; dim++) {
      const auto input_dim = dim == 0 ? input_shape.dim(dim) : input_shape.dim(dim + 1);
      const auto target_dim = target_shape.dim(dim);
      if (input_dim.has_dim_value() && target_dim.has_dim_value() && input_dim.dim_value() != target_dim.dim_value())
        fail_shape_inference(
            "Input and target dimension value mismatch. input_dim_value=",
            input_dim.dim_value(),
            " target_dim_value=",
            target_dim.dim_value());
    }

    if (ctx.getNumInputs() == 3 && hasInputShape(ctx, 2)) {
      const TensorShapeProto& weight_shape = ctx.getInputType(2)->tensor_type().shape();
      const auto weight_rank = weight_shape.dim_size();
      if (weight_rank != 1) {
        fail_shape_inference("Weight rank must be 1. weight_rank=", weight_rank);
      }
    }

    TensorShapeProto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
    if (getAttribute(ctx, "reduction", "mean") == "none") {
      // output tensor is of shape (N, d1, d2, ..., dk) if
      // reduction attribute is "none".
      for (int i = 0; i < input_rank - 1; i++) {
        auto dim = output_shape->add_dim();
        if (i == 0)
          *dim = input_shape.dim(i);
        else
          *dim = input_shape.dim(i + 1);
      }
    }
    // otherwise output is a scalar.
  }
}

bool BuildContextDependentFunctionBodySCE_opset12(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  std::vector<FunctionBodyHelper::NodeDef> body;
  body.reserve(9);

  // Using stable implementation of LogSoftmax
  body.push_back({{"Shape3D"}, "Constant", {}, {MakeAttribute("value", ToDimensionOneInt64Tensor_old({0, 0, -1}))}});
  body.push_back({{"X_NCD"}, "Reshape", {"scores", "Shape3D"}});
  body.push_back({{"X_NDC"}, "Transpose", {"X_NCD"}, {MakeAttribute("perm", std::vector<int64_t>({0, 2, 1}))}});
  body.push_back({{"X_LogSM"}, "LogSoftmax", {"X_NDC"}, {MakeAttribute("axis", static_cast<int64_t>(2))}});
  body.push_back({{"X_LogSM_NCD"}, "Transpose", {"X_LogSM"}, {MakeAttribute("perm", std::vector<int64_t>({0, 2, 1}))}});
  body.push_back({{"X_shape"}, "Shape", {"scores"}});
  body.push_back({{"X_Log"}, "Reshape", {"X_LogSM_NCD", "X_shape"}});

  // Review(mzs): Ideally we want to reuse the output from Log for sub-graph
  // output as well but looking at the graph resolve code it does not include
  // graph outputs as intermediate outputs, hence if intermediate X_log is
  // renamed as log_prob then it will be treated as graph output and will not be
  // available to NegativeLogLikelihoodLoss. May be my understanding is
  // incorrect or there is a bug in function population code in ORTbut I will
  // dig further to be 100%. In the meantime we just replicate the log.
  if (ctx.hasOutput(1)) {
    body.push_back({{"log_prob"}, "Identity", {"X_Log"}});
  }

  std::vector<std::string> input_tensor_names{"X_Log", "labels"};
  std::vector<FunctionBodyHelper::AttributeProtoWrapper> attributes{
      MakeRefAttribute("reduction", AttributeProto::STRING)};
  // Add weights as input if needed.
  if (ctx.hasInput(2)) {
    input_tensor_names.emplace_back("weights");
  }

  // add ignore_index attributes if needed.
  if (ctx.getAttribute("ignore_index") != nullptr) {
    attributes.emplace_back(MakeRefAttribute("ignore_index", AttributeProto::INT));
  }

  body.push_back({{"output"}, "NegativeLogLikelihoodLoss", input_tensor_names, attributes});

  auto func_nodes = FunctionBodyHelper::BuildNodes(body);
  for (const auto& node : func_nodes) {
    auto* new_node = functionProto.add_node();
    new_node->CopyFrom(node);
  }

  schema.BuildFunction(functionProto);
  return true;
}

void SoftmaxCrossEntropyLossShapeInference_opset12(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  std::string reduction = getAttribute(ctx, "reduction", "mean");
  if (reduction == "none") {
    if (hasInputShape(ctx, 1)) {
      propagateShapeFromInputToOutput(ctx, 1, 0);
    }
  } else {
    updateOutputShape(ctx, 0, TensorShapeProto());
  }

  if (ctx.getNumOutputs() == 2) {
    propagateElemTypeFromInputToOutput(ctx, 0, 1);
    propagateShapeFromInputToOutput(ctx, 0, 1);
  }
}

void PowShapeInference_opset7(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void GemmShapeInference_opset6(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 2)) {
    auto transAAttr = ctx.getAttribute("transA");
    bool transA = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
    auto transBAttr = ctx.getAttribute("transB");
    bool transB = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;

    checkInputRank(ctx, 0, 2);
    checkInputRank(ctx, 1, 2);

    auto& first_input_shape = getInputShape(ctx, 0);
    auto& second_input_shape = getInputShape(ctx, 1);
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = first_input_shape.dim(transA ? 1 : 0);
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = second_input_shape.dim(transB ? 0 : 1);
  } else if (
      hasInputShape(ctx, 2) &&
      (!ctx.getAttribute("broadcast") || static_cast<int>(ctx.getAttribute("broadcast")->i()) == 0)) {
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(2)->tensor_type().shape();
  }
}

void GemmShapeInference_opset7(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 2)) {
    auto transAAttr = ctx.getAttribute("transA");
    bool transA = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
    auto transBAttr = ctx.getAttribute("transB");
    bool transB = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
    auto& first_input_shape = getInputShape(ctx, 0);
    auto& second_input_shape = getInputShape(ctx, 1);
    if (first_input_shape.dim_size() != 2) {
      fail_shape_inference("First input does not have rank 2");
    }
    if (second_input_shape.dim_size() != 2) {
      fail_shape_inference("Second input does not have rank 2");
    }
    updateOutputShape(ctx, 0, {first_input_shape.dim(transA ? 1 : 0), second_input_shape.dim(transB ? 0 : 1)});
  }
}

void GemmShapeInference_opset9(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 2)) {
    auto transAAttr = ctx.getAttribute("transA");
    bool transA = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
    auto transBAttr = ctx.getAttribute("transB");
    bool transB = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
    auto& first_input_shape = getInputShape(ctx, 0);
    auto& second_input_shape = getInputShape(ctx, 1);
    if (first_input_shape.dim_size() != 2) {
      fail_shape_inference("First input does not have rank 2");
    }
    if (second_input_shape.dim_size() != 2) {
      fail_shape_inference("Second input does not have rank 2");
    }
    updateOutputShape(ctx, 0, {first_input_shape.dim(transA ? 1 : 0), second_input_shape.dim(transB ? 0 : 1)});
  }
}

void MatMulShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  const auto shape0 = ctx.getInputType(0)->tensor_type().shape();
  const auto shape1 = ctx.getInputType(1)->tensor_type().shape();

  if (shape0.dim_size() == 0 || shape1.dim_size() == 0) {
    fail_shape_inference("Input tensors of wrong rank (0).");
  }

  TensorShapeProto shapeL, shapeR;

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
    auto const& dimL = shapeL.dim(shapeL.dim_size() - 1);
    auto const& dimR = shapeR.dim(shapeR.dim_size() - 2);
    if (dimL.has_dim_value() && dimR.has_dim_value() && dimL.dim_value() != dimR.dim_value()) {
      fail_shape_inference("Incompatible dimensions for matrix multiplication");
      ;
    }
  }

  TensorShapeProto resultShape;

  // Now call out to generic multidimensional broadcasting for
  // the broadcastable prefixes.
  {
    TensorShapeProto prefixShapeL, prefixShapeR;
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

void TopKShapeInference_opset1(InferenceContext& ctx) {
  // Type inference:
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  updateOutputElemType(ctx, 1, TensorProto::INT64);

  // Shape inference:
  if (!hasInputShape(ctx, 0))
    return;
  auto& input_shape = getInputShape(ctx, 0);
  int64_t rank = input_shape.dim_size();
  int64_t axis = getAttribute(ctx, "axis", -1);
  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank) {
    fail_shape_inference("Invalid value for attribute axis");
  }
  int64_t k = getAttribute(ctx, "k", -1);
  if (k <= 0) {
    fail_shape_inference("Invalid value for attribute k");
  }
  TensorShapeProto result_shape = input_shape;
  result_shape.mutable_dim(static_cast<int>(axis))->set_dim_value(k);
  updateOutputShape(ctx, 0, result_shape);
  updateOutputShape(ctx, 1, result_shape);
}

void TopKShapeInference_opset10(InferenceContext& ctx) {
  // Type inference:
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  updateOutputElemType(ctx, 1, TensorProto::INT64);
  // Shape inference:
  if (!hasInputShape(ctx, 0))
    return;
  auto& input_shape = getInputShape(ctx, 0);
  int64_t rank = input_shape.dim_size();
  int64_t axis = getAttribute(ctx, "axis", -1);
  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank) {
    fail_shape_inference("Invalid value for attribute axis");
  }

  const auto& axis_dim = input_shape.dim(static_cast<int>(axis));
  const auto k = ctx.getInputData(1);

  int64_t k_value = 0;
  if (nullptr != k) {
    if (k->dims_size() != 1 || k->dims(0) != 1) {
      fail_shape_inference("K input must be a one-dimensional tensor of size 1.");
    }

    if (k->data_type() == TensorProto::INT64) {
      const auto data = ParseData<int64_t>(k);
      k_value = data[0];
    } else {
      fail_shape_inference("K input must be of type int64.");
    }

    if (k_value <= 0) {
      fail_shape_inference("K input must contain a single positive value.");
    }
  }

  // Infer output shape if:
  // (1) 'K' is available
  // (2) axis_dim has dim value
  // Otherwise cannot reliably compute output shape as axis dim value is
  // unknown and hence cannot determine if axis dim value >= k (which
  // should be enforced)
  if (nullptr != k && axis_dim.has_dim_value()) {
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
  auto output_shape_0 = getOutputShape(ctx, 0);
  auto output_shape_1 = getOutputShape(ctx, 1);
  for (int i = 0; i < input_shape.dim_size(); ++i) {
    output_shape_0->add_dim();
    output_shape_1->add_dim();
  }

  return;
}

void ElementwiseMultiOpShapeInference_old(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto num_inputs = ctx.getNumInputs();
  std::vector<const TensorShapeProto*> shapes;
  for (size_t i = 0; i < num_inputs; ++i) {
    const auto* const input_type = ctx.getInputType(i);
    if (nullptr == input_type || !input_type->has_tensor_type() || !input_type->tensor_type().has_shape()) {
      return;
    }
    shapes.push_back(&input_type->tensor_type().shape());
  }

  multidirectionalBroadcastShapeInference(shapes, *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void DFTShapeInference_opset17(ONNX_NAMESPACE::InferenceContext& ctx) {
  bool is_onesided = static_cast<bool>(getAttribute(ctx, "onesided", 0));
  bool inverse = static_cast<bool>(getAttribute(ctx, "inverse", 0));

  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasInputShape(ctx, 0)) {
    // If no shape is available for the input, skip shape inference...
    return;
  }

  // In general the output shape will match the input shape exactly
  // So initialize the output shape with the input shape
  auto& input_shape = getInputShape(ctx, 0);
  ONNX_NAMESPACE::TensorShapeProto result_shape_proto = input_shape;

  // Get the axis where the DFT will be performed.
  auto axis = static_cast<int>(getAttribute(ctx, "axis", 1));
  // The last dimension is the real and imaginary parts of the value.
  const int64_t rank = input_shape.dim_size();
  if (rank < 2) {
    fail_shape_inference("input tensor must have rank >= 2, including the complex dimension.");
  }

  // check the inputs are correct types for one-sided DFT
  if (is_onesided) {
    auto last_dim = input_shape.dim(rank - 1);
    if (inverse) {
      // Check last dimension is 2 (complex input required)
      if (last_dim.has_dim_value() && last_dim.dim_value() != 2) {
        fail_shape_inference("inverse one-sided DFT requires complex input (last dimension must be 2)");
      }
    } else {
      // Check last dimension is 1 (real input required)
      if (last_dim.has_dim_value() && last_dim.dim_value() != 1) {
        fail_shape_inference("one-sided DFT requires real input (last dimension must be 1)");
      }
    }
  }
  // NOLINTNEXTLINE(readability-simplify-boolean-expr)
  if (!(-rank <= axis && axis != -1 && axis < rank - 1)) {
    fail_shape_inference(
        "axis attribute value ",
        axis,
        " is invalid for a tensor of rank ",
        rank,
        ". Valid values are '-rank <= axis && axis != -1 && axis < rank - 1'");
  }

  int axis_idx = static_cast<int>(axis >= 0 ? axis : axis + rank);

  // If dft_length is specified, then we should honor the shape.
  // Set the output dimension to match the dft_length on the axis.
  const TensorProto* dft_length = nullptr;
  if (ctx.hasInput(1)) {
    dft_length = ctx.getInputData(1);
    if (dft_length == nullptr) {
      // If we cannot read the dft_length, we cannot infer shape
      // return...
      return;
    }
  }

  if (nullptr != dft_length) {
    if (dft_length->dims_size() != 0) {
      fail_shape_inference("dft_length input must be a scalar.");
    }
    auto dft_length_value = defs::math::utils::GetScalarValueFromTensor<int64_t>(dft_length);

    // For RFFT, output size on signal axis is floor(dft_length/2) + 1
    if (is_onesided && !inverse) {
      // RFFT: one-sided output
      auto half_signal_size = (dft_length_value / 2) + 1;
      result_shape_proto.mutable_dim(axis_idx)->set_dim_value(half_signal_size);
    } else {
      // Standard FFT/IFFT and IRFFT: full length
      result_shape_proto.mutable_dim(axis_idx)->set_dim_value(dft_length_value);
    }
  } else if (is_onesided) {
    auto axis_dimension = result_shape_proto.dim(axis_idx);
    if (axis_dimension.has_dim_value()) {
      auto axis_dimension_value = axis_dimension.dim_value();
      if (inverse) {
        // IRFFT without explicit dft_length: cannot reliably infer full signal length
        // Default to even length: N = 2 * (input_size - 1)
        auto full_signal_size = checkedMultiply(2, checkedSubtract(axis_dimension_value, 1));
        result_shape_proto.mutable_dim(axis_idx)->set_dim_value(full_signal_size);
      } else {
        // RFFT without explicit dft_length: infer one-sided output size from input
        auto half_signal_size = (axis_dimension_value / 2) + 1;
        result_shape_proto.mutable_dim(axis_idx)->set_dim_value(half_signal_size);
      }
    } else {
      result_shape_proto.mutable_dim(axis_idx)->clear_dim_value();
      result_shape_proto.mutable_dim(axis_idx)->clear_dim_param();
    }
  }

  // Set the last dimension based on whether output is real or complex
  auto dim_size = static_cast<int64_t>(result_shape_proto.dim_size());
  if (is_onesided && inverse) {
    // IRFFT: complex input -> real output (last dim = 1)
    result_shape_proto.mutable_dim(static_cast<int>(dim_size - 1))->set_dim_value(1);
  } else {
    // All other cases: complex output (last dim = 2)
    result_shape_proto.mutable_dim(static_cast<int>(dim_size - 1))->set_dim_value(2);
  }

  updateOutputShape(ctx, 0, result_shape_proto);
}

static float celu_ver12_default_alpha = 1.0;

bool BuildContextDependentFunctionBodyCelu(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  float alpha = ctx.getAttribute("alpha") != nullptr ? ctx.getAttribute("alpha")->f() : celu_ver12_default_alpha;
  FunctionBuilder builder(functionProto);
  builder.Const("alpha", std::vector<float>{alpha}).Add(R"(
            X_alpha = Div (X, alpha)
            Elu_Result = Elu <alpha = 1.0>(X_alpha)
            Y = Mul (alpha, Elu_Result)
        )");
  schema.BuildFunction(functionProto);
  return true;
}

void ModShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

} // namespace ONNX_NAMESPACE::defs::math::utils
