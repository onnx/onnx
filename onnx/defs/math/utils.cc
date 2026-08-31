// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/math/utils.h"

#include <string>
#include <utility>
#include <vector>

#include "onnx/common/safe_math.h"
#include "onnx/defs/doc_strings.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE::defs::math::utils {

std::function<void(OpSchema&)> TopKOpGenerator(std::vector<std::string> allowed_types) {
  return [allowed_types = std::move(allowed_types)](OpSchema& schema) {
    schema.SetDoc(kDoc_TopK_ver11)
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

} // namespace ONNX_NAMESPACE::defs::math::utils
