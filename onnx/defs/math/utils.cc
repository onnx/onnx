/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/math/utils.h"

#include <string>

namespace ONNX_NAMESPACE {

std::function<void(OpSchema&)>
SoftmaxFamilyDocGenerator(const char* name, const char* description, const char* equation) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
The operator computes the {description} values for the given input:

 {equation}

The "axis" attribute indicates the dimension along which {name}
will be performed. The output tensor has the same shape
and contains the {name} values of the corresponding input.
)DOC";
                        ReplaceAll(doc, "{name}", name);
                        ReplaceAll(doc, "{description}", description);
                        ReplaceAll(doc, "{equation}", equation););
    std::string axis_attr;
    POPULATE_OP_DOC_STR(axis_attr = R"DOC(
Describes the dimension {name} will be performed on.
Negative value means counting dimensions
from the back. Accepted range is [-r, r-1] where r = rank(input).
)DOC";
                        ReplaceAll(axis_attr, "{name}", name););
    schema.SetDoc(doc);
    schema.Attr("axis", axis_attr, AttributeProto::INT, static_cast<int64_t>(-1));
    schema.Input(
        0, "input", "The input tensor of rank >= axis.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable);
    schema.Output(
        0,
        "output",
        "The output values with the same shape as the input tensor.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
        "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);

      // Shape inference starts
      if (!hasNInputShapes(ctx, 1)) {
        return;
      }

      // Validate the value of 'axis'
      const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
      int r = input_shape.dim_size();
      int axis = static_cast<int>(getAttribute(ctx, "axis", -1));
      if (axis < -r || axis >= r) {
        fail_shape_inference("'axis' must be in [", -r, " , ", (r - 1), "]. Its actual value is: ", axis);
      }

      // Shape inference
      propagateShapeFromInputToOutput(ctx, 0, 0);
    });
  };
}

void qlinear_matmul_shape_inference(ONNX_NAMESPACE::InferenceContext& ctx) {
  auto a_type = ctx.getInputType(0);
  auto b_type = ctx.getInputType(3);
  if (nullptr == a_type || nullptr == b_type || a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
      b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
    fail_type_inference("inputs are expected to have tensor type.");
  }

  auto a_zero_point_type = ctx.getInputType(2);
  if (nullptr == a_zero_point_type ||
      a_zero_point_type->tensor_type().elem_type() != a_type->tensor_type().elem_type()) {
    fail_type_inference("input and zero_point pair is expected to have be same type.");
  }

  auto b_zero_point_type = ctx.getInputType(5);
  if (nullptr == b_zero_point_type ||
      b_zero_point_type->tensor_type().elem_type() != b_type->tensor_type().elem_type()) {
    fail_type_inference("input and zero_point pair is expected to have same type.");
  }

  propagateElemTypeFromInputToOutput(ctx, 7, 0);

  matmulShapeInference(ctx, 0, 3);
}

const char* qlinear_matmul_doc() {
  static const char* QLinearMatMul_doc = R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
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

} // namespace ONNX_NAMESPACE
