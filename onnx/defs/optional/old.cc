// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/optional/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {
static void OptionalHasElementShapeInference_opset15(InferenceContext& ctx) {
  const size_t numInputs = ctx.getNumInputs();
  if (numInputs != 1) {
    fail_type_inference("OptionalHasElement is expected to have 1 input.");
  }
  const size_t numOutputs = ctx.getNumOutputs();
  if (numOutputs != 1) {
    fail_type_inference("OptionalHasElement is expected to have 1 output.");
  }
  auto output_tensor_type = ctx.getOutputType(0)->mutable_tensor_type();
  output_tensor_type->set_elem_type(TensorProto::BOOL);
  output_tensor_type->mutable_shape()->Clear();
}

ONNX_OPERATOR_SET_SCHEMA(
    OptionalHasElement,
    15,
    generated::ai_onnx::OptionalHasElement_ver15().TypeAndShapeInferenceFunction(
        OptionalHasElementShapeInference_opset15));

static void OptionalGetElementShapeInference_opset15(InferenceContext& ctx) {
  const size_t numInputs = ctx.getNumInputs();
  if (numInputs != 1) {
    fail_type_inference("OptionalGetElement must have an input element.");
  }
  auto input_type = ctx.getInputType(0);
  if (input_type == nullptr) {
    fail_type_inference("Input type is null. Input must have Type information.");
  }
  if (!input_type->has_optional_type() || !input_type->optional_type().has_elem_type()) {
    fail_type_inference("Input must be an optional-type value containing an element with type information.");
  }
  ctx.getOutputType(0)->CopyFrom(input_type->optional_type().elem_type());
}

ONNX_OPERATOR_SET_SCHEMA(
    OptionalGetElement,
    15,
    generated::ai_onnx::OptionalGetElement_ver15().TypeAndShapeInferenceFunction(
        OptionalGetElementShapeInference_opset15));

ONNX_OPERATOR_SET_SCHEMA(
    Optional,
    15,
    generated::ai_onnx::Optional_ver15().FillUsing(defs::optional::utils::OptionalOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    OptionalHasElement,
    18,
    generated::ai_onnx::OptionalHasElement_ver18().FillUsing(defs::optional::utils::OptionalHasElementOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    OptionalGetElement,
    18,
    generated::ai_onnx::OptionalGetElement_ver18().FillUsing(defs::optional::utils::OptionalGetElementOpGenerator()));

} // namespace ONNX_NAMESPACE
