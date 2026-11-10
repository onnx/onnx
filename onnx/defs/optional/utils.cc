// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/optional/utils.h"

#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE::defs::optional::utils {

void OptionalInference(InferenceContext& ctx) {
  const size_t numOutputs = ctx.getNumOutputs();
  if (numOutputs != 1) {
    fail_type_inference("Optional is expected to have an output.");
  }

  const size_t numInputs = ctx.getNumInputs();
  const auto* attr_proto = ctx.getAttribute("type");

  if ((numInputs == 0) && (attr_proto != nullptr)) {
    if (!attr_proto->has_tp()) {
      fail_type_inference("Attribute 'type' should be a TypeProto and it should specify a type.");
    }
    ctx.getOutputType(0)->mutable_optional_type()->mutable_elem_type()->CopyFrom(attr_proto->tp());
  } else if (numInputs == 1) {
    const auto* input_type = ctx.getInputType(0);
    if (input_type == nullptr) {
      fail_type_inference("Input type is null. Type information is expected for the input.");
    }
    ctx.getOutputType(0)->mutable_optional_type()->mutable_elem_type()->CopyFrom(*input_type);
  } else {
    fail_type_inference("Optional is expected to have either an input or the type attribute set.");
  }
}

void OptionalHasElementInference(InferenceContext& ctx) {
  const size_t numInputs = ctx.getNumInputs();
  if (numInputs != 0 && numInputs != 1) {
    fail_type_inference("OptionalHasElement is expected to have 0 or 1 input.");
  }
  const size_t numOutputs = ctx.getNumOutputs();
  if (numOutputs != 1) {
    fail_type_inference("OptionalHasElement is expected to have 1 output.");
  }
  auto* output_tensor_type = ctx.getOutputType(0)->mutable_tensor_type();
  output_tensor_type->set_elem_type(TensorProto::BOOL);
  output_tensor_type->mutable_shape()->Clear();
}

void OptionalGetElementInference(InferenceContext& ctx) {
  const size_t numInputs = ctx.getNumInputs();
  if (numInputs != 1) {
    fail_type_inference("OptionalGetElement must have an input element.");
  }
  const auto* input_type = ctx.getInputType(0);
  if (input_type == nullptr) {
    fail_type_inference("Input type is null. Input must have Type information.");
  }
  if (input_type->has_optional_type()) {
    if (!input_type->optional_type().has_elem_type()) {
      fail_type_inference("Optional-type input must contain an element with type information.");
    }
    ctx.getOutputType(0)->CopyFrom(input_type->optional_type().elem_type());
  } else {
    propagateShapeAndTypeFromFirstInput(ctx);
  }
}

void OptionalHasElementShapeInference_opset15(InferenceContext& ctx) {
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

void OptionalGetElementShapeInference_opset15(InferenceContext& ctx) {
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

} // namespace ONNX_NAMESPACE::defs::optional::utils
