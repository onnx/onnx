#pragma once

#include "onnx/defs/data_type_utils.h"
#include "onnx/proto_utils.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {

struct InferenceContext {
  virtual const AttributeProto* getAttribute(const std::string& name) const = 0;
  virtual size_t getNumInputs() const = 0;
  virtual const TypeProto* getInputType(size_t index) const = 0;
  virtual size_t getNumOutputs() const = 0;
  virtual TypeProto* getOutputType(size_t index) = 0;
  virtual ~InferenceContext() {}
};

typedef void (*InferenceFunction)(InferenceContext&);

template <typename T>
inline bool getRepeatedAttribute(
    InferenceContext& ctx,
    std::string attr_name,
    std::vector<T>& values) {
  const auto* attr = ctx.getAttribute(attr_name);
  if (attr) {
    values = RetrieveValues<T>(*attr);
    return true;
  } else {
    return false;
  }
}

inline bool
hasExactlyNInputTypes(InferenceContext& ctx, int n, const std::string& opname) {
  if (static_cast<int>(ctx.getNumInputs()) != n) {
    throw std::runtime_error(opname + " has wrong number of inputs");
  }
  for (int i = 0; i < n; i++) {
    if (!ctx.getInputType(i)) {
      return false;
    }
  }
  return true;
}

inline void propagateElemTypeFromInputToOutput(
    InferenceContext& ctx,
    size_t inputIndex,
    size_t outputIndex) {
  auto input_type = ctx.getInputType(inputIndex);
  if (nullptr == input_type ||
      input_type->value_case() != TypeProto::kTensorType) {
    return;
  }
  auto output_type = ctx.getOutputType(outputIndex);
  if (output_type->value_case() == TypeProto::kTensorType ||
      output_type->value_case() == TypeProto::VALUE_NOT_SET) {
    output_type->mutable_tensor_type()->set_elem_type(
        input_type->tensor_type().elem_type());
  }
}

inline bool hasNInputShapes(InferenceContext& ctx, int n) {
  if (static_cast<int>(ctx.getNumInputs()) < n) {
    throw std::runtime_error("operator has too few inputs");
  }
  for (int i = 0; i < n; i++) {
    auto input_type = ctx.getInputType(i);
    if (nullptr == input_type || !input_type->has_tensor_type() ||
        !input_type->tensor_type().has_shape()) {
      return false;
    }
  }
  return true;
}

inline void appendSingleDimCopiedFromInputTypeToOutputType(
    InferenceContext& ctx,
    size_t inputIndex,
    size_t outputIndex,
    size_t fromDimIndex) {
  auto output_type = ctx.getOutputType(outputIndex);
  auto input_type = ctx.getInputType(inputIndex);
  if (TypeProto::kTensorType != output_type->value_case() ||
      TypeProto::kTensorType != input_type->value_case()) {
    return;
  }
  auto* dim = ctx.getOutputType(outputIndex)
                  ->mutable_tensor_type()
                  ->mutable_shape()
                  ->add_dim();
  *dim = input_type->tensor_type().shape().dim(static_cast<int>(fromDimIndex));
}

inline void propagateShapeFromInputToOutput(
    InferenceContext& ctx,
    size_t inputIndex,
    size_t outputIndex) {
  auto output_type = ctx.getOutputType(outputIndex);
  auto input_type = ctx.getInputType(inputIndex);
  if (TypeProto::kTensorType != input_type->value_case() ||
      TypeProto::kTensorType != output_type->value_case()) {
    throw std::runtime_error(
        "zhangke: " +
        ONNX_NAMESPACE::to_string(
            ctx.getInputType(inputIndex)->tensor_type().shape().dim_size()));
    return;
  }

  *ctx.getOutputType(outputIndex)->mutable_tensor_type()->mutable_shape() =
      ctx.getInputType(inputIndex)->tensor_type().shape();
}

} // namespace ONNX_NAMESPACE
