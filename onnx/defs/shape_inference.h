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

inline int64_t getAttribute(InferenceContext& ctx, const std::string& attributeName, int64_t defaultValue) {
  auto attr_proto = ctx.getAttribute(attributeName);
  if ((nullptr != attr_proto) && attr_proto->has_i())
    return attr_proto->i();
  return defaultValue;
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

inline bool hasInputShape(InferenceContext& ctx, int n) {
  return ctx.getNumInputs() > static_cast<size_t>(n) &&
    ctx.getInputType(n) &&
    ctx.getInputType(n)->has_tensor_type() &&
    ctx.getInputType(n)->tensor_type().has_shape();
}

inline bool hasNInputShapes(InferenceContext& ctx, int n) {
  for (int i = 0; i < n; i++) {
    if (!hasInputShape(ctx, i)) {
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
        ONNX_NAMESPACE::to_string(
            ctx.getInputType(inputIndex)->tensor_type().shape().dim_size()));
    return;
  }

  *ctx.getOutputType(outputIndex)->mutable_tensor_type()->mutable_shape() =
      ctx.getInputType(inputIndex)->tensor_type().shape();
}

inline void propagateShapeAndTypeFromFirstInput(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  propagateShapeFromInputToOutput(ctx, 0, 0);
}

inline void updateOutputElemType(
    InferenceContext& ctx,
    size_t outputIndex,
    TensorProto_DataType elemType) {
  auto output_type = ctx.getOutputType(outputIndex);
  if ((output_type != nullptr) &&
      (output_type->value_case() == TypeProto::kTensorType ||
       output_type->value_case() == TypeProto::VALUE_NOT_SET)) {
    output_type->mutable_tensor_type()->set_elem_type(elemType);
  }
}

// Infer type of an output from the value of a specified attribute, which is expected
// to have a valid value representing a TensorProto_DataType.
inline void propagateElemTypeFromAttributeToOutput(
    InferenceContext& ctx,
    const std::string& attributeName,
    size_t outputIndex,
    TensorProto_DataType default_value = TensorProto::UNDEFINED) {
  auto attr_proto = ctx.getAttribute(attributeName);
  if (nullptr == attr_proto) { // attribute not present
    if (default_value != TensorProto::UNDEFINED)
      updateOutputElemType(ctx, outputIndex, default_value);
    return;
  }  
  if (!attr_proto->has_i()) return; // attribute not of right type
  auto attr_value = attr_proto->i();
  auto elem_type = static_cast<TensorProto_DataType>(attr_value);
  if (!TensorProto_DataType_IsValid(elem_type)) return; // out-of-range attribute value
  updateOutputElemType(ctx, outputIndex, elem_type);
}

inline void updateOutputShape(
    InferenceContext& ctx,
    size_t outputIndex,
    const TensorShapeProto& shape) {
  auto output_type = ctx.getOutputType(outputIndex);
  if ((output_type != nullptr) &&
      (output_type->value_case() == TypeProto::kTensorType ||
       output_type->value_case() == TypeProto::VALUE_NOT_SET)) {
    *output_type->mutable_tensor_type()->mutable_shape() = shape;
  }
}

inline void updateOutputShape(
    InferenceContext& ctx,
    size_t outputIndex,
    const TensorProto& tensorProto) {
  auto output_type = ctx.getOutputType(outputIndex);
  if ((output_type != nullptr) &&
      (output_type->value_case() == TypeProto::kTensorType ||
       output_type->value_case() == TypeProto::VALUE_NOT_SET)) {
    auto* output_shape = output_type->mutable_tensor_type()->mutable_shape();
    for (auto d : tensorProto.dims()) {
      auto* dim = output_shape->add_dim();
      dim->set_dim_value(d);
    }
  }
}

// Infer shape of an output from the value of a specified attribute, which is expected
// to be a list of integers specifying a valid shape.
inline void propagateShapeFromAttributeToOutput(
    InferenceContext& ctx,
    const std::string& attributeName,
    size_t outputIndex) {

  auto attr_proto = ctx.getAttribute(attributeName);
  if (nullptr == attr_proto) return; // attribute not present
  if (!attr_proto->has_type()) return; // invalid attribute: has no type
  if (attr_proto->type() != AttributeProto_AttributeType_INTS) return; // invalid attribute type
  auto& int_list = attr_proto->ints();
  TensorShapeProto shape;
  for (auto dim_size : int_list) {
    if (dim_size < 0) return; // invalid dimension size
    shape.add_dim()->set_dim_value(dim_size);
  }

  updateOutputShape(ctx, outputIndex, shape);
}

} // namespace ONNX_NAMESPACE
