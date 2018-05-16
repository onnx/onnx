#pragma once

#include "onnx/common/status.h"
#include "onnx/defs/data_type_utils.h"
#include "onnx/proto_utils.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {

using namespace Common;

struct InferenceContext {
  virtual const AttributeProto* getAttribute(const std::string& name) const = 0;
  virtual size_t getNumInputs() const = 0;
  virtual const TypeProto* getInputType(size_t index) const = 0;
  virtual size_t getNumOutputs() const = 0;
  virtual TypeProto* getOutputType(size_t index) = 0;
  virtual ~InferenceContext() {}
};

typedef Common::Status (*InferenceFunction)(InferenceContext&);

// This no-op inference function is used for operators without an
// inference implementation.
inline void dummyInferenceFunction(InferenceContext&) { };

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

inline TensorShapeProto::Dimension operator*(TensorShapeProto::Dimension dim1, TensorShapeProto::Dimension dim2) {
  TensorShapeProto::Dimension result;
  if (dim1.has_dim_value() && dim2.has_dim_value()) {
    result.set_dim_value(dim1.dim_value() * dim2.dim_value());
  }	else if (dim1.has_dim_value() && (dim1.dim_value() == 1)) {
    return dim2;
  }	else if (dim2.has_dim_value() && (dim2.dim_value() == 1)) {
    return dim1;
  }
  return result;
}

inline TensorShapeProto::Dimension operator*(TensorShapeProto::Dimension dim1, int64_t dim2) {
  TensorShapeProto::Dimension result;
  if (dim1.has_dim_value()) {
	  result.set_dim_value(dim1.dim_value() * dim2);
  }	else if (dim2 == 1) {
    return dim1;
  }
  return result;
}

inline TensorShapeProto::Dimension operator/(TensorShapeProto::Dimension dim1, int64_t dim2) {
  TensorShapeProto::Dimension result;
  if (dim1.has_dim_value()) {
    result.set_dim_value(dim1.dim_value() / dim2);
  }	else if (dim2 == 1) {
    return dim1;
  }
  return result;
}

inline TensorShapeProto::Dimension multiplyDims(const TensorShapeProto& shape, int from, int upto_exclusive) {
  TensorShapeProto::Dimension dim;
  dim.set_dim_value(1);
  for (int i = from; i < upto_exclusive; ++i) {
    dim = dim * shape.dim(i);
  }
  return dim;
}

inline Status propagateElemTypeFromInputToOutput(
    InferenceContext& ctx,
    size_t inputIndex,
    size_t outputIndex) {
  auto input_type = ctx.getInputType(inputIndex);
  if (nullptr == input_type
	  || input_type->value_case() != TypeProto::kTensorType
	  || input_type->tensor_type().elem_type() == TensorProto::UNDEFINED) {
	  return Status(INFERENCE, INVALID_PROTOBUF, MakeString("Type propagation failed, input type should be tensor-type."));
  }

  auto output_type = ctx.getOutputType(outputIndex);
  if (output_type->value_case() != TypeProto::kTensorType &&
      output_type->value_case() != TypeProto::VALUE_NOT_SET) {
	  return Status(INFERENCE, INVALID_PROTOBUF, MakeString("Type propagation failed, output type should be tensor-type."));
  }

  output_type->mutable_tensor_type()->set_elem_type(
	  input_type->tensor_type().elem_type());
  return Status::OK();
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

inline const TensorShapeProto& getInputShape(InferenceContext& ctx, size_t n) {
  return ctx.getInputType(n)->tensor_type().shape();
}

inline Status appendSingleDimCopiedFromInputTypeToOutputType(
    InferenceContext& ctx,
    size_t inputIndex,
    size_t outputIndex,
    size_t fromDimIndex) {
  auto output_type = ctx.getOutputType(outputIndex);
  auto input_type = ctx.getInputType(inputIndex);
  if (TypeProto::kTensorType != output_type->value_case() ||
      TypeProto::kTensorType != input_type->value_case()) {
    return Status(INFERENCE, INVALID_PROTOBUF, "input and output should be tensor-type.");
  }
  auto* dim = ctx.getOutputType(outputIndex)
                  ->mutable_tensor_type()
                  ->mutable_shape()
                  ->add_dim();
  *dim = input_type->tensor_type().shape().dim(static_cast<int>(fromDimIndex));
  return Status::OK();
}

inline Status propagateShapeFromInputToOutput(
    InferenceContext& ctx,
    size_t inputIndex,
    size_t outputIndex) {
  auto output_type = ctx.getOutputType(outputIndex);
  auto input_type = ctx.getInputType(inputIndex);
  if (TypeProto::kTensorType != input_type->value_case() ||
      TypeProto::kTensorType != output_type->value_case()) {
	return Status(INFERENCE, INVALID_PROTOBUF, MakeString("Shape inference failed. Both input and output should be tensor-type."));
  }

  *ctx.getOutputType(outputIndex)->mutable_tensor_type()->mutable_shape() =
      ctx.getInputType(inputIndex)->tensor_type().shape();
  return Status::OK();
}

inline Status propagateShapeAndTypeFromFirstInput(InferenceContext& ctx) {
  ONNX_RETURN_IF_ERROR(propagateElemTypeFromInputToOutput(ctx, 0, 0));
  if (hasNInputShapes(ctx, 1)) {
    return propagateShapeFromInputToOutput(ctx, 0, 0);
  }
  return Status::OK();
}

inline Status updateOutputElemType(
    InferenceContext& ctx,
    size_t outputIndex,
    TensorProto_DataType elemType) {
  auto output_type = ctx.getOutputType(outputIndex);
  if ((output_type != nullptr) &&
      (output_type->value_case() == TypeProto::kTensorType ||
       output_type->value_case() == TypeProto::VALUE_NOT_SET)) {
    output_type->mutable_tensor_type()->set_elem_type(elemType);
	return Status::OK();
  }
  return Status(INFERENCE, INVALID_PROTOBUF, "Update output elem type failed. output_type should be a tensor-type.");
}

// Infer type of an output from the value of a specified attribute, which is expected
// to have a valid value representing a TensorProto_DataType.
inline Status propagateElemTypeFromAttributeToOutput(
    InferenceContext& ctx,
    const std::string& attributeName,
    size_t outputIndex,
    TensorProto_DataType default_value = TensorProto::UNDEFINED) {
  auto attr_proto = ctx.getAttribute(attributeName);
  if (nullptr == attr_proto) { // attribute not present
    if (default_value != TensorProto::UNDEFINED) {
      return updateOutputElemType(ctx, outputIndex, default_value);
	}
    return Status(INFERENCE, INVALID_PROTOBUF, MakeString("Attribute: ", attributeName, " does not exist."));
  }
  if (!attr_proto->has_i()) {
	  // attribute not of right type
	  return Status(INFERENCE, INVALID_PROTOBUF, MakeString("Attribute: ", attributeName, " should be integer type."));
  }
  auto attr_value = attr_proto->i();
  auto elem_type = static_cast<TensorProto_DataType>(attr_value);
  if (!TensorProto_DataType_IsValid(elem_type)) {
	  return Status(INFERENCE, INVALID_PROTOBUF, MakeString("Attribute: ", attributeName, " is not specifying a valid tensor element type."));
  }
  return updateOutputElemType(ctx, outputIndex, elem_type);
}

inline TensorShapeProto* getOutputShape(InferenceContext& ctx, size_t n) {
  auto output_type = ctx.getOutputType(n);
  if ((output_type != nullptr) &&
      (output_type->value_case() == TypeProto::kTensorType ||
       output_type->value_case() == TypeProto::VALUE_NOT_SET)) {
      return output_type->mutable_tensor_type()->mutable_shape();
  }
  else {
	  return nullptr;
  }
}

inline Status updateOutputShape(
    InferenceContext& ctx,
    size_t outputIndex,
    const TensorShapeProto& shape) {
  auto* output_shape = getOutputShape(ctx, outputIndex);
  if (output_shape == nullptr) {
	return Status(INFERENCE, INVALID_PROTOBUF, MakeString("Output: ", outputIndex, " should be tensor-type or no-set."));
  }
  *output_shape = shape;
  return Status::OK();  
}

inline Status updateOutputShape(
    InferenceContext& ctx,
    size_t outputIndex,
    const TensorProto& tensorProto) {
  auto* output_shape = getOutputShape(ctx, outputIndex);
  if (output_shape == nullptr) {
	  return Status(INFERENCE, INVALID_PROTOBUF, MakeString("Output: ", outputIndex, " should be tensor-type or no-set."));
  }
  for (auto d : tensorProto.dims()) {
    auto* dim = output_shape->add_dim();
    dim->set_dim_value(d);
  }
  return Status::OK();
}

inline Status updateOutputShape(
	InferenceContext& ctx,
	size_t outputIndex,
	std::initializer_list<TensorShapeProto::Dimension> dims) {
  auto* output_shape = getOutputShape(ctx, outputIndex);
  if (output_shape == nullptr) {
	  return Status(INFERENCE, INVALID_PROTOBUF, MakeString("Output: ", outputIndex, " should be tensor-type or no-set."));
  }
  for (auto& d : dims) {
    auto* dim = output_shape->add_dim();
    *dim = d;
  }
  return Status::OK();
}

// Infer shape of an output from the value of a specified attribute, which is expected
// to be a list of integers specifying a valid shape.
inline Status propagateShapeFromAttributeToOutput(
    InferenceContext& ctx,
    const std::string& attributeName,
    size_t outputIndex) {
  auto attr_proto = ctx.getAttribute(attributeName);
  auto& int_list = attr_proto->ints();
  TensorShapeProto shape;
  for (auto dim_size : int_list) {
	if (dim_size < 0) {
	  return Status(INFERENCE, INVALID_PROTOBUF, MakeString("dim_size specified by attribute: ", attributeName, " is negative (", dim_size, ")."));
	}
    shape.add_dim()->set_dim_value(dim_size);
  }

  return updateOutputShape(ctx, outputIndex, shape);
}

} // namespace ONNX_NAMESPACE
