#pragma once

#include "data_type_utils.h"

namespace ONNX_NAMESPACE {

struct InferenceContext {
  virtual const AttributeProto* getAttribute(const std::string& name) const = 0;
  virtual size_t getNumInputs() const = 0;
  virtual const TypeProto_Tensor* getInputType(size_t index) const = 0;
  virtual size_t getNumOutputs() const = 0;
  virtual TypeProto_Tensor* getOutputType(size_t index) = 0;
  virtual ~InferenceContext() {}
};

typedef void (*InferenceFunction)(InferenceContext&);

template<typename T> inline google::protobuf::RepeatedField<T> retrieveValues(const AttributeProto& attr);
template<> inline google::protobuf::RepeatedField<int64_t> retrieveValues(const AttributeProto& attr) { return attr.ints(); }

template<typename T>
inline bool getRepeatedAttribute(InferenceContext& ctx,
                                 std::string attr_name,
                                 std::vector<T>& values) {
  const auto* attr = ctx.getAttribute(attr_name);
  if (attr) {
    for (const auto& value : retrieveValues<T>(*attr)) {
      values.push_back(value);
    }
    return true;
  } else {
    return false;
  }

}

inline void propagateElemTypeFromInputToOutput(InferenceContext& ctx, size_t inputIndex, size_t outputIndex) {
  ctx.getOutputType(outputIndex)->set_elem_type(ctx.getInputType(inputIndex)->elem_type());
}

inline void appendSingleDimCopiedFromInputTypeToOutputType(InferenceContext& ctx, size_t inputIndex, size_t outputIndex, size_t fromDimIndex) {
  auto* dim = ctx.getOutputType(outputIndex)->mutable_shape()->add_dim();
  *dim = ctx.getInputType(inputIndex)->shape().dim(fromDimIndex);
}

} // namespace ONNX_NAMESPACE
