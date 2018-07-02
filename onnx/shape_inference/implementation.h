#pragma once

#include "onnx/defs/schema.h"
#include "onnx/proto_utils.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {
namespace shape_inference {

struct InferenceContextImpl : public InferenceContext {
  InferenceContextImpl(
      const NodeProto& n,
      const std::unordered_map<std::string, TypeProto*>& valueTypesByName) {
    for (const auto& attr : n.attribute()) {
      attributesByName_[attr.name()] = &attr;
    }

    for (const auto& input : n.input()) {
      auto iter = valueTypesByName.find(input);
      if (iter != valueTypesByName.end()) {
        allInputTypes_.push_back(iter->second);
      } else {
        allInputTypes_.push_back(nullptr);
      }
    }

    allOutputTypes_.resize(n.output_size());
  }

  const AttributeProto* getAttribute(const std::string& name) const override {
    auto iter = attributesByName_.find(name);
    if (iter == attributesByName_.end()) {
      return nullptr;
    } else {
      return iter->second;
    }
  }
  size_t getNumInputs() const override {
    return allInputTypes_.size();
  }

  const TypeProto* getInputType(size_t index) const override {
    if (index >= allInputTypes_.size()) {
      throw std::runtime_error(
          "input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds");
    }
    return allInputTypes_[index];
  }

  size_t getNumOutputs() const override {
    return allOutputTypes_.size();
  }

  TypeProto* getOutputType(size_t index) override {
    if (index >= allOutputTypes_.size()) {
      throw std::runtime_error(
          "output " + ONNX_NAMESPACE::to_string(index) + " is out of bounds");
    }
    return &allOutputTypes_[index];
  }
  std::unordered_map<std::string, const AttributeProto*> attributesByName_;
  std::vector<const TypeProto*> allInputTypes_;
  std::vector<TypeProto> allOutputTypes_;
};

void checkShapesAndTypes(
    const TypeProto_Tensor& inferredType,
    const TypeProto_Tensor& existingType);

void mergeShapesAndTypes(
    const TypeProto_Tensor& inferredType,
    TypeProto_Tensor* existingType);

void InferShapes(
    ModelProto& m,
    const ISchemaRegistry* schema_registry = OpSchemaRegistry::Instance());

} // namespace shape_inference
} // namespace ONNX_NAMESPACE
