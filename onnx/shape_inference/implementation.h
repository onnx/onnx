#pragma once

#include "onnx/defs/schema.h"
#include "onnx/proto_utils.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {
namespace shape_inference {

struct InferenceContextImpl : public InferenceContext {
  InferenceContextImpl(
      const NodeProto& n,
      const std::unordered_map<std::string, TypeProto*>& valueTypesByName,
      const std::unordered_map<std::string, const TensorProto*>& inputDataByName) {
    for (const auto& attr : n.attribute()) {
      attributesByName_[attr.name()] = &attr;
    }

    for (const auto& input : n.input()) {
      auto valueTypesIter = valueTypesByName.find(input);
      if (valueTypesIter != valueTypesByName.end()) {
        allInputTypes_.push_back(valueTypesIter->second);
      } else {
        allInputTypes_.push_back(nullptr);
      }

      const auto inputDataIter = inputDataByName.find(input);
      if (inputDataIter != inputDataByName.cend()) {
        allInputData_.push_back(inputDataIter->second);
      } else {
        allInputData_.push_back(nullptr);
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

  const TensorProto* getInputData(size_t index) const override {
    if (index >= allInputData_.size()) {
      throw std::runtime_error(
          "input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds");
    }
    return allInputData_[index];
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
  std::vector<const TensorProto*> allInputData_;
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
