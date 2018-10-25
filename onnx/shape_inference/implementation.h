#pragma once

#include "onnx/common/stl_backports.h"
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/proto_utils.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {
namespace shape_inference {

struct GraphInferenceContext {
  const std::unordered_map<std::string, int> opset_imports;
  const ISchemaRegistry* schema_registry = OpSchemaRegistry::Instance();
  const IFunctionBuilderRegistry* func_registry =
      &FunctionBuilderRegistry::OnnxInstance();
};

class GraphInfererImpl : public GraphInferencer {
 public:
  GraphInfererImpl(GraphProto& g, const GraphInferenceContext& context)
      : g_{&g}, context_{&context} {}

  std::vector<const TypeProto*> doInferencing(
      const std::vector<const TypeProto*>& inferredInputTypes) override;

 private:
  GraphProto* g_;
  const GraphInferenceContext* context_;
};

struct InferenceContextImpl : public InferenceContext {
  InferenceContextImpl(
      NodeProto& n,
      const std::unordered_map<std::string, TypeProto*>& valueTypesByName,
      const std::unordered_map<std::string, const TensorProto*>&
          inputDataByName,
      const GraphInferenceContext* graphInferenceContext = nullptr)
      : graphInferenceContext_{graphInferenceContext} {
    for (auto& attr : *n.mutable_attribute()) {
      attributesByName_[attr.name()] = &attr;
      if (attr.has_g()) {
        // need a mutable GraphProto to run inferencing on this attribute
        graphProtoAttributesByName_[attr.name()] = attr.mutable_g();
      }
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

  const GraphInferencer* getGraphAttributeInferer(
      const std::string& attr_name) const override {
    if (!graphInferenceContext_) {
      fail_type_inference(
          "GraphProto attribute inferencing is not enabled in this InferenceContextImpl instance.");
    }

    GraphInferencer* inferer = nullptr;

    auto entry = graphAttributeInferers_.find(attr_name);
    if (entry == graphAttributeInferers_.cend()) {
      // create GraphInferencer instance
      auto attrNameToGraphProto = graphProtoAttributesByName_.find(attr_name);
      if (attrNameToGraphProto == graphProtoAttributesByName_.cend()) {
        fail_type_inference(
            "Attribute ", attr_name, " does not contain a graph.");
      }

      std::unique_ptr<GraphInferencer> new_inferer =
          make_unique<GraphInfererImpl>(
              *attrNameToGraphProto->second, *graphInferenceContext_);
      inferer = new_inferer.get();
      graphAttributeInferers_.insert({attr_name, std::move(new_inferer)});
    } else {
      inferer = entry->second.get();
    }

    return inferer;
  }

  std::vector<const TensorProto*> allInputData_;
  std::unordered_map<std::string, const AttributeProto*> attributesByName_;
  std::unordered_map<std::string, GraphProto*> graphProtoAttributesByName_;
  std::vector<const TypeProto*> allInputTypes_;
  std::vector<TypeProto> allOutputTypes_;
  const GraphInferenceContext* graphInferenceContext_;

  // mutable as internal cache of GraphInferencer instances
  mutable std::unordered_map<std::string, std::unique_ptr<GraphInferencer>>
      graphAttributeInferers_;
};

void checkShapesAndTypes(
    const TypeProto_Tensor& inferredType,
    const TypeProto_Tensor& existingType);

void mergeShapesAndTypes(
    const TypeProto_Tensor& inferredType,
    TypeProto_Tensor* existingType);

void InferShapes(
    ModelProto& m,
    const ISchemaRegistry* schema_registry = OpSchemaRegistry::Instance(),
    const IFunctionBuilderRegistry* func_registry =
        &FunctionBuilderRegistry::OnnxInstance());

void InferShapes(
    GraphProto* g,
    const std::unordered_map<std::string, int>& opset_imports,
    const ISchemaRegistry* schema_registry = OpSchemaRegistry::Instance(),
    const IFunctionBuilderRegistry* func_registry =
        &FunctionBuilderRegistry::OnnxInstance());

void InferShapeForFunctionNode(
    const FunctionProto& func,
    const ISchemaRegistry* schema_registry,
    InferenceContext& ctx);

} // namespace shape_inference
} // namespace ONNX_NAMESPACE
