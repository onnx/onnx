/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/proto_utils.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {
namespace shape_inference {

struct GraphInferenceContext {
  GraphInferenceContext(
      const std::unordered_map<std::string, TypeProto*>&
          outer_scope_value_types_by_name_in,
      const std::unordered_map<std::string, int> opset_imports_in,
      SymbolTable& symbolTable_in,
      const ISchemaRegistry* schema_registry_in = OpSchemaRegistry::Instance())
      : outer_scope_value_types_by_name{&outer_scope_value_types_by_name_in},
        opset_imports{opset_imports_in},
        schema_registry{schema_registry_in},
        symbolTable{symbolTable_in} {}


  const std::unordered_map<std::string, TypeProto*>*
      outer_scope_value_types_by_name;
  const std::unordered_map<std::string, int> opset_imports;
  const ISchemaRegistry* schema_registry;
  SymbolTable& symbolTable;
};

class GraphInferencerImpl : public GraphInferencer {
 public:
  GraphInferencerImpl(GraphProto& g, const GraphInferenceContext& context)
      : g_{&g}, context_{&context} {}

  std::vector<const TypeProto*> doInferencing(
      const std::vector<const TypeProto*>& inputTypes,
      const std::vector<const TensorProto*>& inputData) override;

  SymbolTable& getSymbolTable() {
    auto* contextForSymbol = const_cast<GraphInferenceContext*>(context_);
    return contextForSymbol->symbolTable;
  }

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
      const std::unordered_map<std::string, const SparseTensorProto*>& 
          inputSparseDataByName,
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
        allInputSparseData_.push_back(nullptr);
      } else {
        allInputData_.push_back(nullptr);
        const auto inputSparseDataIter = inputSparseDataByName.find(input);
        if (inputSparseDataIter != inputSparseDataByName.cend()) {
          allInputSparseData_.push_back(inputSparseDataIter->second);
        } else {
          allInputSparseData_.push_back(nullptr);
        }
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
      ONNX_THROW("input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds");
    }
    return allInputTypes_[index];
  }

  const TensorProto* getInputData(size_t index) const override {
    if (index >= allInputData_.size()) {
      ONNX_THROW("input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds");
    }
    return allInputData_[index];
  }

  const SparseTensorProto* getInputSparseData(size_t index) const override {
    if (index >= allInputSparseData_.size()) {
      ONNX_THROW("input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds");
    }
    return allInputSparseData_[index];
  }

  size_t getNumOutputs() const override {
    return allOutputTypes_.size();
  }

  TypeProto* getOutputType(size_t index) override {
    if (index >= allOutputTypes_.size()) {
      ONNX_THROW("output " + ONNX_NAMESPACE::to_string(index) + " is out of bounds");
    }
    return &allOutputTypes_[index];
  }

  GraphInferencer* getGraphAttributeInferencer(
      const std::string& attr_name) override {
    if (!graphInferenceContext_) {
      fail_type_inference(
          "GraphProto attribute inferencing is not enabled in this InferenceContextImpl instance.");
    }

    GraphInferencer* inferencer = nullptr;

    auto entry = graphAttributeInferencers_.find(attr_name);
    if (entry == graphAttributeInferencers_.cend()) {
      // create GraphInferencer instance
      auto attrNameToGraphProto = graphProtoAttributesByName_.find(attr_name);
      if (attrNameToGraphProto == graphProtoAttributesByName_.cend()) {
        fail_type_inference(
            "Attribute ", attr_name, " does not contain a graph.");
      }

      std::unique_ptr<GraphInferencer> new_inferencer{new GraphInferencerImpl(
          *attrNameToGraphProto->second, *graphInferenceContext_)};

      inferencer = new_inferencer.get();
      graphAttributeInferencers_.emplace(attr_name, std::move(new_inferencer));
    } else {
      inferencer = entry->second.get();
    }

    return inferencer;
  }
  std::vector<const TensorProto*> allInputData_;
  std::vector<const SparseTensorProto*> allInputSparseData_;
  std::unordered_map<std::string, const AttributeProto*> attributesByName_;
  std::unordered_map<std::string, GraphProto*> graphProtoAttributesByName_;
  std::vector<const TypeProto*> allInputTypes_;
  std::vector<TypeProto> allOutputTypes_;
  const GraphInferenceContext* graphInferenceContext_;

  // mutable as internal cache of GraphInferencer instances
  mutable std::unordered_map<std::string, std::unique_ptr<GraphInferencer>>
      graphAttributeInferencers_;
};

void checkShapesAndTypes(
    const TypeProto_Sequence& inferredType,
    const TypeProto_Sequence& existingType);

void checkShapesAndTypes(
    const TypeProto& inferredType,
    const TypeProto& existingType);

template <typename T>
void generateSymbolicShape(T* inferredType, SymbolTable& symbolTable);

void materializeSymbolicShape(TypeProto* inferredType, SymbolTable& symbolTable);

void mergeShapesAndTypes(
    const TypeProto_Tensor& inferredType,
    TypeProto_Tensor* existingType);

void mergeShapesAndTypes(
    const TypeProto_SparseTensor& inferredType,
    TypeProto_SparseTensor* existingType);

void mergeShapesAndTypes(
    const TypeProto_Sequence& inferredType,
    TypeProto_Tensor* existingType);

void mergeShapesAndTypes(
    const TypeProto& inferredType,
    TypeProto* existingType);

void InferShapes(
    ModelProto& m,
    const bool check_type = false,
    const ISchemaRegistry* schema_registry = OpSchemaRegistry::Instance(),
    const int error_mode = 0
    );

void InferShapes(
    GraphProto* g,
    const std::unordered_map<std::string, int>& opset_imports,
    const bool check_type = false,
    const ISchemaRegistry* schema_registry = OpSchemaRegistry::Instance(),
    const int error_mode = 0
    );

void InferShapes(
    const std::string& model_path,
    const bool check_type = false,
    const std::string& save_path = "",
    const ISchemaRegistry* schema_registry = OpSchemaRegistry::Instance(),
    const int error_mode = 0
    );

void InferShapeForFunctionNode(
    const FunctionProto* func,
    const ISchemaRegistry* schema_registry,
    InferenceContext& ctx,
    SymbolTable &symbolTable);

void InferShapeForFunctionNode(
    const FunctionProto* func,
    const std::unordered_map<std::string, int>& func_opset_imports,
    const ISchemaRegistry* schema_registry,
    InferenceContext& ctx,
    SymbolTable &symbolTable);

std::string getErrorWithNodeInfo(NodeProto n, std::runtime_error err);

} // namespace shape_inference
} // namespace ONNX_NAMESPACE
