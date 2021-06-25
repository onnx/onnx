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

class SymbolTableImpl : public SymbolTable {
 public:
  SymbolTableImpl() : index_(0){}

  void addFromGraph(const GraphProto& g) {
    AddExistingSymbolicDims(g.input());
    AddExistingSymbolicDims(g.output());
    AddExistingSymbolicDims(g.value_info());
  }
  // Creates a new unique symbol with the given prefix and adds it to the SymbolTable
  // Returns the newly created symbol
  std::string createNew(const std::string& symbol_prefix="unk__") {
    std::string newSymbol;
    do {
      newSymbol = symbol_prefix + std::to_string(index_++);
    } while(existing_symbols.count(newSymbol) > 0);
    existing_symbols.insert(newSymbol);
    return newSymbol;
  }

 private:
  unsigned int index_;
  std::unordered_set<std::string> existing_symbols;

  // TypeProto_Tensor or TypeProto_SparseTensor
  template <typename TensorTypeProto>
  void AddExistingSymbolicDims(const TensorTypeProto& tensorType) {
    if (tensorType.has_shape()) {
      for (int i = 0; i < tensorType.shape().dim_size(); ++i) {
        if (tensorType.shape().dim(i).has_dim_param()) {
          existing_symbols.insert(tensorType.shape().dim(i).dim_param());
        }
      }
    }
  }

  void AddExistingSymbolicDims(const TypeProto& typeProto) {
      const auto val_case = typeProto.value_case();
      switch (val_case) {
        case TypeProto::kTensorType:
          AddExistingSymbolicDims(typeProto.tensor_type());
          break;
        case TypeProto::kSparseTensorType:
          AddExistingSymbolicDims(typeProto.sparse_tensor_type());
          break;
        case TypeProto::kSequenceType:
          AddExistingSymbolicDims(typeProto.sequence_type().elem_type());
          break;
        default:
          break;
    }
  }

  void AddExistingSymbolicDims(const google::protobuf::RepeatedPtrField<ValueInfoProto>& protos) {
    for (const auto& proto : protos) {
      AddExistingSymbolicDims(proto.type());
    }
  }
};

struct GraphInferenceContext {
  GraphInferenceContext(
      const std::unordered_map<std::string, TypeProto*>&
          outer_scope_value_types_by_name_in,
      const std::unordered_map<std::string, int> opset_imports_in,
      SymbolTableImpl& symbolTable_in,
      const ISchemaRegistry* schema_registry_in = OpSchemaRegistry::Instance())
      : outer_scope_value_types_by_name{&outer_scope_value_types_by_name_in},
        opset_imports{opset_imports_in},
        schema_registry{schema_registry_in},
        symbolTable{symbolTable_in} {}


  const std::unordered_map<std::string, TypeProto*>*
      outer_scope_value_types_by_name;
  const std::unordered_map<std::string, int> opset_imports;
  const ISchemaRegistry* schema_registry;
  SymbolTableImpl& symbolTable;
};

class GraphInferencerImpl : public GraphInferencer {
 public:
  GraphInferencerImpl(GraphProto& g, GraphInferenceContext& context)
      : g_{&g}, context_{&context} {}

  std::vector<const TypeProto*> doInferencing(
      const std::vector<const TypeProto*>& inputTypes,
      const std::vector<const TensorProto*>& inputData) override;

  SymbolTableImpl& getSymbolTable() {
    return context_->symbolTable;
  }

 private:
  GraphProto* g_;
  GraphInferenceContext* context_;
};

struct InferenceContextImpl : public InferenceContext {
  InferenceContextImpl(
      NodeProto& n,
      const std::unordered_map<std::string, TypeProto*>& valueTypesByName,
      const std::unordered_map<std::string, const TensorProto*>&
          inputDataByName,
      const std::unordered_map<std::string, const SparseTensorProto*>& 
          inputSparseDataByName,
      GraphInferenceContext* graphInferenceContext = nullptr)
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
  GraphInferenceContext* graphInferenceContext_;

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

template <typename TensorTypeProto>
void generateSymbolicShape(TensorTypeProto* inferredType, SymbolTableImpl& symbolTable);

void materializeSymbolicShape(TypeProto* inferredType, SymbolTableImpl& symbolTable);

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
    SymbolTableImpl &symbolTable);

void InferShapeForFunctionNode(
    const FunctionProto* func,
    const std::unordered_map<std::string, int>& func_opset_imports,
    const ISchemaRegistry* schema_registry,
    InferenceContext& ctx,
    SymbolTableImpl &symbolTable);

std::string getErrorWithNodeInfo(NodeProto n, std::runtime_error err);

void traverseGraphsToAddExistingSymbols(const GraphProto& g, SymbolTableImpl& symbolTable);

} // namespace shape_inference
} // namespace ONNX_NAMESPACE
