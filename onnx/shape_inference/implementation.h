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
    const TypeProto_Tensor& existingType) {
  if (inferredType.elem_type() != TensorProto::UNDEFINED &&
      existingType.elem_type() != TensorProto::UNDEFINED &&
      existingType.elem_type() != inferredType.elem_type()) {
    std::stringstream ss;
    ss << "Inferred elem type differs from existing elem type: ("
       << inferredType.elem_type() << ") vs (" << existingType.elem_type()
       << ")";
    throw std::runtime_error(ss.str());
  }

  if (!inferredType.has_shape() || !existingType.has_shape()) {
    return;
  }

  if (inferredType.shape().dim_size() != existingType.shape().dim_size()) {
    std::stringstream ss;
    ss << "Inferred shape and existing shape differ in rank: ("
       << inferredType.shape().dim_size() << ") vs ("
       << existingType.shape().dim_size() << ")";
    throw std::runtime_error(ss.str());
  }

  for (int i = 0; i < inferredType.shape().dim_size(); ++i) {
    const auto& inferredDim = inferredType.shape().dim(i);
    const auto& existingDim = existingType.shape().dim(i);
    if (inferredDim.has_dim_value() && existingDim.has_dim_value() &&
        inferredDim.dim_value() != existingDim.dim_value()) {
      std::stringstream ss;
      ss << "Inferred shape and existing shape differ in dimension " << i
         << ": (" << inferredDim.dim_value() << ") vs ("
         << existingDim.dim_value() << ")";
      throw std::runtime_error(ss.str());
    }
  }
}

void mergeShapesAndTypes(
    const TypeProto_Tensor& inferredType,
    TypeProto_Tensor* existingType) {
  if (inferredType.elem_type() != TensorProto::UNDEFINED &&
      existingType->elem_type() == TensorProto::UNDEFINED) {
    existingType->set_elem_type(inferredType.elem_type());
  }

  if (!inferredType.has_shape()) {
    return;
  }

  if (!existingType->has_shape()) {
    // Ensure the shape is initialized. Note that this must be done
    // even for (zero-dimensional) scalars.
    existingType->mutable_shape();

    for (int j = 0; j < inferredType.shape().dim_size(); ++j) {
      existingType->mutable_shape()->add_dim();
    }
  }

  for (int i = 0; i < inferredType.shape().dim_size(); ++i) {
    const auto& inferredDim = inferredType.shape().dim(i);
    auto* existingDim = existingType->mutable_shape()->mutable_dim(i);
    if (!existingDim->has_dim_value()) {
      *existingDim = inferredDim;
    }
  }
}

void InferShapes(
    ModelProto& m,
    const ISchemaRegistry* schema_registry = OpSchemaRegistry::Instance()) {
  std::unordered_map<std::string, int> opset_imports;
  for (const auto& opset_import : m.opset_import()) {
    opset_imports[opset_import.domain()] =
        static_cast<int>(opset_import.version());
  }

  auto* g = m.mutable_graph();

  std::unordered_map<std::string, TypeProto*> valueTypesByName;
  for (auto& vi : *g->mutable_value_info()) {
    if (vi.has_type())
      valueTypesByName[vi.name()] = vi.mutable_type();
  }
  for (auto& vi : *g->mutable_input()) {
    if (vi.has_type())
      valueTypesByName[vi.name()] = vi.mutable_type();
  }
  for (auto& vi : *g->mutable_output()) {
    if (vi.has_type())
      valueTypesByName[vi.name()] = vi.mutable_type();
  }

  for (const auto& n : g->node()) {
    // Resolve domain for node
    auto dit = opset_imports.find(n.domain());
    if (dit == opset_imports.end()) {
      continue;
    }
    auto domain_version = dit->second;

    const auto schema =
        schema_registry->GetSchema(n.op_type(), domain_version, n.domain());
    if (!schema) {
      continue;
    }

    InferenceContextImpl ctx(n, valueTypesByName);
    try {
      schema->GetTypeAndShapeInferenceFunction()(ctx);
    } catch (const ONNX_NAMESPACE::InferenceError& ex) {
      (void)ex;
      // Continue with inference for remaining nodes
      continue;
    }

    for (int i = 0; i < n.output_size(); ++i) {
      if (!ctx.getOutputType(i)->has_tensor_type()) {
        continue;
      }
      const auto& inferredType = ctx.getOutputType(i)->tensor_type();

      // Bail out early if shape inference does nothing useful.
      if (inferredType.elem_type() == TensorProto::UNDEFINED &&
          !inferredType.has_shape()) {
        continue;
      }

      // Find any pre-existing type and shape info. If there is such,
      // then check for compatability with the inferred
      // information. Otherwise, initialize it in an empty state.
      auto iter = valueTypesByName.find(n.output(i));
      TypeProto* existingType = nullptr;
      if (iter != valueTypesByName.end()) {
        existingType = iter->second;
        checkShapesAndTypes(inferredType, existingType->tensor_type());
      } else {
        auto vi = g->add_value_info();
        vi->set_name(n.output(i));
        existingType = vi->mutable_type();
      }

      // Now we can merge pre-existing and inferred info, without
      // further need for error-checking.
      mergeShapesAndTypes(inferredType, existingType->mutable_tensor_type());

      // Make merged info available to futher inference.
      valueTypesByName[n.output(i)] = existingType;
    }
  }
}

} // namespace shape_inference
} // namespace ONNX_NAMESPACE
