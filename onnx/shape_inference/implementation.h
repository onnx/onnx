#pragma once

#include "onnx/proto_utils.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE { namespace shape_inference {

struct InferenceContextImpl : public InferenceContext {
  InferenceContextImpl(const NodeProto & n,
                       const std::unordered_map<std::string, TypeProto_Tensor *>& valueTypesByName) {
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
  const TypeProto_Tensor* getInputType(size_t index) const override {
    if (index >= allInputTypes_.size()) {
      throw std::runtime_error("input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds");
    }
    return allInputTypes_[index];
  }
  size_t getNumOutputs() const override {
    return allOutputTypes_.size();
  }
  TypeProto_Tensor* getOutputType(size_t index) override {
    if (index >= allOutputTypes_.size()) {
      throw std::runtime_error("output " + ONNX_NAMESPACE::to_string(index) + " is out of bounds");
    }
    return &allOutputTypes_[index];
  }
  std::unordered_map<std::string, const AttributeProto *> attributesByName_;
  std::vector<const TypeProto_Tensor*> allInputTypes_;
  std::vector<TypeProto_Tensor> allOutputTypes_;
};

 void mergeShapesAndTypes(const TypeProto_Tensor& inferredType, TypeProto_Tensor* existingType, const std::string& output) {
}

void InferShapes(ModelProto& m) {
  std::unordered_map<std::string, int> opset_imports;
  for (const auto& opset_import : m.opset_import()) {
    opset_imports[opset_import.domain()] = static_cast<int>(opset_import.version());
  }

  auto* g = m.mutable_graph();

  std::unordered_map<std::string, TypeProto_Tensor *> valueTypesByName;
  for (auto& vi : *g->mutable_value_info()) {
    valueTypesByName[vi.name()] = vi.mutable_type()->mutable_tensor_type();
  }
  for (auto& vi : *g->mutable_input()) {
    valueTypesByName[vi.name()] = vi.mutable_type()->mutable_tensor_type();
  }
  for (auto& vi : *g->mutable_output()) {
    valueTypesByName[vi.name()] = vi.mutable_type()->mutable_tensor_type();
  }

  for (const auto& n : g->node()) {
    // Resolve domain for node
    auto dit = opset_imports.find(n.domain());
    if (dit == opset_imports.end()) {
      continue;
    }
    auto domain_version = dit->second;

    const auto schema = OpSchemaRegistry::Schema(n.op_type(), domain_version, n.domain());
    if (!schema) {
      continue;
    }

    InferenceContextImpl ctx(n, valueTypesByName);

    schema->GetShapeInferenceFunction()(ctx);

    for (int i = 0; i < n.output_size(); ++i) {
      const auto& output = n.output(i);
      const auto& inferredType = *ctx.getOutputType(i);

      // In this case, we have no new information, so don't bother
      // to add a contentless ValueInfo.
      if (inferredType.elem_type() == TensorProto::UNDEFINED &&
          !inferredType.has_shape()) {
        continue;
      }

      // If there is already a ValueInfo associated with this
      // output, reuse it. Otherwise add a new one.
      auto iter = valueTypesByName.find(output);
      TypeProto_Tensor* existingType = nullptr;
      if (iter != valueTypesByName.end()) {
        existingType = iter->second;
      } else {
        auto vi = g->add_value_info();
        vi->set_name(output);
        existingType = vi->mutable_type()->mutable_tensor_type();
      }

      // Incorporate the inferred information.
      if (inferredType.elem_type() != TensorProto::UNDEFINED) {
        if (existingType->elem_type() != TensorProto::UNDEFINED &&
            existingType->elem_type() != inferredType.elem_type()) {
          throw std::runtime_error("inferred type differs from existing type");
        } else {
          existingType->set_elem_type(inferredType.elem_type());
        }
      }

      if (inferredType.has_shape()) {
        if (existingType->has_shape()) {
          if (inferredType.shape().dim_size() != existingType->shape().dim_size()) {
            throw std::runtime_error("inferred type and existing type are of different rank");
          }
        } else {
          // make sure has_shape() == True for scalars
          existingType->mutable_shape();

          for (int j = 0; j < inferredType.shape().dim_size(); ++j) {
            existingType->mutable_shape()->add_dim();
          }
        }

        for (int j = 0; j < inferredType.shape().dim_size(); ++j) {
          const auto& inferredDim = inferredType.shape().dim(j);
          auto* existingDim = existingType->mutable_shape()->mutable_dim(j);
          if (inferredDim.has_dim_value()) {
            auto inferredDimValue = inferredDim.dim_value();
            if (existingDim->has_dim_value() && existingDim->dim_value() != inferredDimValue) {
              throw std::runtime_error("inferred dimension differs from existing dimension");
            }
            *existingDim = inferredDim;
          }
        }
      }

      // Make it available to futher inference.
      valueTypesByName[output] = existingType;
    }
  }
}

}}
