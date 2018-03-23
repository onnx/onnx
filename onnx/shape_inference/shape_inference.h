#pragma once

#include "onnx/proto_utils.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE { namespace shape_inference {

struct InferenceContextImpl : public InferenceContext {
  virtual const AttributeProto* getAttribute(const std::string& name) const {
    auto iter = attributesByName_.find(name);
    if (iter == attributesByName_.end()) {
      return nullptr;
    } else {
      return iter->second;
    }
  }
  virtual size_t getNumInputTypes() const {
    return allInputTypes_.size();
  }
  virtual const TypeProto_Tensor* getInputType(size_t index) const {
    return allInputTypes_[index];
  }
  virtual size_t getNumOutputTypes() const {
    return allOutputTypes_.size();
  }
  virtual TypeProto_Tensor* getOutputType(size_t index) {
    return &allOutputTypes_[index];
  }
  std::unordered_map<std::string, const AttributeProto *> attributesByName_;
  std::vector<const TypeProto_Tensor*> allInputTypes_;
  std::vector<TypeProto_Tensor> allOutputTypes_;
};

void InferShapes(ONNX_NAMESPACE::ModelProto& m) {
  auto g = m.mutable_graph();

  std::unordered_map<std::string, TypeProto_Tensor> valueTypesByName;
  for (auto vi : g->value_info()) {
    valueTypesByName[vi.name()] = vi.type().tensor_type();
  }
  for (auto vi : g->input()) {
    valueTypesByName[vi.name()] = vi.type().tensor_type();
  }
  for (auto vi : g->output()) {
    valueTypesByName[vi.name()] = vi.type().tensor_type();
  }

  for (auto n : g->node()) {
    auto schema = OpSchemaRegistry::Schema(n.op_type());
    if (schema) {
      InferenceContextImpl ctx;

      {
        std::unordered_map<std::string, const AttributeProto *> attributesByName;
        for (auto& attr : n.attribute()) {
          attributesByName[attr.name()] = &attr;
        }
        ctx.attributesByName_ = std::move(attributesByName);
      }

      {
        std::vector<const TypeProto_Tensor *> allInputTypes;

        for (auto input : n.input()) {
          auto iter = valueTypesByName.find(input);
          if (iter != valueTypesByName.end()) {
            allInputTypes.push_back(&iter->second);
          } else {
            allInputTypes.push_back(nullptr);
          }
        }

        ctx.allInputTypes_ = std::move(allInputTypes);
      }

      for (auto i = 0; i < n.output_size(); i++) {
        ctx.allOutputTypes_.emplace_back();
      }

      schema->GetShapeInferenceFunction()(ctx);

      // TODO: this handles the simple case of adding value_infos when
      // there were none before. It should be enhanced to refine
      // partially-specified types as well.

      for (int i = 0; i < n.output_size(); i++) {
        auto output = n.output(i);
        if (ctx.getOutputType(i)->elem_type() == TensorProto::UNDEFINED) {
          continue;
        }

        if (valueTypesByName.find(output) == valueTypesByName.end()) {
          valueTypesByName[output] = *ctx.getOutputType(i);
          auto vi = g->add_value_info();
          vi->set_name(output);
          *vi->mutable_type()->mutable_tensor_type() = *ctx.getOutputType(i);
        }
      }
    }
  }
}

}}
