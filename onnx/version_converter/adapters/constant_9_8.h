// Adapter for Constant in default domain from version 9 to 8

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct Constant_9_8 final : public Adapter {
  explicit Constant_9_8()
    : Adapter("Constant", OpSetID(9), OpSetID(8)) {
    }

  template<typename T1, typename T2>
  void cast_func(std::vector<T1>& v1, std::vector<T2>& v2) const {
    for (size_t i = 0; i < v1.size(); i++) {
      v2.push_back(static_cast<T2>(v1[i]));
    }
  }

  void adapt_constant_9_8(std::shared_ptr<Graph> graph, Node* node) const {

    const ArrayRef<Value*>& outputs = node->outputs();
    const std::string original_output_name = node->output()->uniqueName();
    const int output_type = outputs[0]->elemType();
    int const_type;
    Tensor val;

    if (output_type == TensorProto_DataType::TensorProto_DataType_FLOAT ||
        output_type == TensorProto_DataType::TensorProto_DataType_FLOAT16 ||
        output_type == TensorProto_DataType::TensorProto_DataType_DOUBLE)
        return;
    
    const std::unordered_set<int> &cast_to_float_types = { 
        TensorProto_DataType::TensorProto_DataType_BOOL,
        TensorProto_DataType::TensorProto_DataType_INT8,
        TensorProto_DataType::TensorProto_DataType_INT16,
        TensorProto_DataType::TensorProto_DataType_INT32,
        TensorProto_DataType::TensorProto_DataType_UINT8,
        TensorProto_DataType::TensorProto_DataType_UINT16,
    };

    const std::unordered_set<int> &cast_to_double_types = {
        TensorProto_DataType::TensorProto_DataType_UINT32, 
        TensorProto_DataType::TensorProto_DataType_INT64,
        TensorProto_DataType::TensorProto_DataType_UINT64,
    };

    if (cast_to_float_types.find(output_type) != cast_to_float_types.end()) {
       const_type = TensorProto_DataType::TensorProto_DataType_FLOAT;
    } else if (cast_to_double_types.find(output_type) != cast_to_double_types.end()) {
       const_type = TensorProto_DataType::TensorProto_DataType_DOUBLE;
    } else {
       ONNX_ASSERT("Unsupported Output Type");
    }

    if (node->hasAttribute(kvalue)) {
      Tensor t(node->t(kvalue));
      node->removeAttribute(kvalue);
      val.sizes() = t.sizes();
      val.elem_type() = const_type;
      if (cast_to_double_types.find(output_type) == cast_to_double_types.end()) {
        cast_func<int32_t, float> (t.int32s(), val.floats());
      } else if (output_type == TensorProto_DataType::TensorProto_DataType_INT64) {
        cast_func<int64_t, double> (t.int64s(), val.doubles());
      } else {
        cast_func<uint64_t, double> (t.uint64s(), val.doubles());
      }

      node->t_(kvalue, val);

      const use_list original_uses(node->output()->uses());
      node->output()->setElemType(const_type);
      node->output()->setUniqueName(original_output_name + "_intermediate_output");
      Node *post_cast = graph->create(kCast, outputs[0]);
      post_cast->i_(kto, output_type);
      post_cast->output()->setUniqueName(original_output_name);
      post_cast->output()->setSizes(outputs[0]->sizes());
      post_cast->output()->setElemType(output_type);

      post_cast->insertAfter(node);

      for (Use u : original_uses) {
        u.user->replaceInputWith(node->output(), post_cast->output());
      }

    }
  }

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_constant_9_8(graph, node);
  }

};

}}