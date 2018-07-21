// Adapter for Add in default domain from version 6 to 5

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class TypeRestriction : public Adapter {
  public:
    explicit TypeRestriction(const std::string& op_name, const OpSetID&
      initial, const OpSetID& target): Adapter(std::move(op_name), std::move(
        initial), std::move(target)) {}

    void adapt_type_restriction(std::shared_ptr<Graph> graph, Node* node,
      std::vector<TensorProto_DataType> unallowed_types) const {
      // Since consumed_inputs is optional, no need to add it (as in batchnorm)
      // Need to enforce types: unit32, uint64, int32, int64 not allowed
      // Iterate over all inputs and outputs
      for (Value* input : node->inputs()) {
        isUnallowed(input, unallowed_types);
      }
      for (Value* output : node->outputs()) {
        isUnallowed(output, unallowed_types);
      }
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      std::vector<TensorProto_DataType> unallowed_types = {TensorProto_DataType_INT32,
        TensorProto_DataType_INT64, TensorProto_DataType_UINT32,
        TensorProto_DataType_UINT64};
      adapt_type_restriction(graph, node, unallowed_types);
    }

  private:
    void isUnallowed(Value* val, std::vector<TensorProto_DataType> unallowed_types) const {
      ONNX_ASSERTM(std::find(std::begin(unallowed_types), std::end(unallowed_types),
        val->elemType()) == std::end(unallowed_types), "DataType of Input or Output"
        " of Add is of an unallowed type for Opset Version 1.");
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
