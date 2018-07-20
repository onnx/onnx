// Adapter for Add in default domain from version 6 to 5

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class Add_6_5 final : public Adapter {
  public:
    explicit Add_6_5()
      : Adapter("Add", OpSetID(6), OpSetID(5)) {
      }

    void adapt_add_6_5(std::shared_ptr<Graph> graph, Node* node)
      const {
      // Since consumed_inputs is optional, no need to add it (as in batchnorm)
      // Need to enforce types: unit32, uint64, int32, int64 not allowed
      // Iterate over all inputs and outputs
      for (Value* input : node->inputs()) {
        isUnallowed(input);
      }
      for (Value* output : node->outputs()) {
        isUnallowed(output);
      }
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_add_6_5(graph, node);
    }

  private:
    void isUnallowed(Value* val) const {
      TensorProto_DataType unallowed_types[] = {TensorProto_DataType_INT32,
        TensorProto_DataType_INT64, TensorProto_DataType_UINT32,
        TensorProto_DataType_UINT64};
      ONNX_ASSERTM(std::find(std::begin(unallowed_types), std::end(unallowed_types),
        val->elemType()) == std::end(unallowed_types), "DataType of Input or Output"
        " of Add is of an unallowed type for Opset Version 1.");
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
