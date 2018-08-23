// Adapter for Concat in default domain from version 4 to 3

#pragma once

#include "onnx/version_converter/adapters/type_restriction.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class Concat_4_3 final : public TypeRestriction {
  public:
    explicit Concat_4_3()
      : TypeRestriction("Concat", OpSetID(4), OpSetID(3)) {}

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      std::vector<TensorProto_DataType> unallowed_types = {TensorProto_DataType_INT32,
        TensorProto_DataType_INT64, TensorProto_DataType_UINT32,
        TensorProto_DataType_UINT64, TensorProto_DataType_UINT8,
        TensorProto_DataType_UINT16, TensorProto_DataType_INT8,
        TensorProto_DataType_INT16, TensorProto_DataType_STRING,
        TensorProto_DataType_BOOL};
      TypeRestriction::adapt_type_restriction(graph, node, unallowed_types);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
