// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnx/common/constants.h"
#include "onnx/common/status.h"
#include "onnx/onnx-operators_pb.h"

#include "attr_proto_util.h"

namespace ONNX_NAMESPACE {
// Helper function to expand a function node given the function proto
void FunctionExpandHelper(
    const NodeProto& node,
    const FunctionProto& func,
    GraphProto& g,
    const std::string& node_prefix = "");

class FunctionBodyHelper {
 public:
  struct AttributeProtoWrapper {
    AttributeProto proto;

    static std::unordered_map<std::string, AttributeProto_AttributeType>
        attr_name_map;

    AttributeProtoWrapper() {}

    AttributeProtoWrapper(const AttributeProto& attr_prot) {
      proto = attr_prot;
    }

    template <typename T>
    AttributeProtoWrapper(const std::string& attr_name, T value) {
      proto = MakeAttribute(attr_name, value);
    }

   private:
    void InitFromString(const std::string& attr_name, const std::string& value);
  };

  struct NodeDef {
    std::vector<std::string> outputs;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<AttributeProtoWrapper> attributes;
  };

  static std::vector<NodeProto> Define(const std::vector<NodeDef>& node_defs);

  template <typename T>
  static TensorProto ToTensor(const T& value);

  template <typename T>
  static TensorProto ToTensor(const std::vector<T>& values);

#define DEFINE_TO_TENSOR_ONE(type, enumType, field)      \
  template <>                                            \
  static TensorProto ToTensor<type>(const type& value) { \
    TensorProto t;                                       \
    t.set_data_type(##enumType);                         \
    t.add_##field##_data(value);                         \
    return t;                                            \
  }

#define DEFINE_TO_TENSOR_LIST(type, enumType, field)                   \
  template <>                                                          \
  static TensorProto ToTensor<type>(const std::vector<type>& values) { \
    TensorProto t;                                                     \
    t.clear_##field##_data();                                          \
    t.set_data_type(##enumType);                                       \
    for (const auto& val : values) {                                   \
      t.add_##field##_data(val);                                       \
    }                                                                  \
    return t;                                                          \
  }

  DEFINE_TO_TENSOR_ONE(float, TensorProto_DataType_FLOAT, float);
  DEFINE_TO_TENSOR_ONE(int, TensorProto_DataType_INT32, int32);
  DEFINE_TO_TENSOR_ONE(double, TensorProto_DataType_DOUBLE, double);
  DEFINE_TO_TENSOR_LIST(float, TensorProto_DataType_FLOAT, float);
  DEFINE_TO_TENSOR_LIST(int, TensorProto_DataType_INT32, int32);
  DEFINE_TO_TENSOR_LIST(double, TensorProto_DataType_DOUBLE, double);

  template <typename T>
  static NodeDef Const(const std::string& name, const T& value) {
    AttributeProto attr = MakeAttribute("value", ToTensor<T>(value));
    return NodeDef{{name}, "Constant", {}, {AttributeProtoWrapper(attr)}};
  }

  template <typename T>
  static NodeDef Const(const std::string& name, const std::vector<T>& values) {
    AttributeProto attr = MakeAttribute("value", ToTensor<T>(value));
    return NodeDef{{name}, "Constant", {}, {AttributeProtoWrapper(attr)}};
  }
};

} // namespace ONNX_NAMESPACE
