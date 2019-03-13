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

    template <typename T>
    AttributeProtoWrapper(T val) {
      SetAttrValue(val, &proto);
    }

   private:
    void InitFromString(const std::string& val);
  };

  struct NodeDef {
    std::vector<std::string> outputs;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::pair<std::string, AttributeProtoWrapper>> attributes;
  };

  static std::vector<NodeProto> Define(std::vector<NodeDef> node_defs);

  template <typename T>
  static ONNX_NAMESPACE::TensorProto ToTensor(T val);

  template <typename T>
  static ONNX_NAMESPACE::TensorProto ToTensor(std::vector<T> vals);

#define DEFINE_TO_TENSOR_ONE(ARG_TYPE, ATTR_TYPE, FIELD)                \
  template <>                                                           \
  static ONNX_NAMESPACE::TensorProto ToTensor<ARG_TYPE>(ARG_TYPE val) { \
    TensorProto t;                                                      \
    t.set_data_type(TensorProto_DataType_##ATTR_TYPE);                  \
    t.add_##FIELD##_data(val);                                          \
    return t;                                                           \
  }

#define DEFINE_TO_TENSOR_LIST(ARG_TYPE, ATTR_TYPE, FIELD) \
  template <>                                             \
  static ONNX_NAMESPACE::TensorProto ToTensor<ARG_TYPE>(  \
      std::vector<ARG_TYPE> vals) {                       \
    TensorProto t;                                        \
    t.clear_float_data();                                 \
    t.set_data_type(TensorProto_DataType_##ATTR_TYPE);    \
    for (const auto& val : vals) {                        \
      t.add_##FIELD##_data(val);                          \
    }                                                     \
    return t;                                             \
  }

  DEFINE_TO_TENSOR_ONE(float, FLOAT, float);
  DEFINE_TO_TENSOR_ONE(int, INT32, int32);
  DEFINE_TO_TENSOR_ONE(double, DOUBLE, double);
  DEFINE_TO_TENSOR_LIST(float, FLOAT, float);
  DEFINE_TO_TENSOR_LIST(int, INT32, int32);
  DEFINE_TO_TENSOR_LIST(double, DOUBLE, double);

  template <typename T>
  static NodeDef Const(const std::string& name, const T& val) {
    NodeDef n = {{name}, "Constant"};

    AttributeProto attr;
    attr.set_name("value");
    attr.set_type(AttributeProto_AttributeType_TENSOR);

    TensorProto* t_proto = attr.mutable_t();
    *t_proto = ToTensor<T>(val);

    n.attributes.push_back(
        std::make_pair("value", AttributeProtoWrapper(attr)));
    return n;
  }

  template <typename T>
  static NodeDef Const(const std::string& name, const std::vector<T>& vals) {
    NodeDef n = {{name}, "Constant"};

    AttributeProto attr;
    attr.set_name("value");
    attr.set_type(AttributeProto_AttributeType_TENSOR);

    TensorProto* t_proto = attr.mutable_t();
    *t_proto = ToTensor(vals);

    n.attributes.push_back(
        std::make_pair("value", AttributeProtoWrapper(attr)));
    return n;
  }
};

template <>
inline FunctionBodyHelper::AttributeProtoWrapper::AttributeProtoWrapper(
    const char* val) {
  InitFromString(val);
}

template <>
inline FunctionBodyHelper::AttributeProtoWrapper::AttributeProtoWrapper(
    const std::string& val) {
  InitFromString(val);
}

} // namespace ONNX_NAMESPACE
