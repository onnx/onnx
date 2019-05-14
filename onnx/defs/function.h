// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "attr_proto_util.h"
#include "onnx/common/constants.h"
#include "onnx/common/status.h"
#include "onnx/onnx-operators_pb.h"
#include "tensor_proto_util.h"

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

    AttributeProtoWrapper() {}

    AttributeProtoWrapper(const AttributeProto& attr_prot) {
      proto = attr_prot;
    }

    template <typename T>
    AttributeProtoWrapper(const std::string& attr_name, T value) {
      proto = MakeAttribute(attr_name, value);
    }
  };

  struct NodeDef {
    NodeDef(
        const std::vector<std::string>& outputs,
        const std::string& op_type,
        const std::vector<std::string>& inputs)
        : outputs(outputs), op_type(op_type), inputs(inputs) {}

    NodeDef(
        const std::vector<std::string>& outputs,
        const std::string& op_type,
        const std::vector<std::string>& inputs,
        const std::vector<AttributeProtoWrapper>& attributes)
        : outputs(outputs),
          op_type(op_type),
          inputs(inputs),
          attributes(attributes) {}

    std::vector<std::string> outputs;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<AttributeProtoWrapper> attributes;
  };

  /*
  BuildNodes() is an utility function for easily define a Function Body.

  To build a simple node:
    {{"Z"}, "Add", {"X", "Y"}} represents Z = Add(X,Y)

  To build a node with attribute:
    {{"Y"}, "Concat", {"X1", "X2", "X3"}, {{"axis", 1}}}
      represents Y = Concat(X1,X2,X3) with axis = 1
    The attribute type are infered from the attribute value's c++ type
    Supported value types are
      int64_t -> int, vector<int64_t> -> ints
      float -> float, vector<float> -> floats
      string -> string, vector<string> ->strings
    For refering an attribute from parent, use:
      {MakeRefAttribute("axes", AttributeProto::INTS)}}

  For more examples, please find the references of this function
  */
  static std::vector<NodeProto> BuildNodes(
      const std::vector<NodeDef>& node_defs);

  template <typename T>
  static NodeDef Const(const std::string& name, const T& value) {
    return NodeDef{{name}, "Constant", {}, {{"value", ToTensor<T>(value)}}};
  }

  template <typename T>
  static NodeDef Const(const std::string& name, const std::vector<T>& values) {
    return NodeDef{{name}, "Constant", {}, {{"value", ToTensor<T>(values)}}};
  }
};

} // namespace ONNX_NAMESPACE
