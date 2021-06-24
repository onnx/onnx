/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "attr_proto_util.h"
#include "onnx/common/constants.h"
#include "onnx/common/status.h"
#include "onnx/defs/schema.h"
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
        std::vector<std::string> outputs,
        std::string op_type,
        std::vector<std::string> inputs,
        std::vector<AttributeProtoWrapper> attributes = {},
        std::string domain = "")
        : outputs(std::move(outputs)),
          op_type(std::move(op_type)),
          inputs(std::move(inputs)),
          attributes(std::move(attributes)),
          domain(std::move(domain)) {}

    std::vector<std::string> outputs;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<AttributeProtoWrapper> attributes;
    std::string domain;
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

  To build a node which belongs to a domain other than onnx standard domain:
    {{"Z"}, "Foo", {"X", "Y"}, "customdomain"} represents Z = customdomain.Foo(X,Y)
    or
    {{"Y"}, "Bar", {"X1", "X2", "X3"}, {{"axis", 1}}, "customdomain"}
      represents Y = customdomain.Bar(X1,X2,X3) with axis = 1

  For more examples, please find the references of this function
  */
  static std::vector<NodeProto> BuildNodes(const std::vector<NodeDef>& node_defs);

  static void BuildNodes(FunctionProto& functionProto, const std::vector<NodeDef>& node_defs);

  static bool BuildFunctionProto(
      FunctionProto& functionProto,
      const OpSchema& schema,
      const std::vector<NodeDef>& node_defs,
      const std::vector<OperatorSetIdProto>& relied_opsets);

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
