// Copyright (c) Facebook Inc. and Microsoft Corporation.
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

  static std::vector<NodeProto> BuildNodes(
      const std::vector<NodeDef>& node_defs);

  template <typename T>
  static NodeDef Const(const std::string& name, const T& value) {
    AttributeProto attr = MakeAttribute("value", ToTensor<T>(value));
    return NodeDef{{name}, "Constant", {}, {AttributeProtoWrapper(attr)}};
  }

  template <typename T>
  static NodeDef Const(const std::string& name, const std::vector<T>& values) {
    AttributeProto attr = MakeAttribute("value", ToTensor<T>(values));
    return NodeDef{{name}, "Constant", {}, {AttributeProtoWrapper(attr)}};
  }
};

} // namespace ONNX_NAMESPACE
