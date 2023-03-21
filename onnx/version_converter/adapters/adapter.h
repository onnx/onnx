/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Interface for Op Version Adapters

#pragma once

#include <functional>

#include "onnx/onnx_pb.h"
#include "onnx/version_converter/helper.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

OperatorSetIdProto make_opset_proto(int64_t version, const std::string& domain = "") {
  OperatorSetIdProto opset_proto;
  opset_proto.set_domain("");
  opset_proto.set_version(version);
  return opset_proto;
}
class Adapter {
 private:
  std::string name_;
  OperatorSetIdProto initial_version_;
  OperatorSetIdProto target_version_;

 public:
  virtual ~Adapter() noexcept = default;

  explicit Adapter(const std::string& name, const OperatorSetIdProto& initial_version, const OperatorSetIdProto& target_version)
      : name_(name), initial_version_(initial_version), target_version_(target_version) {}

  // This will almost always return its own node argument after modifying it in place.
  // The only exception are adapters for deprecated operators: in this case the input
  // node must be destroyed and a new one must be created and returned. See e.g.
  // upsample_9_10.h
  virtual NodeProto* adapt(std::shared_ptr<GraphProto> /*graph*/, NodeProto* node) const = 0;

  const std::string& name() const {
    return name_;
  }

  const OperatorSetIdProto& initial_version() const {
    return initial_version_;
  }

  const OperatorSetIdProto& target_version() const {
    return target_version_;
  }
};

using NodeTransformerFunction = std::function<NodeProto*(std::shared_ptr<GraphProto>, NodeProto* node)>;

class GenericAdapter final : public Adapter {
 public:
  GenericAdapter(const char* op, int64_t from, int64_t to, NodeTransformerFunction transformer)
      : Adapter(op, OperatorSetIdProto(make_opset_proto(from)), OperatorSetIdProto(make_opset_proto(to))), transformer_(transformer) {}

  NodeProto* adapt(std::shared_ptr<GraphProto> graph, NodeProto* node) const override {
    return transformer_(graph, node);
  }

 private:
  NodeTransformerFunction transformer_;
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
