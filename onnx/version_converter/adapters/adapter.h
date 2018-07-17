// Interface for Op Version Adapters

#pragma once

#include "onnx/common/ir.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class Adapter {

  virtual ~Adapter() noexcept = default;

  std::string name;
  OpSetID initial_version;
  OpSetID target_version;

  explicit Adapter(std::string name, OpSetID initial_version, OpSetID target_version)
    : name(std::move(name)), initial_version(std::move(initial_version)),
      target_version(std::move(target_version)) {
  }

  virtual void adapt(std::shared_ptr<Graph> /*graph*/, const Node* node) const {}
};

}} // namespace ONNX_NAMESPACE::version_conversion
