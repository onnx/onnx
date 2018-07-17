// Interface for Op Version Adapters

#pragma once

#include "onnx/common/ir.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class Adapter {
  private:
    std::string name_;
    OpSetID initial_version_;
    OpSetID target_version_;

  public:
    virtual ~Adapter() noexcept = default;

    explicit Adapter(std::string name, OpSetID initial_version, OpSetID target_version)
      : name_(std::move(name)), initial_version_(std::move(initial_version)),
        target_version_(std::move(target_version)) {
    }

    virtual void adapt(std::shared_ptr<Graph> /*graph*/, const Node* node) const {}

    const std::string& name() {
      return name_;
    }

    const OpSetID& initial_version() {
      return initial_version_;
    }

    const OpSetID& target_version() {
      return target_version_;
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
