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

    explicit Adapter(const std::string& name, OpSetID initial_version, OpSetID target_version)
      : name_(name), initial_version_(std::move(initial_version)),
        target_version_(std::move(target_version)) {
    }

    virtual void adapt(std::shared_ptr<Graph> /*graph*/, Node* node) const = 0;

    const std::string& name() const {
      return name_;
    }

    const OpSetID& initial_version() const {
      return initial_version_;
    }

    const OpSetID& target_version() const {
      return target_version_;
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
