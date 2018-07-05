// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Interface for Op Version Adapters

#pragma once

#include "onnx/common/ir.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct Adapter {

  virtual ~Adapter() noexcept = 0;

  std::string name;
  OpSetID initial_version;
  OpSetID target_version;

  explicit Adapter(std::string name, OpSetID initial_version, OpSetID target_version)
    : name(std::move(name)), initial_version(initial_version), target_version(target_version) {
  }

  virtual void adapt(Graph& /*graph*/) {}
};

inline Adapter::~Adapter() noexcept = default;

}} // namespace ONNX_NAMESPACE::version_conversion
