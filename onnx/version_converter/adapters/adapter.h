// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Interface for Op Version Adapters

#pragma once

#include "onnx/common/ir.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

enum class API_TYPE : uint8_t {
  PROTO, IR
};

struct Adapter {

  virtual ~Adapter() noexcept = 0;

  std::string name;
  API_TYPE type;

  explicit Adapter(std::string name, API_TYPE type)
    : name(std::move(name)), type(type) {
  }

  virtual void adapt(ONNX_NAMESPACE::ModelProto& /*mp*/) {}

  virtual void adapt(Graph& /*graph*/) {}
};

inline Adapter::~Adapter() noexcept = default;

}} // namespace ONNX_NAMESPACE::version_conversion
