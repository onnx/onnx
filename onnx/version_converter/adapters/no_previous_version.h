// Adapter indicating lack of a previous version of some op before a given
// opset version.

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class NoPreviousVersionAdapter final : public Adapter {
  public:
    explicit NoPreviousVersionAdapter(const std::string& op_name, const OpSetID&
      initial, const OpSetID& target): Adapter(op_name,
      initial, target) {}

  void adapt(std::shared_ptr<Graph>, Node*) const override {
    ONNX_ASSERTM(false, "No Previous Version of %s exists", name().c_str());
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion
