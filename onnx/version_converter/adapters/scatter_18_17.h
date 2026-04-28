// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for ScatterElements and ScatterND in default domain from version 18 to 17.
// Opset 18 added reduction values "max" and "min"; downgrading rejects those.

#pragma once

#include <string>

#include "onnx/common/interned_strings.h"
#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Scatter_18_17 : public Adapter {
 public:
  explicit Scatter_18_17(const std::string& op_name) : Adapter(op_name, OpSetID(18), OpSetID(17)) {}

  Node* adapt(std::shared_ptr<Graph> /*graph*/, Node* node) const override {
    if (node->hasAttribute(Symbol("reduction"))) {
      const std::string& r = node->s(Symbol("reduction"));
      ONNX_ASSERTM(
          r != "max" && r != "min", "Scatter reduction 'max' and 'min' are not supported when downgrading to opset 17");
    }
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
