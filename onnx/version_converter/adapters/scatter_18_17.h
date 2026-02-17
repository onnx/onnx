// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for ScatterElements and ScatterND in default domain from version 18 to 17.

#pragma once

#include "onnx/common/interned_strings.h"
#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class ScatterElements_18_17 : public Adapter {
 public:
  ScatterElements_18_17() : Adapter("ScatterElements", OpSetID(18), OpSetID(17)) {}

  Node* adapt(std::shared_ptr<Graph> /*graph*/, Node* node) const override {
    if (node->hasAttribute(Symbol("reduction"))) {
      const std::string& r = node->s(Symbol("reduction"));
      ONNX_ASSERTM(
          r != "max" && r != "min", "Scatter reduction 'max' and 'min' are not supported when downgrading to opset 17");
    }
    return node;
  }
};

class ScatterND_18_17 : public Adapter {
 public:
  ScatterND_18_17() : Adapter("ScatterND", OpSetID(18), OpSetID(17)) {}

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
