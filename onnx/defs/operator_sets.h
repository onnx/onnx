// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx/defs/generated/opsets_onnx.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

// The operator set classes and 'RegisterOnnxOperatorSetSchema' are generated
// from the spec files in 'operator-specifications'.

ONNX_API inline void DeregisterOnnxOperatorSetSchema() {
  OpSchemaRegistry::OpSchemaDeregisterAll(ONNX_DOMAIN);
  // -1 means no ONNX schema is loaded
  OpSchemaRegistry::SetLoadedSchemaVersion(-1);
}

// Returns true if ONNX was built with static schema registration disabled
// (i.e., with __ONNX_DISABLE_STATIC_REGISTRATION defined).
//
// When static registration is disabled, the linking module is responsible for
// calling the desired registration methods to register the required opsets:
//   - RegisterOnnxOperatorSetSchema() for ai.onnx domain operators
//   - RegisterOnnxMLOperatorSetSchema() for ai.onnx.ml domain operators (if ONNX_ML is enabled)
//   - RegisterOnnxTrainingOperatorSetSchema() for ai.onnx.training domain operators
//   - RegisterOnnxPreviewOperatorSetSchema() for ai.onnx.preview domain operators
//
// When static registration is enabled (this function returns false), the above
// opsets are automatically registered during ONNX library initialization, and
// the linking module should not call the registration methods again to avoid
// duplicate schema registration errors.
ONNX_API bool IsOnnxStaticRegistrationDisabled();

} // namespace ONNX_NAMESPACE
