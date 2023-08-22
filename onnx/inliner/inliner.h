// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {
namespace inliner {

using FunctionIdVector = std::vector<std::pair<std::string, std::string>>;

class FunctionIdSet {
 public:
  virtual bool Contains(const std::string& function_domain, const std::string& function_name) const = 0;
  virtual ~FunctionIdSet() = default;

  // Factory methods for creating FunctionIdSet instances.

  // Creates a set representing the elements in the given vector, if invert is false.
  // Otherwise, creates a set representing elements not in the given vector.
  static std::unique_ptr<FunctionIdSet> Create(FunctionIdVector&& function_ids, bool invert = false);
};

void InlineSelectedFunctions(ModelProto& model, const FunctionIdSet& to_inline);

void InlineLocalFunctions(ModelProto& model, bool convert_version = false);

} // namespace inliner
} // namespace ONNX_NAMESPACE
