// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {
namespace inliner {

using FunctionIdVector = std::vector<std::pair<std::string, std::string>>;

class FunctionIdSet {
public:
  virtual bool contains(const std::string& function_domain, const std::string& function_name) const = 0;
  virtual ~FunctionIdSet() = default;

  static std::unique_ptr<FunctionIdSet> create(FunctionIdVector&& function_ids);
};

void InlineSelectedFunctions(ModelProto& model, const FunctionIdSet& to_inline);

void InlineLocalFunctions(ModelProto& model, bool convert_version = false);

} // namespace inliner
} // namespace ONNX_NAMESPACE
