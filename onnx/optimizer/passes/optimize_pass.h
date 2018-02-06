// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/common/ir.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE { namespace optimization {

enum class API_TYPE {
  PROTO, IR
};

struct OptimizePass {

  virtual ~OptimizePass() {}

  std::string name;
  API_TYPE type;

  explicit OptimizePass(const std::string& name, API_TYPE type)
    : name(name), type(type) {
  }

  virtual void optimize(ONNX_NAMESPACE::ModelProto& mp) {}

  virtual void optimize(Graph& graph) {}

};


}} // namespace ONNX_NAMESPACE::optimization
