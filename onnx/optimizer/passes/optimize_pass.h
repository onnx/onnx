#pragma once

#include "onnx/common/ir.h"
#include "onnx/onnx_pb.h"

namespace onnx { namespace optimization {

enum class API_TYPE {
  PROTO, IR
};

struct OptimizePass {

  std::string name;
  API_TYPE type;

  explicit OptimizePass(const std::string& name, API_TYPE type)
    : name(name), type(type) {
  }

  virtual void optimize(onnx::ModelProto& mp) {}

  virtual void optimize(Graph& graph) {}

};


}} // namespace onnx::optimization
