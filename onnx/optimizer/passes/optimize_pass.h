#pragma once

#include "onnx/onnx_pb.h"
#include "onnx/ir.h"

namespace onnx { namespace optimization {

enum class API_TYPE {
  proto, ir
};

struct OptimizePass {

  std::string name;
  API_TYPE type;

  explicit OptimizePass(const std::string name, API_TYPE type)
    : name(name), type(type) {
  }

  virtual void optimize(onnx::ModelProto& mp) {}

  virtual void optimize(std::shared_ptr<Graph>& graph) {}

};


}} // namespace onnx::optimization
