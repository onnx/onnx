// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/common/ir.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE { namespace optimization {

enum class API_TYPE : uint8_t {
  PROTO, IR
};

struct OptimizePass {

  virtual ~OptimizePass() noexcept = 0;

  std::string name;
  API_TYPE type;

  explicit OptimizePass(std::string name, API_TYPE type)
    : name(std::move(name)), type(type) {
  }

  virtual void optimize(ONNX_NAMESPACE::ModelProto& /*mp*/) {}

  virtual void optimize(Graph& /*graph*/) {}

  void DescendOnGraphAttributes(Node * n, std::function<void(Graph&)> fn) {
    for (auto name : n->attributeNames()) {
      auto kind = n->kindOf(name);
      if (kind == AttributeKind::g) {
        fn(*n->g(name));
      }
      if (kind == AttributeKind::gs) {
        for (auto & g  : n->gs(name)) {
          fn(*g);
        }
      }
    }
  }

};

inline OptimizePass::~OptimizePass() noexcept = default;

}} // namespace ONNX_NAMESPACE::optimization
