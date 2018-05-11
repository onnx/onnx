#ifndef NOP_H
#define NOP_H

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE { namespace optimization {

struct Nop final : public OptimizePass {
  explicit Nop()
    : OptimizePass("nop", API_TYPE::IR) {
  }

  void optimize(Graph& /*graph*/) override {
  }

  void optimize(ONNX_NAMESPACE::ModelProto& /*mp*/) override {
  }
};

}} // namespace ONNX_NAMESPACE::optimization

#endif
