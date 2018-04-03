#ifndef NOP_H
#define NOP_H

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE { namespace optimization {

struct Nop : public OptimizePass {
  explicit Nop()
    : OptimizePass("nop", API_TYPE::IR) {
  }

  virtual void optimize(Graph& graph) {
  }
};

}} // namespace ONNX_NAMESPACE::optimization

#endif
