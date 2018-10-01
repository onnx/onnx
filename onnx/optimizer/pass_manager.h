#pragma once
// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct PassManagerAnalysis {};
struct EmptyPassManagerAnalysis : PassManagerAnalysis {};

class PassManager {
 public:
  PassManager();
  virtual ~PassManager();

  virtual void add(Pass* P) = 0;
  virtual PassManagerAnalysis run(Graph& graph) = 0;
};

class GeneralPassManager : public PassManager {
 public:
  GeneralPassManager() {}
  ~GeneralPassManager() override {}

  void add(Pass* pass) override;
  PassManagerAnalysis run(Graph& graph) override;

 private:
  std::set<Pass*> passes;
};
} // namespace optimization
} // namespace ONNX_NAMESPACE
