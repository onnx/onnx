#pragma once
// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include <set>
#include "onnx/optimizer/pass.h"
#include "onnx/optimizer/passes/eliminate_deadend.h"

namespace ONNX_NAMESPACE {
namespace optimization {

// An analysis returned from the run done by a manager
struct PassManagerAnalysis {};
struct EmptyPassManagerAnalysis : PassManagerAnalysis {};

// Base class of all PassManager's. The class should be able to add new passes
// as well as run the passes given a graph.
class PassManager {
 public:
  PassManager();
  virtual ~PassManager();

  virtual void add(std::shared_ptr<Pass> P) = 0;
  virtual PassManagerAnalysis* run(Graph& graph) = 0;
};

// The GeneralPassManager has no restriction on type of Pass and runs the passes
// once in a linear fashion.
class GeneralPassManager : public PassManager {
 public:
  GeneralPassManager() {}
  ~GeneralPassManager() override;

  void add(std::shared_ptr<Pass> pass) override;
  PassManagerAnalysis* run(Graph& graph) override;

 protected:
  std::set<std::shared_ptr<Pass>> passes;
};

// Exhibits the same behavior as GeneralPassManager but will instead check
// whether or not fixed point optimization is needed.
class FixedPointPassManager : public GeneralPassManager {
  PassManagerAnalysis* run(Graph& graph) override;
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
