// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <string>
#include "onnx/common/ir.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {
namespace optimization {

// Base struct representing result of a pass.
struct PostPassAnalysis {};

// Enum that represents the type of optimization it is.
enum PassType {
  // Class of optimizations that fuses operations.
  Fuse = 0,
  // Class of optimizations that remove useless operations.
  Nop = 1,
  // Class of optimizations that includes some form of seperation.
  Seperate = 2,
  // Immutable pass, also sometimes referred to as an analysis pass.
  Immutable = 3,
  // Unknown type of pass.
  Unk = 4
};

enum PassEfficiency {
  // A partially efficient optimization pass cannot guarantee that running two
  // consecutive passes
  // will return the same result as running a single pass.
  Partial = 0,
  // A completely efficient optimization guarantees that running two consecutive
  // passes is equivalent
  // to running a single pass.
  Complete = 1
};

// Describes what the optimization pass is attempting to optimize.
enum PassOptimizationType {
  // Is not optimizing anything. Most likely will be used in an immutable pass.
  None = 0,
  // Optimizes for compute.
  Compute = 1,
  // Optimizes for memory.
  Memory = 2,
  // Optimizes for both compute and memory.
  ComputeMemory = 3,
  // Optimizes for stability (e.g. log-sum-exp trick).
  Stability = 4
};

// Base class for all optimizations within ONNX. A pass must contain the
// annotations described above. Furthermore each pass is given the ability to
// initialize and finalize it's pass. Each pass must have a unique name that
// pass managers/registry will use as identification. Finally the pass
// implements runPass which completes the pass inplace.
class Pass {
  PassType pass_type;
  PassEfficiency pass_efficiency;
  PassOptimizationType pass_optimization_type;

 public:
  Pass(
      PassType pass_type,
      PassEfficiency pass_efficiency,
      PassOptimizationType pass_optimization_type);
  virtual ~Pass();

  PassType getPassType() const {
    return this->pass_type;
  }
  PassEfficiency getPassEfficiency() const {
    return this->pass_efficiency;
  }
  PassOptimizationType getPassOptimizationType() const {
    return this->pass_optimization_type;
  }
  virtual std::string getPassName() const = 0;

  virtual bool initializePass(Graph& graph) {
    return false;
  }
  virtual bool finalizePass(Graph& graph) {
    return false;
  }
  virtual PostPassAnalysis runPass(Graph& graph) = 0;

 protected:
  // Iterates through the elements in the graph and counts the number of times
  // the transform is succesfully run.
  uint DescendOnGraphAttributesAndCount(
      Node* n,
      std::function<uint(Graph&)> fn);
  // A more general version of the function above that doesn't constrain the
  // return type of fn.
  void DescendOnGraphAttributesUnconstrained(
      Node* n,
      std::function<void(Graph&)> fn);
};

class ImmutablePass : Pass {
 public:
  explicit ImmutablePass()
      : Pass(
            PassType::Immutable,
            PassEfficiency::Complete,
            PassOptimizationType::None) {}
  ~ImmutablePass() override;
};

// Pass Analysis done after a predicate based pass.
struct PostPredicateBasedPassAnalysis : PostPassAnalysis {
  Pass* pass;
  uint num_positive_transforms;
  bool initialization_done;
  bool finalization_done;

 public:
  explicit PostPredicateBasedPassAnalysis(
      Pass* pass,
      uint num_positive_transforms,
      bool initialization_done,
      bool finalization_done);

  bool graphChanged() {
    return this->num_positive_transforms > 0;
  }
  bool numSucceededTransforms() {
    return this->num_positive_transforms;
  }

  // Whether or not a repeated application of the pass might be useful.
  bool fixedPointOptimizationNeeded() {
    return this->graphChanged() &&
        pass->getPassEfficiency() == PassEfficiency::Partial;
  }
};

// A pass that is based on pattern matching. The majority of passes will
// implement this pass. In order for the pass to work the patternMatchPredicate
// function must be implemented matches a subgraph to the respective
// optimization pass. Lastly the runTransform method must also be implemented
// which simply implements the pass on any node which passes
// patternMatchPredicate.
class PredicateBasedPass : public Pass {
 public:
  explicit PredicateBasedPass(
      PassType pass_type,
      PassEfficiency pass_efficiency,
      PassOptimizationType pass_optimization_type)
      : Pass(pass_type, pass_efficiency, pass_optimization_type) {}
  ~PredicateBasedPass() override;

  virtual bool patternMatchPredicate(Node* node) = 0;
  virtual bool
  runTransform(Node* node, Graph& graph, bool& destroy_current) = 0;

  PostPassAnalysis runPass(Graph& graph) override;

 private:
  uint _runPassInternal(Graph& graph);
};

// The most general pass which allows the user to run a pass given only a graph.
class FullGraphBasedPass : public Pass {
 public:
  explicit FullGraphBasedPass(
      PassType pass_type,
      PassEfficiency pass_efficiency,
      PassOptimizationType pass_optimization_type)
      : Pass(pass_type, pass_efficiency, pass_optimization_type) {}
  ~FullGraphBasedPass() override;
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
