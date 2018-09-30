#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

Pass::Pass(
    PassType pass_type,
    PassEfficiency pass_efficiency,
    PassOptimizationType pass_optimization_type) {
  this->pass_type = pass_type;
  this->pass_efficiency = pass_efficiency;
  this->pass_optimization_type = pass_optimization_type;
}
uint Pass::DescendOnGraphAttributes(Node* n, std::function<uint(Graph&)> fn) {
  uint num_changes = 0;
  for (auto name : n->attributeNames()) {
    auto kind = n->kindOf(name);
    if (kind == AttributeKind::g) {
      num_changes += fn(*n->g(name));
    }
    if (kind == AttributeKind::gs) {
      for (auto& g : n->gs(name)) {
        num_changes += fn(*g);
      }
    }
  }
  return num_changes;
}

uint PredicateBasedPass::_runPassInternal(Graph& graph) {
  uint num_changes = false;
  for (auto it = graph.begin(); it != graph.end(); ++it) {
    auto* n = *it;
    num_changes += this->DescendOnGraphAttributes(
        n, [this](Graph& g) { return _runPassInternal(g); });

    if (this->patternMatchPredicate(n)) {
      bool destroy_current = false;
      num_changes += this->runTransform(n, destroy_current);

      if (destroy_current) {
        it.destroyCurrent();
        it.destroyCurrent();
      }
    }
  }
  return num_changes;
}

PostPassAnalysis PredicateBasedPass::runPass(Graph& graph) {
  bool initialized_pass = this->initializePass(graph);
  uint touched_optimizations = this->_runPassInternal(graph);
  bool finalized_pass = this->finalizePass(graph);

  return PostPredicateBasedPassAnalysis(
      this, touched_optimizations, initialized_pass, finalized_pass);
}

PostPredicateBasedPassAnalysis::PostPredicateBasedPassAnalysis(
    Pass* pass,
    uint num_positive_transforms,
    bool succesful_initialization,
    bool succesful_finalization) {
  this->pass = pass;
  this->num_positive_transforms = num_positive_transforms;
  this->succesful_initialization = succesful_initialization;
  this->succesful_finalization = succesful_finalization;
}
} // namespace optimization
} // namespace ONNX_NAMESPACE
