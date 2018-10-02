#include "onnx/optimizer/pass.h"
#include "onnx/common/assertions.h"

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
Pass::~Pass() {}

unsigned int Pass::DescendOnGraphAttributesAndCount(
    Node* n,
    std::function<unsigned int(Graph&)> fn) {
  unsigned int num_changes = 0;
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

void Pass::DescendOnGraphAttributesUnconstrained(
    Node* n,
    std::function<void(Graph&)> fn) {
  for (auto name : n->attributeNames()) {
    auto kind = n->kindOf(name);
    if (kind == AttributeKind::g) {
      fn(*n->g(name));
    }
    if (kind == AttributeKind::gs) {
      for (auto& g : n->gs(name)) {
        fn(*g);
      }
    }
  }
}
PredicateBasedPass::~PredicateBasedPass() {}
unsigned int PredicateBasedPass::_runPassInternal(Graph& graph) {
  unsigned int num_changes = false;
  for (auto it = graph.begin(); it != graph.end(); ++it) {
    auto* n = *it;
    num_changes += this->DescendOnGraphAttributesAndCount(
        n, [this](Graph& g) { return _runPassInternal(g); });

    if (this->patternMatchPredicate(n)) {
      bool destroy_current = false;
      num_changes += this->runTransform(n, graph, destroy_current);

      if (destroy_current) {
        it.destroyCurrent();
        try {
          it.destroyCurrent();
        } catch (const assert_error error) {
          continue;
        }
      }
    }
  }
  return num_changes;
}

PostPassAnalysis PredicateBasedPass::runPass(Graph& graph) {
  bool initialized_pass = this->initializePass(graph);
  unsigned int touched_optimizations = this->_runPassInternal(graph);
  bool finalized_pass = this->finalizePass(graph);

  return PostPredicateBasedPassAnalysis(
      this, touched_optimizations, initialized_pass, finalized_pass);
}

PostPredicateBasedPassAnalysis::PostPredicateBasedPassAnalysis(
    Pass* pass,
    unsigned int num_positive_transforms,
    bool initialization_done,
    bool finalization_done) {
  this->pass = pass;
  this->num_positive_transforms = num_positive_transforms;
  this->initialization_done = initialization_done;
  this->finalization_done = finalization_done;
}

FullGraphBasedPass::~FullGraphBasedPass() {}

} // namespace optimization
} // namespace ONNX_NAMESPACE
