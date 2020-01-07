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
      NodeDestroyType destroy_type = NodeDestroyType::DestroyZero;
      num_changes += this->runTransform(n, graph, destroy_type);

      if (destroy_type == NodeDestroyType::DestroyOne) {
        it.destroyCurrent();
      }
      if (destroy_type == NodeDestroyType::DestroyTwo) {
        it.destroyCurrent();
        it.destroyCurrent();
      }
    }
  }
  return num_changes;
}

PassAnalysisType PredicateBasedPass::getPassAnalysisType() const {
  return PassAnalysisType::CountBased;
}

std::shared_ptr<PostPassAnalysis> PredicateBasedPass::runPass(Graph& graph) {
  bool initialized_pass = this->initializePass(graph);
  unsigned int touched_optimizations = this->_runPassInternal(graph);
  bool finalized_pass = this->finalizePass(graph);

  return std::shared_ptr<PostPassAnalysis>(new CountBasedPassAnalysis(
      this, touched_optimizations, initialized_pass, finalized_pass));
}

CountBasedPassAnalysis::CountBasedPassAnalysis(
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
