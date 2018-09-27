#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

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
      num_changes += this->runTransform(n);
    }
  }
  return num_changes;
}

PostPassAnalysis PredicateBasedPass::runPass(Graph& graph) {
  auto initialized_pass = this->initializePass(graph);
  uint touched_optimizations = this->_runPassInternal(graph);
  auto finalized_pass = this->finalizePass(graph);

  return PostPredictBasedPassAnalysis(
      this, touched_optimizations, initialized_pass, finalized_pass);
}

} // namespace optimization
} // namespace ONNX_NAMESPACE
