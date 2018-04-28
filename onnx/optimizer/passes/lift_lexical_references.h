#pragma once

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE { namespace optimization {

// Lift lexically-scoped references within control operators to be inputs of the
// ops themselves. This transformation yields a graph that does not conform to
// the ONNX spec.
//
// The purpose of this pass is to expose the data dependencies within control
// blocks for frameworks that use those dependencies to schedule parallel
// execution. e.g. caffe2 graph execution.
//
// Example:
// ******************************** Before *************************************
// graph test (%X[FLOAT, 5]) {
//   %Y = Identity(%X)
//   %trip_count = Constant[value = <Scalar Tensor [10]>]()
//   %condition = Constant[value = <Scalar Tensor [1]>]()
//   %Y2, %Y3 = Loop[body = <graph body_graph>](%trip_count, %condition, %)
//   return %Y, %Y2
// }
//
// graph body_graph (%i[INT32, scalar], %cond[BOOL, scalar]) {
//   %_Y2 = Identity(%X)
//   %_Y3 = Identity(%Y)
//   return %cond, %_Y2, %_Y3
// }
//
// ******************************** After **************************************
// graph test (%X[FLOAT, 5]) {
//   %Y = Identity(%X)
//   %trip_count = Constant[value = <Scalar Tensor [10]>]()
//   %condition = Constant[value = <Scalar Tensor [1]>]()
//   %Y2, %Y3 = Loop[__control_inputs = ['X', 'Y'], body = <graph body_graph>](%trip_count, %condition, %)
//                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//   return %Y, %Y2
// }
//
// graph body_graph (%i[INT32, scalar], %cond[BOOL, scalar]) {
//   %_Y2 = Identity(%X)
//   %_Y3 = Identity(%Y)
//   return %cond, %_Y2, %_Y3
// }
//
// ******************************** Continue Docs*******************************
//
// The algorithm is roughly:
//  symbol_table_stack = empty stack of symbol tables
//
//  liftreferences(graph)
//      -> a set of unresolved reference strings:
//    unresolved_references = {}
//
//    symbol_table_stack.push(new symbol table containing inputs for this sub-graph)
//    for each node in the graph:
//      for input in node.inputs:
//        if input is not in this frame:
//          unresolved_references.insert(input)
//      if node is a control flow operator:
//        for each sub-graph g:
//          refs = liftreferences(g)
//          for each ref in refs:
//            if ref is in this frame:
//              insert ref as an input to node
//            else:
//              unresolved_references.insert(ref)
//        for output in node.outputs:
//          symbol_table_stack.top()[output] = Value*
//    return unresolved_references
struct LiftLexicalReferences : public OptimizePass {
  explicit LiftLexicalReferences()
    : OptimizePass("lift_lexical_references", API_TYPE::IR) {
  }

  using ValueTable = std::unordered_map<std::string, Value*>;
  using EnvStack = std::vector<ValueTable>;

  std::set<std::string> liftReferences(Graph* g, EnvStack *es) {
    std::set<std::string> unresolved_references;
    es->push_back(ValueTable());
    for (auto &inp : g->inputs()) {
      es->back()[inp->uniqueName()] = inp;
    }

    for (auto *n : g->nodes()) {
      // Skip optional input/captured value node.
      if (n->kind() == ONNX_NAMESPACE::kUndefined ||
            n->kind() == ONNX_NAMESPACE::kCaptured) {
        continue;
      }
      for (auto *inp : n->inputs()) {
        // Empty string is 0-input variadic argument. Skip that one.
        if (!inp->uniqueName().empty() && !es->back().count(inp->uniqueName())) {
          unresolved_references.insert(inp->uniqueName());
        }
      }

      std::set<std::string> local_unresolved;
      if (n->kind() == ONNX_NAMESPACE::kLoop) {
        auto *body_graph = n->g(ONNX_NAMESPACE::kbody).get();
        local_unresolved = liftReferences(body_graph, es);
      } else if (n->kind() == ONNX_NAMESPACE::kIf) {
        auto *then_graph = n->g(ONNX_NAMESPACE::kthen_branch).get();
        auto then_unresolved = liftReferences(then_graph, es);
        local_unresolved.insert(then_unresolved.begin(), then_unresolved.end());
        auto *else_graph = n->g(ONNX_NAMESPACE::kelse_branch).get();
        auto else_unresolved = liftReferences(else_graph, es);
        local_unresolved.insert(else_unresolved.begin(), else_unresolved.end());
      }

      std::vector<std::string> control_inputs;
      for (auto &unresolved : local_unresolved) {
        if (es->back().count(unresolved)) {
          control_inputs.push_back(unresolved);
        } else {
          unresolved_references.insert(unresolved);
        }
      }

      // Create this attribute so the backend knows how many of these inputs
      // are simply there for control dependencies
      if (!control_inputs.empty()) {
        n->ss_(ONNX_NAMESPACE::k__control_inputs, std::move(control_inputs));
      }

      for (auto *out : n->outputs()) {
        es->back()[out->uniqueName()] = out;
      }
    }

    es->pop_back();
    return unresolved_references;
  }


  void optimize(Graph& graph) override {
    EnvStack es;
    auto unresolved = liftReferences(&graph, &es);

    if (unresolved.size()) {
      std::string errmsg = "Unresolved value references: ";
      for (auto& ref : unresolved) {
        errmsg += ref + ",";
      }
      throw std::runtime_error(errmsg);
    }
  }
};

}} // namespace ONNX_NAMESPACE::optimization
