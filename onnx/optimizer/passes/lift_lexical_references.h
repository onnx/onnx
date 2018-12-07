#pragma once

#include <set>
#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

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
//   %Y2, %Y3 = Loop[__control_inputs = ['X', 'Y'], body = <graph
//   body_graph>](%trip_count, %condition, %)
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
//    symbol_table_stack.push(new symbol table containing inputs for this
//    sub-graph) for each node in the graph:
//      for input in node.inputs:
//        if input is not in this frame:
//          unresolved_references.insert(input)
//      if node is a control flow operator:
//        for each sub-graph g:
//          for each output in g's body:
//            if output is defined in current scope:
//              control_inputs.insert(output)
//          refs = liftreferences(g)
//          for each ref in refs:
//            if ref is in this frame or any parent frame (control_inputs):
//              control_inputs.insert(ref)
//            else:
//              unresolved_references.insert(ref)
//          set the control inputs attribute to the node
//        for output in node.outputs:
//          symbol_table_stack.top()[output] = Value*
//    return unresolved_references
struct LiftLexicalReferences : public FullGraphBasedPass {
  explicit LiftLexicalReferences()
      : FullGraphBasedPass(
            PassType::Seperate,
            PassEfficiency::Complete,
            PassOptimizationType::Memory) {}

  std::string getPassName() const override {
    return "lift_lexical_references";
  }
  PassAnalysisType getPassAnalysisType() const override {
    return PassAnalysisType::Empty;
  }

  using ValueTable = std::unordered_map<std::string, Value*>;

  // Environment stack, please to store value table and
  // controlled inputs
  struct Environment {
    Environment(std::shared_ptr<Environment> next = nullptr) : next(next) {}

    std::shared_ptr<Environment> next;

    Value* findInThisFrame(const std::string& name) {
      auto it = value_table.find(name);
      if (it != value_table.end()) {
        return it->second;
      }
      return nullptr;
    }

    Value* findInParentFrame(const std::string& name) {
      return next ? next->findInAnyFrame(name) : nullptr;
    }

    Value* findInAnyFrame(const std::string& name) {
      for (auto runner = this; runner; runner = runner->next.get()) {
        if (auto r = runner->findInThisFrame(name)) {
          return r;
        }
      }
      return nullptr;
    }

    void setVar(const std::string& name, Value* value) {
      value_table[name] = value;
    }

   private:
    ValueTable value_table;
  };

  std::shared_ptr<Environment> environment_stack;

  // environment stack helper
  void pushFrame() {
    environment_stack = std::make_shared<Environment>(environment_stack);
  }

  std::shared_ptr<Environment> popFrame() {
    auto old_frame = environment_stack;
    environment_stack = environment_stack->next;
    return old_frame;
  }

  std::set<std::string> liftReferences(Graph* g) {
    std::set<std::string> unresolved_references;
    pushFrame();
    for (auto& inp : g->inputs()) {
      environment_stack->setVar(inp->uniqueName(), inp);
    }

    for (auto* n : g->nodes()) {
      // Skip optional input/captured value node.
      if (n->kind() == ONNX_NAMESPACE::kUndefined ||
          n->kind() == ONNX_NAMESPACE::kCaptured) {
        continue;
      }
      for (auto* inp : n->inputs()) {
        // Empty string is 0-input variadic argument. Skip that one.
        if (!inp->uniqueName().empty() &&
            !environment_stack->findInThisFrame(inp->uniqueName())) {
          unresolved_references.insert(inp->uniqueName());
        }
      }

      std::set<std::string> local_unresolved;

      // if a graph body output has already already been emitted outside of the
      // subgraph scope, then it must be added as an input to the subgraph
      auto add_subgraph_outputs = [&](Graph* body_graph) {
        for (auto* out : body_graph->outputs()) {
          if (environment_stack->findInAnyFrame(out->uniqueName())) {
            local_unresolved.insert(out->uniqueName());
          }
        }
      };

      if (n->kind() == ONNX_NAMESPACE::kLoop) {
        auto* body_graph = n->g(ONNX_NAMESPACE::kbody).get();
        local_unresolved = liftReferences(body_graph);
        add_subgraph_outputs(body_graph);
      } else if (n->kind() == ONNX_NAMESPACE::kIf) {
        auto* then_graph = n->g(ONNX_NAMESPACE::kthen_branch).get();
        add_subgraph_outputs(then_graph);
        auto then_unresolved = liftReferences(then_graph);
        local_unresolved.insert(then_unresolved.begin(), then_unresolved.end());
        auto* else_graph = n->g(ONNX_NAMESPACE::kelse_branch).get();
        add_subgraph_outputs(else_graph);
        auto else_unresolved = liftReferences(else_graph);
        local_unresolved.insert(else_unresolved.begin(), else_unresolved.end());
      }

      std::vector<std::string> control_inputs;
      for (auto& unresolved : local_unresolved) {
        if (environment_stack->findInAnyFrame(unresolved)) {
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

      for (auto* out : n->outputs()) {
        environment_stack->setVar(out->uniqueName(), out);
      }
    }

    popFrame();
    return unresolved_references;
  }

  std::shared_ptr<PostPassAnalysis> runPass(Graph& graph) override {
    auto unresolved = liftReferences(&graph);

    if (unresolved.size()) {
      std::string errmsg = "Unresolved value references: ";
      for (auto& ref : unresolved) {
        errmsg += ref + ",";
      }
      throw std::runtime_error(errmsg);
    }
    return std::shared_ptr<PostPassAnalysis>(new PostPassAnalysis());
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
