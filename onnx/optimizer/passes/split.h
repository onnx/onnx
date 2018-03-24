// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE { namespace optimization {

static const char* impure_operators[] = {
  "RandomNormal",
  "RandomNormalLike",
  "RandomUniform",
  "RandomUniformLike"
};

static bool is_pure_operator(Node * n) {
  for (auto x : impure_operators) {
    if (n->kind() == Symbol(x)) {
      return false;
    }
  }
  return true;
}

// Split the graph into 'init' and 'predict' nets. This is kind of
// like constant folding, except that rather than actually execute the
// constant computations, we simply split them out into a separate
// graph. Nodes that have any transitive dependency on the
// initializers, or on impure operators, must remain in the predict
// net. All others may be moved to the init net.
//
// This function destructively mutates the graph into either the init
// or the predict net. If you want both, which you probably do,
// arrange to call it twice.
//
// NOTE POTENTIAL BREAKAGE:
//
// The ONNX spec provides no guarantees about "staging", i.e. which
// inputs change on every invocation vs which generally stay the same.
// Here we make the assumption that inputs which have an initializer
// value provided for them vary only between invocations of the init
// net, and are constant across runs of the predict net.
//
static void split_init_and_predict(Graph& graph, bool init, bool predict) {
  // The first step is to identify which Values are reachable from
  // either of
  //   - inputs without corresponding initializers
  //   - impure operators
  // Any such Values belong to the predict net. Nodes belong to the
  // predict net if they are impure or if any of their inputs do.

  std::unordered_set<Value *> predict_net_values;

  auto value_belongs_to_predict_net = [&](Value * v) {
    return predict_net_values.count(v) > 0;
  };
  auto node_belongs_to_predict_net = [&](Node * n) {
    return !is_pure_operator(n) ||
        std::any_of(n->inputs().begin(),
                    n->inputs().end(),
                    value_belongs_to_predict_net);
  };

  {
    std::unordered_set<std::string> initializer_names(
      graph.initializer_names().begin(),
      graph.initializer_names().end());

    for (Value * v : graph.inputs()) {
      if (initializer_names.count(v->uniqueName()) == 0) {
        predict_net_values.insert(v);
      }
    }
  }

  for (Node * n : graph.nodes()) {
    if (node_belongs_to_predict_net(n)) {
      for (Value * v : n->outputs()) {
        predict_net_values.insert(v);
      }
    }
  }

  // Any Value which is not itself in the predict net, but which
  // is used by a Node which is, becomes an output of the init
  // graph and an input of the predict net
  std::unordered_set<Value *> new_interface;
  for (Node * n : graph.nodes()) {
    if (node_belongs_to_predict_net(n)) {
      for (Value * v : n->inputs()) {
        if (!value_belongs_to_predict_net(v)) {
          new_interface.insert(v);
        }
      }
    }
  }

  for (Value * v : graph.outputs()) {
    if (!value_belongs_to_predict_net(v)) {
      new_interface.insert(v);
    }
  }

  if (init) {
    // Add new outputs corresponding to the boundary between init and
    // predict nets, ensuring that we don't duplicate outputs.
    for (Value * v : graph.outputs()) {
      new_interface.erase(v);
    }
    for (Value * v : new_interface) {
      if (v->node()->kind() == kUndefined) {
        continue;
      }
      graph.registerOutput(v);
    }

    // Remove outputs that belong to the predict net.
    for (auto i = graph.outputs().size(); i--;) {
      if (value_belongs_to_predict_net(graph.outputs()[i])) {
        graph.return_node()->removeInput(i);
      }
    }

    // Delete nodes that belong to the predict net, in reverse
    // topological order.
    for (auto it = graph.nodes().rbegin(); it != graph.nodes().rend(); it++) {
      if (node_belongs_to_predict_net(*it)) {
        it.destroyCurrent();
      }
    }

    // Remove inputs that belong to the predict net.
    for (auto i = graph.inputs().size(); i--;) {
      if (value_belongs_to_predict_net(graph.inputs()[i])) {
        graph.eraseInput(i);
      }
    }
  } else if (predict) {
    // When creating the predict net, 'undefined' nodes will
    // naturally go into the init net. We need to have a place to
    // copy the ones we want to keep in the predict net.
    auto * optionalInputDummyNode = graph.create(kUndefined, 1);
    graph.appendNode(optionalInputDummyNode);
    optionalInputDummyNode->outputs()[0]->setUniqueName("");

    // Add new inputs, ensuring that we don't introduce duplicates.
    // Also cut the boundary between init and predict net by replacing
    // the Values along the boundary with replaceAllUsesWith.
    for (Value * v : graph.inputs()) {
      new_interface.erase(v);
    }
    for (Value * v : new_interface) {
      if (v->node()->kind() == kUndefined) {
        v->replaceAllUsesWith(optionalInputDummyNode->outputs()[0]);
      } else {
        Value * newv = graph.addInput()->copyMetadata(v);
        v->replaceAllUsesWith(newv);
      }
    }

    // Delete nodes that aren't in the predict net, in reverse
    // topological order.
    for (auto it = graph.nodes().rbegin(); it != graph.nodes().rend(); it++) {
      if (*it == optionalInputDummyNode) {
        continue;
      }
      if (node_belongs_to_predict_net(*it)) {
        continue;
      }
      it.destroyCurrent();
    }

    // Remove inputs that aren't used by the predict net.
    for (auto i = graph.inputs().size(); i--;) {
      if (graph.inputs()[i]->uses().empty()) {
        graph.eraseInput(i);
      }
    }

    // Remove all initializers, they are already in the init net.
    graph.clearInitializers();
  }
}

struct SplitInit : public OptimizePass {
  explicit SplitInit()
    : OptimizePass("split_init", API_TYPE::IR) {
  }

  virtual void optimize(Graph& graph) {
    split_init_and_predict(graph, true, false);
  }
};

struct SplitPredict : public OptimizePass {
  explicit SplitPredict()
    : OptimizePass("split_predict", API_TYPE::IR) {
  }

  virtual void optimize(Graph& graph) {
    split_init_and_predict(graph, false, true);
  }
};

}} // namespace ONNX_NAMESPACE::optimization
