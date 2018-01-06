#include "onnx/optimizer/optimize.h"

#include <queue>

namespace onnx { namespace optimization {

const char* impure_operators[] = {
  "RandomNormal",
  "RandomNormalLike",
  "RandomUniform",
  "RandomUniformLike"
};

bool is_pure_operator(Node * n) {
  for (auto x : impure_operators) {
    if (n->kind() == stringToSymbol(x)) {
      return false;
    }
  }
  return true;
}

bool is_nop_transpose(const std::vector<int64_t> & perm) {
  for (size_t i = 0; i < perm.size(); i++)
    if (perm[i] != i)
      return false;
  return true;
}

// returns a vector `ret` such that transposing by `ret` is equivalent
// to transposing by `t1` and then by `t2`
std::vector<int64_t> compose_transposes(const std::vector<int64_t> & t1,
                                        const std::vector<int64_t> & t2) {
  JIT_ASSERT(t1.size() == t2.size());
  std::vector<int64_t> ret;
  for (size_t i = 0; i < t1.size(); i++) {
    JIT_ASSERT(   t1[i]  < t2.size());
    JIT_ASSERT(t2[t1[i]] < t2.size());
    ret.push_back(t2[t1[i]]);
  }
  return ret;
}

void fuse_consecutive_transposes(std::shared_ptr<Graph>& g) {
  for (auto it = g->begin(); it != g->end(); ++it) {
    auto* n = *it;

    if (n->kind() == kTranspose && n->input()->node()->kind() == kTranspose) {
      auto origInput = n->input();
      n->is_(kperm, compose_transposes(origInput->node()->is(kperm), n->is(kperm)));
      n->replaceInput(0, origInput->node()->input());
      if (origInput->uses().size() == 0) {
        origInput->node()->destroy();
      }
      continue;
    }
  }
}

void eliminate_nop_transpose(std::shared_ptr<Graph>& graph) {
  for (auto it = graph->begin(); it != graph->end(); ++it) {
    auto* n = *it;

    if (n->kind() == kTranspose) {
      if (is_nop_transpose(n->is(kperm))) {
        n->replaceAllUsesWith(n->input()->node());
        it.destroyCurrent();
        continue;
      }
    }
  }
}

void fuse_transpose_into_gemm(std::shared_ptr<Graph>& graph) {
  static const std::vector<int64_t> simple_trans_perm({1,0});

  for (auto it = graph->begin(); it != graph->end(); ++it) {
    auto* n = *it;

    if (n->kind() == kGemm) {
      for (size_t i : {0,1}) {
        auto inp = n->inputs()[i];
        auto trans = i == 0 ? ktransA : ktransB;
        if (inp->node()->kind() == kTranspose && inp->node()->is(kperm) == simple_trans_perm) {
          n->replaceInput(i, inp->node()->input());
          n->i_(trans, n->hasAttribute(trans) ? !n->i(trans) : 1);
          if (inp->uses().size() == 0) {
            inp->node()->destroy();
          }
        }
      }
    }
  }
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
void split_init_and_predict(std::shared_ptr<Graph> g, bool init, bool predict) {
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
      g->initializer_names().begin(),
      g->initializer_names().end());

    for (Value * v : g->inputs()) {
      if (initializer_names.count(v->uniqueName()) == 0) {
        predict_net_values.insert(v);
      }
    }
  }

  for (Node * n : g->nodes()) {
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
  for (Node * n : g->nodes()) {
    if (node_belongs_to_predict_net(n)) {
      for (Value * v : n->inputs()) {
        if (!value_belongs_to_predict_net(v)) {
          new_interface.insert(v);
        }
      }
    }
  }

  if (init) {
    // Add new outputs corresponding to the boundary between init and
    // predict nets, ensuring that we don't duplicate outputs.
    for (Value * v : g->outputs()) {
      new_interface.erase(v);
    }
    for (Value * v : new_interface) {
      g->registerOutput(v);
    }

    // Remove outputs that belong to the predict net.
    for (auto i = g->outputs().size(); i--;) {
      if (value_belongs_to_predict_net(g->outputs()[i])) {
        g->return_node()->removeInput(i);
      }
    }

    // Delete nodes that belong to the predict net, in reverse
    // topological order.
    for (auto it = g->nodes().rbegin(); it != g->nodes().rend(); it++) {
      if (node_belongs_to_predict_net(*it)) {
        it.destroyCurrent();
      }
    }

    // Remove inputs that belong to the predict net.
    for (auto i = g->inputs().size(); i--;) {
      if (value_belongs_to_predict_net(g->inputs()[i])) {
        g->eraseInput(i);
      }
    }
  } else if (predict) {
    // Add new inputs, ensuring that we don't introduce duplicates.
    // Also cut the boundary between init and predict net by replacing
    // the Values along the boundary with replaceAllUsesWith.
    for (Value * v : g->inputs()) {
      new_interface.erase(v);
    }
    for (Value * v : new_interface) {
      Value * newv = g->addInput()->copyMetadata(v);
      v->replaceAllUsesWith(newv);
    }

    // Delete nodes that aren't in the predict net, in reverse
    // topological order.
    for (auto it = g->nodes().rbegin(); it != g->nodes().rend(); it++) {
      if (!node_belongs_to_predict_net(*it)) {
        it.destroyCurrent();
      }
    }

    // Remove inputs that aren't used by the predict net.
    for (auto i = g->inputs().size(); i--;) {
      if (g->inputs()[i]->uses().empty()) {
        g->eraseInput(i);
      }
    }
  }
}

void optimize(std::shared_ptr<Graph> g, bool init, bool predict) {
  fuse_consecutive_transposes(g);
  eliminate_nop_transpose(g);
  fuse_transpose_into_gemm(g);
  if (init || predict) {
    // Only 
    split_init_and_predict(g, init, predict);
  }
}

}}
