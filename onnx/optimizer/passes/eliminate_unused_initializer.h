// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE { namespace optimization {

struct EliminateUnusedInitializer final : public OptimizePass {
  explicit EliminateUnusedInitializer()
    : OptimizePass("eliminate_unused_initializer", API_TYPE::IR) {
  }

  void eliminate_unused_initializer(Graph& graph) {
    std::vector<Tensor> new_initializers;
    std::vector<std::string> new_initializer_names;
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      DescendOnGraphAttributes(n, [this](Graph& g){eliminate_unused_initializer(g);});
      // put any initializer used as input of any node to new vector
      for (auto* input : n->inputs()) {
        if (std::find(
                graph.initializer_names().begin(),
                graph.initializer_names().end(),
                input->uniqueName()) != graph.initializer_names().end()) {
          Tensor initializer;
          for (auto _init : graph.initializers()) {
            if (_init.name() == input->uniqueName()) {
              initializer = _init;
            }
          }
          new_initializers.push_back(initializer);
          new_initializer_names.push_back(input->uniqueName());
        }
      }
    }
    // get diff between old and new initializers vector
    std::vector<std::string> diff_initializer_names, removed_initializer_names;
    std::set_difference(
        graph.initializer_names().begin(),
        graph.initializer_names().end(),
        new_initializer_names.begin(),
        new_initializer_names.end(),
        std::inserter(diff_initializer_names, diff_initializer_names.begin()));
    // get removed initializer names if it is not an output
    std::copy_if(
        diff_initializer_names.begin(),
        diff_initializer_names.end(),
        std::back_inserter(removed_initializer_names),
        [&graph](std::string name) {
          for (auto* output : graph.outputs()) {
            if (output->uniqueName() == name) {
              return false;
            }
          }
          return true;
        });
    // re-add initializer to graph
    if (!removed_initializer_names.empty()) {
      graph.clearInitializers();
      for (auto initializer : new_initializers) {
        graph.addInitializer(initializer, initializer.name());
      }
      for (size_t i = 0; i < graph.inputs().size(); i++) {
        for (auto name : removed_initializer_names) {
          if (graph.inputs()[i]->uniqueName() == name) {
            graph.eraseInput(i);
          }
        }
      }
    }
  }

  void optimize(Graph& graph) override {
    eliminate_unused_initializer(graph);
  }
};

}} // namespace ONNX_NAMESPACE::optimization
