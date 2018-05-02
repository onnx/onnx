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
    std::unordered_set<std::string> new_initializer_names;
    // get outputs unique names
    std::vector<std::string> outputs_unique_names;
    std::transform(
        graph.outputs().begin(),
        graph.outputs().end(),
        std::back_inserter(outputs_unique_names),
        [](Value* output) { return output->uniqueName(); });
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      DescendOnGraphAttributes(n, [this](Graph& g){eliminate_unused_initializer(g);});
      // put all initializers used as input of any node to new vector
      for (auto* input : n->inputs()) {
        if (std::find(
                graph.initializer_names().begin(),
                graph.initializer_names().end(),
                input->uniqueName()) != graph.initializer_names().end()) {
          new_initializer_names.insert(input->uniqueName());
        }
      }
    }

    std::vector<std::string> removed_initializer_names;

    // get initializer names should be removed
    // if it is not used and not an output
    std::copy_if(
        graph.initializer_names().begin(),
        graph.initializer_names().end(),
        std::back_inserter(removed_initializer_names),
        [&new_initializer_names, &outputs_unique_names](std::string name) {
          return new_initializer_names.find(name) ==
              new_initializer_names.end() &&
              std::find(
                  outputs_unique_names.begin(),
                  outputs_unique_names.end(),
                  name) == outputs_unique_names.end();
        });

    // remove initializer and input if need
    for (std::string name : removed_initializer_names) {
      graph.eraseInitializer(name);
      auto iter = std::find_if(
          graph.inputs().begin(), graph.inputs().end(), [&name](Value* input) {
            return input->uniqueName() == name;
          });
      if (iter != graph.inputs().end()) {
        graph.eraseInput(std::distance(graph.inputs().begin(), iter));
      }
    }
  }

  void optimize(Graph& graph) override {
    eliminate_unused_initializer(graph);
  }
};

}} // namespace ONNX_NAMESPACE::optimization
