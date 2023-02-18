/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "ir.h"

namespace ONNX_NAMESPACE {
  void GraphBase::forEachNode(const std::function<void(Node*)>& fn) {
    forSelfAndEachSubGraph([fn](GraphBase* graph) {
      for (Node* node : graph->nodes()) {
        fn(node);
      }
    });
  }

  void GraphBase::forEachNode(const std::function<void(const Node*)>& fn) const {
    std::function<void(Node*)> tmp_fn = [fn](Node* node) { fn(node); };
    const_cast<GraphBase*>(this)->forEachNode(tmp_fn);
  }

  void GraphBase::forSelfAndEachSubGraph(const std::function<void(GraphBase*)>& fn) {
    fn(this);

    for (const Node* node : all_nodes) {
      for (const auto& attr : node->attributeNames()) {
        if (node->kindOf(attr) == AttributeKind::g) {
          std::shared_ptr<GraphBase> subgraph = node->g(attr);
          subgraph->forSelfAndEachSubGraph(fn);
        } else if (node->kindOf(attr) == AttributeKind::gs) {
          for (const auto& subgraph : node->gs(attr)) {
            subgraph->forSelfAndEachSubGraph(fn);
          }
        }
      }
    }
  }

  void GraphBase::forSelfAndEachSubGraph(const std::function<void(const GraphBase*)>& fn) const {
    std::function<void(GraphBase*)> tmp_fn = [fn](GraphBase* graph) { fn(graph); };
    const_cast<GraphBase*>(this)->forSelfAndEachSubGraph(tmp_fn);
  }

} // namespace ONNX_NAMESPACE
