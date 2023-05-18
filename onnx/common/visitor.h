// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/common/common.h"
#include "onnx/onnx_pb.h"

// A visitor base class for visiting all nodes and subgraphs in a graph.
// Currently restricted to Nodes, Graphs, and Attributes, which are
// mutually recursive.

namespace ONNX_NAMESPACE {
namespace internal {

struct Visitor {
  // The VisitX methods invoke ProcessX, and if that returns true, will
  // continue to visit all children of the X.

  // Readonly visitor methods:

  virtual void VisitGraph(const GraphProto& graph) {
    if (ProcessGraph(graph))
      for (auto& node : graph.node())
        VisitNode(node);
  }

  virtual void VisitNode(const NodeProto& node) {
    if (ProcessNode(node)) {
      for (auto& attr : node.attribute()) {
        VisitAttribute(attr);
      }
    }
  }

  virtual void VisitAttribute(const AttributeProto& attr) {
    if (ProcessAttribute(attr)) {
      if (attr.has_g()) {
        VisitGraph(attr.g());
      }
      for (auto& graph : attr.graphs())
        VisitGraph(graph);
    }
  }

  virtual bool ProcessGraph(const GraphProto& graph) {
    ONNX_UNUSED_PARAMETER(graph);
    return true;
  }

  virtual bool ProcessNode(const NodeProto& node) {
    ONNX_UNUSED_PARAMETER(node);
    return true;
  }

  virtual bool ProcessAttribute(const AttributeProto& attr) {
    ONNX_UNUSED_PARAMETER(attr);
    return true;
  }

  // Visitor methods that may update objects.

  virtual void VisitGraph(GraphProto* graph) {
    if (ProcessGraph(graph))
      for (auto& node : *(graph->mutable_node()))
        VisitNode(&node);
  }

  virtual void VisitNode(NodeProto* node) {
    if (ProcessNode(node)) {
      for (auto& attr : *(node->mutable_attribute())) {
        VisitAttribute(&attr);
      }
    }
  }

  virtual void VisitAttribute(AttributeProto* attr) {
    if (ProcessAttribute(attr)) {
      if (attr->has_g()) {
        VisitGraph(attr->mutable_g());
      }
      for (auto& graph : *(attr->mutable_graphs()))
        VisitGraph(&graph);
    }
  }

  virtual bool ProcessGraph(GraphProto* graph) {
    ONNX_UNUSED_PARAMETER(graph);
    return true;
  }

  virtual bool ProcessNode(NodeProto* node) {
    ONNX_UNUSED_PARAMETER(node);
    return true;
  }

  virtual bool ProcessAttribute(AttributeProto* attr) {
    ONNX_UNUSED_PARAMETER(attr);
    return true;
  }

  virtual ~Visitor() {}
};

} // namespace internal
} // namespace ONNX_NAMESPACE
