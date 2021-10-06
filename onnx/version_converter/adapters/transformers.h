/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cinttypes>

// Node transformers commonly used in version-adapters:

namespace ONNX_NAMESPACE {
namespace version_conversion {

inline NodeTransformerFunction RemoveAttribute(Symbol attr) {
  return [=](std::shared_ptr<Graph> graph, Node* node) {
    if (node->hasAttribute(attr)) {
      node->removeAttribute(attr);
    }
    return node;
  };
}

inline NodeTransformerFunction RemoveAttribute(Symbol attr, int64_t value) {
  return [=](std::shared_ptr<Graph> graph, Node* node) {
    if (node->hasAttribute(attr)) {
      ONNX_ASSERTM(node->i(attr) == value, "Attribute %s must have value %" PRId64, attr.toString(), value);
      node->removeAttribute(attr);
    }
    return node;
  };
}

inline NodeTransformerFunction RemoveAttributeNotEq(Symbol attr, int64_t value) {
  return [=](std::shared_ptr<Graph> graph, Node* node) {
    if (node->hasAttribute(attr)) {
      ONNX_ASSERTM(node->i(attr) != value, "Attribute %s must not have value %" PRId64, attr.toString(), value);
      node->removeAttribute(attr);
    }
    return node;
  };
}

inline NodeTransformerFunction SetAttribute(Symbol attr, int64_t value) {
  return [=](std::shared_ptr<Graph> graph, Node* node) {
    node->i_(attr, value);
    return node;
  };
}

inline NodeTransformerFunction SetAttribute(Symbol attr, const std::string& value) {
  return [=](std::shared_ptr<Graph> graph, Node* node) {
    node->s_(attr, value);
    return node;
  };
}

inline NodeTransformerFunction SetAttribute(Symbol attr, std::vector<int64_t> value) {
  return [=](std::shared_ptr<Graph> graph, Node* node) {
    std::vector<int64_t> local(value);
    node->is_(attr, std::move(local));
    return node;
  };
}

inline NodeTransformerFunction SetAttributeIfAbsent(Symbol attr, int64_t value) {
  return [=](std::shared_ptr<Graph> graph, Node* node) {
    if (!node->hasAttribute(attr)) {
      node->i_(attr, value);
    }
    return node;
  };
}

} // namespace version_conversion
} // namespace ONNX_NAMESPACE