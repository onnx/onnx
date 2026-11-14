// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "onnx/checker.h"
#include "onnx/common/scoped_resource.h"
#include "onnx/common/visitor.h"

namespace ONNX_NAMESPACE::internal { // internal/private API

using AttributeMap = std::unordered_map<std::string, const AttributeProto*>;
using AttributeSet = std::unordered_set<const AttributeProto*>;

// Class for binding formal attribute-parameters (in a node or graph) to their values.

class AttributeBinder : public MutableVisitor {
 public:
  using CallSiteBoundCallback = std::function<void(const AttributeProto&, AttributeProto&)>;

  explicit AttributeBinder(const AttributeMap& attr_map) : attr_map_(attr_map) {}

  AttributeBinder(
      const AttributeMap& attr_map,
      const AttributeSet& default_attrs,
      CallSiteBoundCallback on_callsite_bound = {})
      : attr_map_(attr_map), default_attrs_(&default_attrs), on_callsite_bound_(std::move(on_callsite_bound)) {}

  // Binding a formal attribute-parameter to a value may, as a special case, also
  // remove the attribute from the list of attributes of a node (when the attribute
  // has no specified value). Hence, we need to do the processing at a Node level
  // rather than an attribute level.
  void VisitNode(NodeProto& node) override {
    auto& attributes = *node.mutable_attribute();
    for (auto attr_iter = attributes.begin(); attr_iter != attributes.end();) {
      auto& attr = *attr_iter;
      if (!attr.ref_attr_name().empty()) {
        // Attribute-references must be replaced by their bound values.
        // If no call-site value or default is available, the attribute must be removed.
        auto it = attr_map_.find(attr.ref_attr_name());
        if (it != attr_map_.end()) {
          const AttributeProto* replacement = it->second;
          const bool is_default = (default_attrs_ != nullptr) && (default_attrs_->count(replacement) > 0);
          if (is_default && !active_defaults_.insert(replacement).second) {
            fail_check("Cycle detected in function attribute defaults for '", attr.ref_attr_name(), "'.");
          }
          ScopeExit guard([&]() noexcept {
            if (is_default) {
              active_defaults_.erase(replacement);
            }
          });
          // Copy value of attribute, but retain original name:
          std::string name = attr.name();
          attr = *replacement;
          attr.set_name(name);
          // Default graphs belong to the callee's attribute scope and must have their
          // inner references bound; call-site graphs belong to the caller's scope.
          if (is_default) {
            VisitAttribute(attr);
          } else if (on_callsite_bound_) {
            on_callsite_bound_(*replacement, attr);
          }
          ++attr_iter;
        } else if (remove_unbound_) {
          attr_iter = attributes.erase(attr_iter);
        } else {
          ++attr_iter;
        }
      } else {
        // For regular attributes, we process subgraphs, if present, recursively.
        VisitAttribute(attr);
        ++attr_iter;
      }
    }
  }

  // Expands only non-overridden function attribute defaults, leaving call-site and missing
  // attribute references intact so callee-scope renaming can run before splicing caller graphs.
  static void BindDefaultAttributes(const NodeProto& callnode, FunctionProto& callee) {
    std::unordered_set<std::string> callsite_names;
    for (const auto& attr : callnode.attribute()) {
      callsite_names.insert(attr.name());
    }
    AttributeMap defaults_map;
    AttributeSet defaults;
    for (const auto& attr : callee.attribute_proto()) {
      if (callsite_names.count(attr.name()) == 0) {
        defaults_map[attr.name()] = &attr;
        defaults.insert(&attr);
      }
    }
    AttributeBinder attr_binder(defaults_map, &defaults, false, {});
    attr_binder.VisitFunction(callee);
  }

  // Binds remaining attribute references from call-site arguments and removes omitted ones.
  static void BindCallSiteAttributes(
      const NodeProto& callnode,
      FunctionProto& callee,
      const std::function<void(AttributeProto&)>& on_bound) {
    AttributeMap callsite_map;
    for (const auto& attr : callnode.attribute()) {
      callsite_map[attr.name()] = &attr;
    }
    CallSiteBoundCallback callback;
    if (on_bound) {
      callback = [&on_bound](const AttributeProto& /*source*/, AttributeProto& bound) { on_bound(bound); };
    }
    AttributeBinder attr_binder(callsite_map, nullptr, true, std::move(callback));
    attr_binder.VisitFunction(callee);
  }

  // Updates a FunctionProto by replacing all attribute-references with the corresponding
  // attribute-values in the call-node or their function defaults. Otherwise, the attribute is removed.
  static void BindAttributes(const NodeProto& callnode, FunctionProto& callee) {
    AttributeMap map;
    AttributeSet defaults;
    for (const auto& attr : callee.attribute_proto()) {
      map[attr.name()] = &attr;
      defaults.insert(&attr);
    }
    for (const auto& attr : callnode.attribute()) {
      map[attr.name()] = &attr;
    }
    AttributeBinder attr_binder(map, defaults);
    attr_binder.VisitFunction(callee);
  }

 private:
  AttributeBinder(
      const AttributeMap& attr_map,
      const AttributeSet* default_attrs,
      bool remove_unbound,
      CallSiteBoundCallback on_callsite_bound)
      : attr_map_(attr_map),
        default_attrs_(default_attrs),
        remove_unbound_(remove_unbound),
        on_callsite_bound_(std::move(on_callsite_bound)) {}

  const AttributeMap& attr_map_;
  const AttributeSet* default_attrs_ = nullptr;
  bool remove_unbound_ = true;
  CallSiteBoundCallback on_callsite_bound_;
  AttributeSet active_defaults_;
};

} // namespace ONNX_NAMESPACE::internal
