// Version converter interface for ONNX models between different opset versions.

#pragma once

#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/common/stl_backports.h"
#include "onnx/proto_utils.h"
#include "onnx/defs/schema.h"
#include <utility>
#include <iostream>
#include <stdlib.h>
// TODO: Remove this import once actual adapters are imported
#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

// TODO: Consider creating interface for this class.
struct BaseVersionConverter {
  // Schema for adapters: {<op_name>:{<from_domain>$<from_version>:{<to_domain>
  // <to_version>: adapter}}}
  std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_map<std::string, std::unique_ptr<Adapter>>>> adapters;

  BaseVersionConverter() {}

  virtual ~BaseVersionConverter() {};

  // TODO: Adjust usage of this elsewhere
  const Adapter& adapter_lookup(Node* op,
      const OpSetID& initial_version,
      const OpSetID& target_version) const {
    // TODO: Abstract to helper?
    barf("BaseConverter does not include an implementation of adapter_lookup.  "
        "Please use a more specific converter, such as DefaultConverter.");
    throw "BaseVersionConverter Exception";
  }

  // TODO: Adjust usage elsewhere
  ONNX_NAMESPACE::ModelProto convert_version(
      const ONNX_NAMESPACE::ModelProto& mp_in,
      const OpSetID& initial_version,
      const OpSetID& target_version) const {
    barf("BaseConverter does not include an implementation of convert_version.  "
        "Please use a more specific converter, such as DefaultConverter.");
    throw "BaseVersionConverter Exception";
  };

  void registerAdapter(Adapter* a_ptr, std::string& domain) {
    OpSetID& iv = a_ptr->initial_version;
    OpSetID& tv = a_ptr->target_version;
    adapters[a_ptr->name][iv.toString()][tv.toString()] = std::unique_ptr<Adapter>(a_ptr);
  }
};

}}
