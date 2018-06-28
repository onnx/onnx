// A converter for ONNX models between different opset versions

#pragma once

#include "onnx/common/ir.h"
#include "onnx/proto_utils.h"
#include "onnx/defs/schema.h"
#include <utility>
#include "onnx/common/graph_node_list.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct VersionConverter {
  VersionConverter() {
    // TODO: Register adapters to the version converter
  }

  virtual ~Converter() = default;

  ONNX_NAMESPACE::ModelProto convert_version(
      const ONNX_NAMESPACE::ModelProto& mp_in,
      const OpSetID target_version) {
    std::shared_ptr<ONNX_NAMESPACE::Graph> g(ONNX_NAMESPACE::ImportModelProto(mp_in));

    if (g.get() == nullptr) {
      std::cerr << "Warning: onnx optimizer is unable to parse input model. "
        << "(The IR version of the ONNX model may be too old.)" << std::endl;
      // If we can't parse the file, just return the input.
      return mp_in;
    }

    // Get initial model version
    OpSetID initial_version = g.opset_version;

    if (initial_version == target_version) {
      return mp_in;
    }

    std::string domain = initial_version.domain;

    // Check if target_version is valid
    std::pair version_range = OpSchemaRegistry::DomainToVersionRange::Instance().Map()[domain];
    if (target_version.version < version_range.first || target_version.version > version_range.second) {
      // Invalid target_version
      std::cerr << "Warning: invalid target_version (must be between "
        << version_range.first << " and " << version_range.second << std::endl;
      return mp_in;
    }

    // Compile list of all ops used in the model
    graph_node_list nodes = g.nodes();

    // Use OpName_Domain_Version_Schema_Map from schema.h to generate map from IR Nodes to OpSchema (particularly for OpSetID)
    std::unordered_map<Node, OpSchema> current_opschemas;
    for (Node& op : nodes) {
      // Iterate through all OperatorSetVersions, select highest that is leq initial_version
      OperatorSetVersion op_opset_version = 0;
      // TODO: Check whether this process accidentally always defaults to initial_version
      // TODO: If so, just take the SinceVersion of this schema (which returns the implementation version)
      for (const auto& version_pair : ONNX_NAMESPACE::OpName_Domain_Version_Schema_Map[*(op->name())][domain]) {
        if (version_pair.first > op_opset_version && version_pair.first <= initial_version.version) {
          op_opset_version = version_pair.first;
        }
      }
      current_opschemas[*op] = ONNX_NAMESPACE::OpName_Domain_Version_Schema_Map[*(op->name())][domain][op_opset_version];
    }

    // Iterate over all versions to target_version
    int curr_version = initial_version.version;
    int next_version;
    if (target_version > initial_version) {
      curr_version++;
      next_version = curr_version + 1;
    } else {
      next_version = curr_version - 1;
    }
    while (curr_version != target_version.version) {
      // TODO: Iterate through and call adapter returned by adapter_lookup for ops from current_version opset
      // TODO: Use optimize procedure of ExportModelProto(&mp_out, g), adapter->adapt(mp_out), ImportModelProto(mp_out)
      //  This is because we always process ModelProtos, rather than separate IRs (though we operate on the IR)
      for (Node& op : nodes) {
        if (current_opschemas[*op].SinceVersion() == curr_version) {
          // Op is specifically defined for this version
          if (target_version.version > initial_version.version) {
            // Need to adapt down
          } else {
            // Need to adapt up
          }
        }
      //   TODO: Replace node in graph
      }
      // TODO: Updated model version
      if (target_version.version > initial_version.version) {
        curr_version++;
        next_version++;
      } else {
        curr_version--;
        next_version--;
      }
    }
  }
};

ONNX_NAMESPACE::ModelProto ConvertVersion(
    const ONNX_NAMESPACE::ModelProto& mp_in,
    const OperatorSetVersion target_version);
}}
