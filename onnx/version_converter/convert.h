// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Default converter for ONNX models between different opset versions
// in the same domain.

#pragma once

#include "onnx/version_converter/BaseConverter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct DefaultVersionConverter : BaseVersionConverter {
  // TODO: Change all existing references to VersionConverter
  DefaultVersionConverter() {
    // TODO: Register adapters to the version converter
  }

  ONNX_NAMESPACE::ModelProto convert_version(
      const ONNX_NAMESPACE::ModelProto& mp_in,
      const OpSetID initial_version,
      const OpSetID target_version) {
    std::shared_ptr<ONNX_NAMESPACE::Graph> g(ONNX_NAMESPACE::ImportModelProto(mp_in));

    if (g.get() == nullptr) {
      std::cerr << "Warning: onnx version converter is unable to parse input model. "
        << "(The IR version of the ONNX model may be too old.)" << std::endl;
      // If we can't parse the file, just return the input.
      return mp_in;
    }

    std::string initial_domain = initial_version.domain;
    std::string target_domain = target_version.domain;
    if ((initial_domain != "" && initial_domain != "ai.onnx") || (target_domain !=
        "" && target_domain != "ai.onnx")) {
      // TODO: Replace with ONNX_ASSERTM?
      std::cerr << "Warning: default onnx version converter can only convert "
        << " between default domain opset versions ('' or 'ai.onnx')" << std::endl;
      std::cerr << "Provided initial_domain: " << initial_domain <<
        ", provided target_domain: " << target_domain << std::endl;
      return mp_in;
    }

    ONNX_NAMESPACE::ModelProto mp_out = PrepareOutput(mp_in);

    // TODO: Move to Inter-Domain Converter
    // Get initial model versions
    // std::vector<OpSetID> initial_versions = g->opset_versions;

    // No conversion necessary if Model has single, equivalent opset version
    // if (initial_versions.size() == 1 && initial_versions[0].version ==
    //    target_version.version && initial_versions[0].domain ==
    //    target_version.domain) {
    //  return mp_in;
    // }

    // Check if target_version is valid
    const std::unordered_map<std::string, std::pair<int, int>>& versions_map = OpSchemaRegistry::DomainToVersionRange::Instance().Map();
    std::string search_domain = target_version.domain;
    if (target_version.domain == "ai.onnx") {
      search_domain = "";
    }
    std::pair<int, int> version_range = versions_map.at(search_domain);
    if (target_version.version < version_range.first || target_version.version > version_range.second) {
      // Invalid target_version
      // TODO: Replace with ONNX_ASSERTM?
      std::cerr << "Warning: invalid target_version (must be between "
        << version_range.first << " and " << version_range.second << std::endl;
      return mp_in;
    }

    // Compile list of all ops used in the model
    graph_node_list nodes = g->nodes();

    std::vector<OpSchema> all_opschemas = ONNX_NAMESPACE::OpSchemaRegistry::get_all_schemas_with_history();

    // Create Map for All Versions
    std::unordered_map<std::basic_string<char>, std::unordered_map<std::basic_string<char>, std::map<int64_t, ONNX_NAMESPACE::OpSchema*>>>  all_schemas;

    for (OpSchema schema : all_opschemas) {
      all_schemas[schema.Name()][schema.domain()][(int64_t) schema.since_version()] = &schema;
    }

    // Create Map for Current Version
    for (Node* op : nodes) {
      // Iterate through all OperatorSetVersions, select highest that is leq initial_version
      int64_t op_opset_version = -1;
      auto op_domain_map = all_schemas[op->kind().toString()];
      if (op_domain_map.find(initial_domain) != op_domain_map.end()) {
        // If op isn't defined for initial domain, we won't convert it
        for (const auto& version_pair : op_domain_map[initial_domain]) {
          if (version_pair.first > op_opset_version && version_pair.first <= initial_version.version) {
            op_opset_version = version_pair.first;
            current_opschemas[op] = op_domain_map[initial_domain][op_opset_version];
          }
        }
      }
    }

    // Iterate over all versions to target_version for specified
    int64_t curr_version = initial_version.version;
    int64_t next_version;
    if (target_version.version > initial_version.version) {
      curr_version++;
      next_version = curr_version + (int64_t) 1;
    } else {
      next_version = curr_version - (int64_t) 1;
    }
    // Identify index of this domain in g.opset_versions
    unsigned int domain_index = 0;
    for (unsigned int i = 0; i < g->opset_versions.size(); i++) {
      if (g->opset_versions[i].domain == "") {
        domain_index = i;
      }
    }
    while (curr_version != target_version.version) {
      std::cerr << "curr_version: " << curr_version << ", next_version: " << next_version << std::endl;
      // Iterate through and call adapter returned by adapter_lookup for ops from current_version opset
      for (Node* op : nodes) {
        auto op_domain_map = all_schemas.at(op->kind().toString());
        if (op_domain_map.find("") != op_domain_map.end() &&
            op_domain_map[""].find(curr_version) !=
            op_domain_map[""].end()) {
          // Op is specifically defined for this domain and version
          OpSetID curr_id;
          OpSetID next_id;
          curr_id.domain = "";
          next_id.domain = "";
          curr_id.version = curr_version;
          next_id.version = next_version;
          auto op_adapter = adapter_lookup(op, curr_id, next_id);
          // If adapter_lookup returns null, no adapter is present.  Error out
          // TODO: Verify that conversion is actually needed (that the operator
          // isn't already optimal, which should be caught by the above condition)
          ONNX_ASSERTM(op_adapter != NULL,
              "No adapter is present for %s in default domain. Please implement one and try again.",
              op->kind().toString());
          std::cerr << "Applying adapter" << std::endl;
          // adapt should handle replacing node in graph
          op_adapter->adapt(*g);
        }
      }
      // Update model version
      if (target_version.version > initial_version.version) {
        curr_version++;
        next_version++;
        g->opset_versions[domain_index].version++;
      } else {
        curr_version--;
        next_version--;
        g->opset_versions[domain_index].version--;
      }
    }
    // Export g as ModelProto
    std::cerr << "Finished conversion; returning model\n";
    ExportModelProto(&mp_out, g);
    return mp_out;
  }
};

ONNX_NAMESPACE::ModelProto ConvertVersion(
    const ONNX_NAMESPACE::ModelProto& mp_in,
    const ONNX_NAMESPACE::OperatorSetIdProto initial_version,
    const ONNX_NAMESPACE::OperatorSetIdProto target_version);
}}
