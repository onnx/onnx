// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Default converter for ONNX models between different opset versions
// in the same domain.

#pragma once

#include "onnx/version_converter/BaseConverter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct DefaultVersionConverter : BaseVersionConverter {
  bool DEBUG = false;

  DefaultVersionConverter() {
    // TODO: Register adapters to the version converter
  }

  Adapter* adapter_lookup(Node* op,
      const OpSetID& initial_version,
      const OpSetID& target_version) {
    std::string op_name = op->name();
    std::string initial = initial_version.toString();
    std::string target = target_version.toString();
    // Find appropriate adapter in adapters map for provided initial and target versions
    // TODO: Consider abstracting elements of this that are specific to
    // DefaultConverter to separate methods here and maintain the procedure in Base Converter
    if (adapters.find(op_name) != adapters.end()) {
      // If we're adapting downwards, we just want to find the one downwards
      // adapter implemented for initial_version. If we're adapting upwards, we
      // want to actually use the SinceVersion value for the given op.
      if (target_version.version < initial_version.version) {
        // Downwards adapter
        if (adapters[op_name].find(initial) != adapters[op_name].end()) {
          // Either an upwards or a downwards adapter exists
          // Check if downwards adapter exists (only one should)
          const auto target_map = adapters[op_name][initial];
          for (auto it = target_map.begin(); it != target_map.end(); ++it) {
            int64_t new_target = (new OpSetID(it->first))->version;
            if (new_target <= target_version.version) {
              // Adapter found
              return &*(it->second);
            }
          }
          // If loop terminates, no downwards adapter was found
          // TODO: Instead return OpAlreadyAtOldestVersion
          return NULL;
        } else {
          // No adapters exist from initial_version
          // TODO: Instead return NoAdapterForCurrentVersion
          return NULL;
        }
      } else {
        // Upwards adapter
        // Either adapt from SinceVersion or Incompatible Breaking Change
        // TODO: Verify that this doesn't end up defaulting to a downwards
        // adapter on accident.
        std::string since = target_version.domain + std::to_string(
            current_opschemas[op]->since_version());
        if (adapters[op_name].find(since) != adapters[op_name].end() && adapters[op_name]
            [since].find(target) != adapters[op_name][since].end()) {
          return &*(adapters[op_name][since][target]);
        } else {
          // TODO: Instead return NoUpwardsAdapter
          return NULL;
        }
      }
    } else {
      // No adapters exist for the given op
      // TODO: Instead return NoAdapterForOp
      return NULL;
    }
  }

  ONNX_NAMESPACE::ModelProto convert_version(
      const ONNX_NAMESPACE::ModelProto& mp_in,
      const OpSetID initial_version,
      const OpSetID target_version) {
    std::shared_ptr<ONNX_NAMESPACE::Graph> g(ONNX_NAMESPACE::ImportModelProto(mp_in));
    ONNX_ASSERTM(g.get() != nullptr,
      "Warning: onnx version converter is unable to parse input model. "
      "(The IR version of the ONNX model may be too old.)");

    const char* initial_domain = initial_version.domain.c_str();
    const char* target_domain = target_version.domain.c_str();
    ONNX_ASSERTM((strcmp(initial_domain, "") == 0 || strcmp(initial_domain,
            "ai.onnx") == 0) && (strcmp(target_domain, "") == 0 || strcmp(
              target_domain, "ai.onnx") == 0),
        "Warning: default onnx version converter can only convert "
        " between default domain opset versions ('' or 'ai.onnx')\n"
        "Provided initial_domain: %s"
        ", provided target_domain: %s", initial_domain, target_domain);

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
    ONNX_ASSERTM(target_version.version >= version_range.first && target_version
        .version <= version_range.second,
      "Warning: invalid target_version (must be between %s and %s",
      version_range.first, version_range.second);
    // Compile list of all ops used in the model
    graph_node_list nodes = g->nodes();

    std::vector<OpSchema> all_opschemas = ONNX_NAMESPACE::OpSchemaRegistry::get_all_schemas_with_history();

    // Create Map for All Versions of format {op_name: {domain: {version: schema}}}
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
    int64_t step;
    if (target_version.version > initial_version.version) {
      curr_version++;
      step = 1;
    } else {
      step = -1;
    }
    // Identify index of this domain in g.opset_versions
    unsigned int domain_index = 0;
    for (unsigned int i = 0; i < g->opset_versions.size(); i++) {
      if (g->opset_versions[i].domain == "") {
        domain_index = i;
      }
    }
    while (curr_version != target_version.version) {
      if (DEBUG) {
        std::cerr << "curr_version: " << curr_version << ", next_version: " <<
          curr_version + step << std::endl;
      }
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
          next_id.version = curr_version + step;
          auto op_adapter = adapter_lookup(op, curr_id, next_id);
          // If adapter_lookup returns null, no adapter is present.  Error out
          // TODO: Verify that conversion is actually needed (that the operator
          // isn't already optimal, which should be caught by the above condition)
          ONNX_ASSERTM(op_adapter != NULL,
              "No adapter is present for %s in default domain. Please implement one and try again.",
              op->kind().toString());
          if (DEBUG) {
            std::cerr << "Applying adapter" << std::endl;
          }
          // adapt should handle replacing node in graph
          op_adapter->adapt(*g, *op);
        }
      }
      // Update model version
      curr_version += step;
      g->opset_versions[domain_index].version += step;
    }
    // Export g as ModelProto
    if (DEBUG) {
      std::cerr << "Finished conversion; returning model\n";
    }
    ExportModelProto(&mp_out, g);
    return mp_out;
  }
};

ONNX_NAMESPACE::ModelProto ConvertVersion(
    const ONNX_NAMESPACE::ModelProto& mp_in,
    const int target_version);
}}
