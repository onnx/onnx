// Converter for ONNX models between different opset versions
// in the same domain.

#pragma once

#include "onnx/version_converter/BaseConverter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct IntraDomainVersionConverter : BaseVersionConverter {
  bool DEBUG = false;

  IntraDomainVersionConverter() {
    // TODO: Register adapters to the version converter
  }

  const Adapter& adapter_lookup(const Node* op,
      const OpSetID& initial_version,
      const OpSetID& target_version,
      const std::unordered_map<const Node*, const OpSchema*>& current_opschemas) const {
    const std::string& op_name = op->name();
    const std::string& initial = initial_version.toString();
    const std::string& target = target_version.toString();
    // Find appropriate adapter in adapters map for provided initial and target versions
    // TODO: Consider abstracting elements of this that are specific to
    // DefaultConverter to separate methods here and maintain the procedure in Base Converter
    if (adapters.find(op_name) != adapters.end()) {
      // If we're adapting downwards, we just want to find the one downwards
      // adapter implemented for initial_version. If we're adapting upwards, we
      // want to actually use the SinceVersion value for the given op.
      if (target_version.version() < initial_version.version()) {
        // Downwards adapter
        if (adapters.at(op_name).find(initial) != adapters.at(op_name).end()) {
          // Either an upwards or a downwards adapter exists
          // Check if downwards adapter exists (only one should)
          const auto& target_map = adapters.at(op_name).at(initial);
          for (auto it = target_map.begin(); it != target_map.end(); ++it) {
            int64_t new_target = (OpSetID(it->first)).version();
            if (new_target <= target_version.version()) {
              // Adapter found
              return *(it->second);
            }
          }
          // If loop terminates, no downwards adapter was found
          // TODO: Instead return OpAlreadyAtOldestVersion
          throw "OpAlreadyAtOldestVersion";
        } else {
          // No adapters exist from initial_version
          // TODO: Instead return NoAdapterForCurrentVersion
          throw "NoAdapterForCurrentVersion";
        }
      } else {
        // Upwards adapter
        // Either adapt from SinceVersion or Incompatible Breaking Change
        // TODO: Verify that this doesn't end up defaulting to a downwards
        // adapter on accident.
        std::string since = target_version.domain() + std::to_string(
            current_opschemas.at(op)->since_version());
        if (adapters.at(op_name).find(since) != adapters.at(op_name).end() &&
          adapters.at(op_name).at(since).find(target) != adapters.at(op_name)
          .at(since).end()) {
          return *(adapters.at(op_name).at(since).at(target));
        } else {
          // TODO: Instead return NoUpwardsAdapter
          throw "NoUpwardsAdapter";
        }
      }
    } else {
      // No adapters exist for the given op
      // TODO: Instead return NoAdapterForOp
      throw "NoAdapterForOp";
    }
  }

  ONNX_NAMESPACE::ModelProto convert_version(
      const ONNX_NAMESPACE::ModelProto& mp_in,
      const OpSetID& initial_version,
      const OpSetID& target_version) const {
    ONNX_ASSERTM(strcmp(initial_version.domain().c_str(), target_version.domain()
          .c_str()) == 0, "initial_version and target_version must have the same "
        "domains");
    for (auto it = mp_in.opset_import().begin(); it != mp_in.opset_import()
        .end(); ++it) {
      if (it->domain() == initial_version.domain()) {
        ONNX_ASSERTM(initial_version.version() == it->version(),
            "initial_version does not reflect current state of model");
      }
    }

    std::shared_ptr<ONNX_NAMESPACE::Graph> g(ONNX_NAMESPACE::ImportModelProto(mp_in));
    ONNX_ASSERTM(g.get() != nullptr,
      "Warning: onnx version converter is unable to parse input model. "
      "(The IR version of the ONNX model may be too old.)");

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
    const std::string& search_domain = target_version.domain() == "ai.onnx" ? "" : target_version.domain();
    const std::pair<int, int>& version_range = versions_map.at(search_domain);
    ONNX_ASSERTM(target_version.version() >= version_range.first && target_version
        .version() <= version_range.second,
      "Warning: invalid target_version (must be between %s and %s",
      version_range.first, version_range.second);
    // Compile list of all ops used in the model
    graph_node_list nodes = g->nodes();

    const std::vector<OpSchema>& all_opschemas = ONNX_NAMESPACE::OpSchemaRegistry::get_all_schemas_with_history();

    // Create Map for All Versions of format {op_name: {domain: {version: schema}}}
    std::unordered_map<std::basic_string<char>, std::unordered_map<std::basic_string<char>, std::map<int64_t, const ONNX_NAMESPACE::OpSchema*>>>  all_schemas;

    for (const OpSchema& schema : all_opschemas) {
      all_schemas[schema.Name()][schema.domain()][(int64_t) schema.since_version()] = &schema;
    }

    const char* initial_domain = initial_version.domain().c_str();
    const char* target_domain = target_version.domain().c_str();

    // Create Map for Current Version
    std::unordered_map<const Node*, const OpSchema*> current_opschemas;
    for (const Node* op : nodes) {
      // Iterate through all OperatorSetVersions, select highest that is leq initial_version
      int64_t op_opset_version = -1;
      auto& op_domain_map = all_schemas[op->kind().toString()];
      if (op_domain_map.find(initial_domain) != op_domain_map.end()) {
        // If op isn't defined for initial domain, we won't convert it
        for (const auto& version_pair : op_domain_map[initial_domain]) {
          if (version_pair.first > op_opset_version && version_pair.first <= initial_version.version()) {
            op_opset_version = version_pair.first;
            current_opschemas[op] = op_domain_map.at(initial_domain).at(op_opset_version);
          }
        }
      }
    }

    // Iterate over all versions to target_version for specified
    int64_t curr_version = initial_version.version();
    int64_t step;
    if (target_version.version() > initial_version.version()) {
      curr_version++;
      step = 1;
    } else {
      step = -1;
    }
    // Identify index of this domain in g.opset_versions
    unsigned int domain_index = 0;
    for (unsigned int i = 0; i < g->opset_versions.size(); i++) {
      if (g->opset_versions[i].domain() == "") {
        domain_index = i;
      }
    }
    while (curr_version != target_version.version()) {
      if (DEBUG) {
        std::cerr << "curr_version: " << curr_version << ", next_version: " <<
          curr_version + step << std::endl;
      }
      // Iterate through and call adapter returned by adapter_lookup for ops from current_version opset
      for (const Node* op : nodes) {
        auto& op_domain_map = all_schemas.at(op->kind().toString());
        if (op_domain_map.find("") != op_domain_map.end() &&
            op_domain_map[""].find(curr_version) !=
            op_domain_map[""].end()) {
          // Op is specifically defined for this domain and version
          OpSetID curr_id(curr_version);
          OpSetID next_id(curr_version + step);
          auto& op_adapter = adapter_lookup(op, curr_id, next_id, current_opschemas);
          // If adapter_lookup returns null, no adapter is present.
          // Error thrown by adapter_lookup
          // TODO: Verify that conversion is actually needed (that the operator
          // isn't already optimal, which should be caught by the above condition)
          if (DEBUG) {
            std::cerr << "Applying adapter" << std::endl;
          }
          // adapt should handle replacing node in graph
          op_adapter.adapt(g, op);
        }
      }
      // Update model version
      curr_version += step;
      g->opset_versions[domain_index].incrementVersion(step);
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
