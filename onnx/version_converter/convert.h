// Default converter for ONNX models between different opset versions
// in the default domain ("" or "ai.onnx").

#pragma once

#include "onnx/version_converter/BaseConverter.h"
#include "onnx/version_converter/adapters/NoPreviousVersionAdapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class DefaultVersionConverter : public BaseVersionConverter {
  private:
    bool DEBUG = false;

    bool searchOpDomainMap(const std::unordered_map<std::string, std::map<
      int64_t, const OpSchema*>>& op_domain_map, int64_t curr_version) const {
      const auto& version_it = op_domain_map.find("");
      return version_it != op_domain_map.end() &&
          version_it->second.find(curr_version) !=
          version_it->second.end();
    }

  public:
    DefaultVersionConverter() {
      // Register adapters to the version converter
      registerAdapter(make_unique<Adapter>(NoPreviousVersionAdapter("Cos",
        OpSetID(7), OpSetID(6))));
    }

    ModelProto convert_version(
        const ModelProto& mp_in,
        const OpSetID& initial_version,
        const OpSetID& target_version) const override {
      const char* initial_domain = initial_version.domain().c_str();
      const char* target_domain = target_version.domain().c_str();

      ONNX_ASSERTM((strcmp(initial_domain, "") == 0 || strcmp(initial_domain,
              "ai.onnx") == 0) && (strcmp(target_domain, "") == 0 || strcmp(
                target_domain, "ai.onnx") == 0),
          "Warning: default onnx version converter can only convert "
          " between default domain opset versions ('' or 'ai.onnx')\n"
          "Provided initial_domain: %s"
          ", provided target_domain: %s", initial_domain, target_domain);

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

      std::shared_ptr<Graph> g(ImportModelProto(mp_in));
      ONNX_ASSERTM(g.get() != nullptr,
        "Warning: onnx version converter is unable to parse input model. "
        "(The IR version of the ONNX model may be too old.)");

      // TODO: Move to Inter-Domain Converter
      // Get initial model versions
      // std::vector<OpSetID> initial_versions = g->opset_versions();

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

      const std::vector<OpSchema>& all_opschemas = OpSchemaRegistry::get_all_schemas_with_history();

      // Create Map for All Versions of format {op_name: {domain: {version: schema}}}
      std::unordered_map<std::string, std::unordered_map<std::string, std::map<int64_t, const OpSchema*>>>  all_schemas;

      for (const OpSchema& schema : all_opschemas) {
        all_schemas[schema.Name()][schema.domain()][(int64_t) schema.since_version()] = &schema;
      }

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
      for (unsigned int i = 0; i < g->opset_versions().size(); i++) {
        if (g->opset_versions()[i].domain() == "") {
          domain_index = i;
        }
      }
      while (curr_version != target_version.version()) {
        if (DEBUG) {
          std::cerr << "curr_version: " << curr_version << ", next_version: " <<
            curr_version + step << std::endl;
        }
        // Iterate through and call adapter returned by adapter_lookup for ops from current_version opset
        for (Node* op : nodes) {
          auto& op_domain_map = all_schemas.at(op->kind().toString());
          if (searchOpDomainMap(op_domain_map, curr_version)) {
            // Op is specifically defined for this domain and version
            OpSetID curr_id(curr_version);
            OpSetID next_id(curr_version + step);
            auto& op_adapter = adapter_lookup(op, curr_id, next_id);
            // If adapter_lookup returns null, no adapter is present.
            // Error thrown by adapter_lookup
            if (DEBUG) std::cerr << "Applying adapter" << std::endl;
            // adapt should handle replacing node in graph
            op_adapter.adapt(g, op);
          }
        }
        // Update model version
        curr_version += step;
        g->opset_versions()[domain_index].incrementVersion(step);
      }
      // Export g as ModelProto
      if (DEBUG) {
        std::cerr << "Finished conversion; returning model\n";
      }
      ModelProto mp_out = PrepareOutput(mp_in);
      ExportModelProto(&mp_out, g);
      return mp_out;
    }
};

ModelProto ConvertVersion(
    const ModelProto& mp_in,
    const int target_version);
}} // namespace ONNX_NAMESPACE::version_conversion
