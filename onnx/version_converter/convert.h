// Default converter for ONNX models between different opset versions
// in the default domain ("" or "ai.onnx").

#pragma once

#include "onnx/version_converter/BaseConverter.h"
#include "onnx/version_converter/adapters/no_previous_version.h"
#include "onnx/version_converter/adapters/broadcast_backward_compatibility.h"
#include "onnx/version_converter/adapters/broadcast_forward_compatibility.h"
#include "onnx/version_converter/adapters/type_restriction.h"
#include "onnx/version_converter/adapters/backwards_compatible.h"
#include "onnx/version_converter/adapters/batch_normalization_6_7.h"
#include "onnx/version_converter/adapters/batch_normalization_6_5.h"
#include "onnx/version_converter/adapters/remove_consumed_inputs.h"
#include "onnx/version_converter/adapters/concat_3_4.h"
#include "onnx/version_converter/adapters/concat_4_3.h"
#include "onnx/version_converter/adapters/reshape_5_4.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class DefaultVersionConverter : public BaseVersionConverter {
  private:
    bool DEBUG = false;

    std::pair<int, int> version_range;

    bool searchOpDomainMap(const std::unordered_map<std::string, std::map<
      int64_t, const OpSchema*>>& op_domain_map, int64_t curr_version,
      int64_t step) const {
      bool up = step == 1;
      const auto version_it = op_domain_map.find("");
      return version_it != op_domain_map.end() &&
          ((version_it->second.find(curr_version) !=
          version_it->second.end() && !up) ||
          (version_it->second.find(curr_version + step) !=
          version_it->second.end() && up));
    }

    void debug(const std::string& str) const {
      if (DEBUG) std::cerr << str << std::endl;
    }

    void assertInVersionRange(int64_t version) const {
      ONNX_ASSERTM(version >= version_range.first && version <=
          version_range.second,
          "Warning: invalid version (must be between %s and %s",
          version_range.first, version_range.second);
    }

    void assertDefaultDomain(const std::string& initial_domain,
        const std::string& target_domain) const {
      ONNX_ASSERTM((initial_domain == "" || initial_domain == "ai.onnx") &&
          (target_domain == "" || target_domain == "ai.onnx"),
          "Warning: default onnx version converter can only convert "
          " between default domain opset versions ('' or 'ai.onnx')\n");
      ONNX_ASSERTM(initial_domain == target_domain,
          "initial_version and target_version must have the same domains");
    }

  public:
    DefaultVersionConverter() {
      const std::unordered_map<std::string, std::pair<int, int>>& versions_map =
        OpSchemaRegistry::DomainToVersionRange::Instance().Map();
      version_range = versions_map.at("");
      // Register adapters to the version converter
      const std::vector<OpSchema> all_opschemas =
        OpSchemaRegistry::get_all_schemas_with_history();
      registerAdapter(make_unique<Adapter>(Add_7_6()));
      registerAdapter(make_unique<Adapter>(Add_6_7()));
      registerAdapter(make_unique<Adapter>(Relu_5_6()));
      registerAdapter(make_unique<Adapter>(
        BackwardsCompatibleAdapter("Relu", OpSetID(6), OpSetID(5))));
      registerAdapter(std::unique_ptr<Adapter>(new
        BackwardsCompatibleAdapter("BatchNormalization", OpSetID(7), OpSetID(6))));
      registerAdapter(std::unique_ptr<Adapter>(new
        BatchNormalization_6_7()));
      registerAdapter(std::unique_ptr<Adapter>(new
        BatchNormalization_6_5()));
      registerAdapter(make_unique<NoPreviousVersionAdapter>("Cos",
        OpSetID(7), OpSetID(6)));
      registerAdapter(make_unique<BroadcastBackwardCompatibility>("Add",
        OpSetID(7), OpSetID(6)));
      registerAdapter(make_unique<BroadcastBackwardCompatibility>("Mul",
        OpSetID(7), OpSetID(6)));
      registerAdapter(make_unique<BroadcastForwardCompatibility>("Add",
        OpSetID(6), OpSetID(7)));
      registerAdapter(make_unique<BroadcastForwardCompatibility>("Mul",
        OpSetID(6), OpSetID(7)));
      registerAdapter(make_unique<TypeRestriction>("Add",
        OpSetID(6), OpSetID(5)));
      registerAdapter(make_unique<TypeRestriction>("Mul",
        OpSetID(6), OpSetID(5)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Relu",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<BackwardsCompatibleAdapter>("Relu",
        OpSetID(6), OpSetID(5)));
      registerAdapter(make_unique<BackwardsCompatibleAdapter>("BatchNormalization",
        OpSetID(7), OpSetID(6)));
      registerAdapter(make_unique<BatchNormalization_6_7>());
      registerAdapter(make_unique<BatchNormalization_6_5>());
      registerAdapter(make_unique<RemoveConsumedInputs>("BatchNormalization",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Add",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Mul",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<Concat_3_4>());
      registerAdapter(make_unique<Concat_4_3>());
      registerAdapter(make_unique<Reshape_5_4>());
    }

    ModelProto convert_version(
        const ModelProto& mp_in,
        const OpSetID& initial_version,
        const OpSetID& target_version) const override {
      const std::string initial_domain = initial_version.domain();
      const std::string target_domain = target_version.domain();

      ONNX_ASSERTM((initial_domain == "" || initial_domain == "ai.onnx") &&
        (target_domain == "" || target_domain == "ai.onnx"),
          "Warning: default onnx version converter can only convert "
          " between default domain opset versions ('' or 'ai.onnx')\n"
          "Provided initial_domain: %s, provided target_domain: %s",
          initial_domain.c_str(), target_domain.c_str());

      ONNX_ASSERTM(initial_domain == target_domain,
        "initial_version and target_version must have the same domains");
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
      // std::vector<OpSetID> initial_versions = g->opset_versions_mutable();

      // No conversion necessary if Model has single, equivalent opset version
      // if (initial_versions.size() == 1 && initial_versions[0].version ==
      //    target_version.version && initial_versions[0].domain ==
      //    target_version.domain) {
      //  return mp_in;
      // }

      // Check if target_version is valid
      const std::unordered_map<std::string, std::pair<int, int>>& versions_map = OpSchemaRegistry::DomainToVersionRange::Instance().Map();
      const std::string search_domain = target_version.domain() == "ai.onnx" ? "" : target_version.domain();
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
>>>>>>> Working!

      for (const OpSchema& schema : all_opschemas) {
        all_schemas[schema.Name()][schema.domain()][(int64_t)
          schema.since_version()] = &schema;
          debug("Schema for " + schema.Name());
      }

      // Iterate through all_schemas to determine NoPreviousVersionAdapters
      for (auto& op_pair : all_schemas) {
        const auto default_versions = op_pair.second.find("");
        if (default_versions != op_pair.second.end()) {
          int64_t min_version = version_range.second;
          for (auto& version_pair : default_versions->second) {
            if (version_pair.first < min_version) {
              min_version = version_pair.first;
            }
          }
          if (min_version > 1) {
            debug("Creating NoPreviousVersionAdapter for " + op_pair.first + " from " + ONNX_NAMESPACE::to_string(min_version));
            registerAdapter(make_unique<NoPreviousVersionAdapter>(op_pair.first,
              OpSetID(min_version), OpSetID(min_version - 1)));
        }
      }
>>>>>>> W
    }

    ModelProto convert_version(
        const ModelProto& mp_in,
        const OpSetID& initial_version,
        const OpSetID& target_version) const override;
};

ModelProto ConvertVersion(
    const ModelProto& mp_in,
    const int target_version);
}} // namespace ONNX_NAMESPACE::version_conversion
