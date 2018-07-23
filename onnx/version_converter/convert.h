// Default converter for ONNX models between different opset versions
// in the default domain ("" or "ai.onnx").

#pragma once

#include "onnx/version_converter/BaseConverter.h"
#include "onnx/version_converter/adapters/NoPreviousVersionAdapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class DefaultVersionConverter : public BaseVersionConverter {
  private:
    bool DEBUG = true;

    std::pair<int, int> version_range;

    bool searchOpDomainMap(const std::unordered_map<std::string, std::map<
      int64_t, const OpSchema*>>& op_domain_map, int64_t curr_version,
      int64_t step, bool up) const {
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

  public:
    DefaultVersionConverter() {
      const std::unordered_map<std::string, std::pair<int, int>>& versions_map =
        OpSchemaRegistry::DomainToVersionRange::Instance().Map();
      version_range = versions_map.at("");
      // Register adapters to the version converter
      const std::vector<OpSchema>& all_opschemas =
        OpSchemaRegistry::get_all_schemas_with_history();

      for (const OpSchema& schema : all_opschemas) {
        all_schemas[schema.Name()][schema.domain()][(int64_t)
          schema.since_version()] = &schema;
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
            debug("Creating NoPreviousVersionAdapter for " + op_pair.first + " from " + std::to_string(min_version));
            registerAdapter(make_unique<NoPreviousVersionAdapter>(op_pair.first,
              OpSetID(min_version), OpSetID(min_version - 1)));
          }
        }
      }
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
