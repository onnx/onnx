// Default converter for ONNX models between different opset versions
// in the default domain ("" or "ai.onnx").

#pragma once

#include "onnx/version_converter/BaseConverter.h"
#include "onnx/version_converter/adapters/NoPreviousVersionAdapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class DefaultVersionConverter : public BaseVersionConverter {
  private:
    bool DEBUG = false;

    // Map of All Versions of format {op_name: {domain: {version: schema}}}
    std::unordered_map<std::string, std::unordered_map<std::string, std::map<int64_t, const OpSchema*>>>  all_schemas;

    bool searchOpDomainMap(const std::unordered_map<std::string, std::map<
      int64_t, const OpSchema*>>& op_domain_map, int64_t curr_version) const {
      const auto version_it = op_domain_map.find("");
      return version_it != op_domain_map.end() &&
          version_it->second.find(curr_version) !=
          version_it->second.end();
    }

    void debug(const std::string& str) const {
      if (DEBUG) std::cerr << str << std::endl;
    }

  public:
    DefaultVersionConverter() {
      // Register adapters to the version converter
      const std::vector<OpSchema>& all_opschemas = OpSchemaRegistry::get_all_schemas_with_history();

      for (const OpSchema& schema : all_opschemas) {
        all_schemas[schema.Name()][schema.domain()][(int64_t) schema.since_version()] = &schema;
      }

      // TODO: Iterate through all_schemas to determine NoPreviousVersionAdapters
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
