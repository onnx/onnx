// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/version_converter/convert.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

static VersionConverter _version_converter;

Adapter* VersionConverter::adapter_lookup(Node* op,
    const OpSetID& initial_version,
    const OpSetID& target_version) {
  std::string op_name = gen_key_string(op->name, initial_version, target_version);
  // TODO: Find appropriate adapter in adapters map for provided initial and target versions
  if (adapters.find(op_name) != adapters.end()) {
    // TODO: If we're adapting downwards, we just want to find the one downwards
    // adapter implemented for initial_version. If we're adapting upwards, we
    // want to actually use the SinceVersion value for the given op.
    if (target_version < initial_version) {
      // Downwards adapter
      if (adapters[op_name].find(initial_version) != adapters[op_name].end()) {
        // Either an upwards or a downwards adapter exists
        // Check if downwards adapter exists (only one should)
        const auto target_map = adapters[op_name][initial_version];
        for (auto it = target_map.begin(); it != target_map.end(); ++it) {
          if (it->first <= target_version) {
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
      OperatorSetVersion since_version = current_opschemas[op].SinceVersion();
      if (adapters[op_name].find(since_version) != adapters[op_name].end() && adapters[op_name]
          [since_version].find(target_version) != adapters[op_name][since_version].end()) {
        return &*(adapters[op_name][since_version][target_version]);
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

std::string VersionConverter::gen_key_string(std::string_op_name, OpSetID
    initial, OpSetID target) {
    return op_name + "$" + initial.domain + initial.version + "$" + target.domain +
      target.version;
}

ONNX_NAMESPACE::ModelProto ConvertVersion(
    const ONNX_NAMESPACE::ModelProto& mp_in,
    const OpSetID initial_version,
    const OpSetID target_version) {
  return _version_converter.convert_version(mp_in, initial_version, target_version);
}

}}
