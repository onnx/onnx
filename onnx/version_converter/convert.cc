// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/version_converter/convert.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

static VersionConverter _version_converter;

ONNX_NAMESPACE::Adapter adapter_lookup(const Node op,
    const OperatorSetVersion initial_version,
    const OperatorSetVersion target_version) {
  std::string op_name = op.name;
  // TODO: Find appropriate adapter in adapters map for provided initial and target versions
  if (adapters.contains(op_name)) {
    // TODO: If we're adapting downwards, we just want to find the one downwards
    // adapter implemented for initial_version. If we're adapting upwards, we
    // want to actually use the SinceVersion value for the given op.
    if (target_version < initial_version) {
      // Downwards adapter
      if (adapters[op_name].contains(initial_version)) {
        // Either an upwards or a downwards adapter exists
        // Check if downwards adapter exists (only one should)
        for (OperatorSetVersion target : adapters[op_name][initial_version]) {
          if (target <= target_version) {
            // Adapter found
            return adapters[op_name][initial_version][target];
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
      if (adapters[op_name].contains(since_version) && adapters[op_name]
          [since_version].contains(target_version)) {
        return adapters[op_name][since_version][target_version];
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

ONNX_NAMESPACE::ModelProto Convert(
    const ONNX_NAMESPACE::ModelProto& mp_in,
    const OpSetID target_version) {
  return _version_converter.convert(mp_in, target_version);
}

}}
