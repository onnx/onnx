// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/version_converter/convert.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

static VersionConverter _version_converter;

Adapter* VersionConverter::adapter_lookup(Node* op,
    const OpSetID& initial_version,
    const OpSetID& target_version) {
  std::string op_name = op->name();
  std::string initial = stringify_opsetid(initial_version);
  std::string target = stringify_opsetid(target_version);
  // TODO: Find appropriate adapter in adapters map for provided initial and target versions
  if (adapters.find(op_name) != adapters.end()) {
    // TODO: If we're adapting downwards, we just want to find the one downwards
    // adapter implemented for initial_version. If we're adapting upwards, we
    // want to actually use the SinceVersion value for the given op.
    if (target_version.version < initial_version.version) {
      // Downwards adapter
      if (adapters[op_name].find(initial) != adapters[op_name].end()) {
        // Either an upwards or a downwards adapter exists
        // Check if downwards adapter exists (only one should)
        const auto target_map = adapters[op_name][initial];
        for (auto it = target_map.begin(); it != target_map.end(); ++it) {
          int new_target;
          sscanf(destringify_opsetid(it->first)[1].c_str(), "%d", &new_target);
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
      std::string since = target_version.domain + std::to_string(
          current_opschemas[op].since_version());
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

std::string VersionConverter::stringify_opsetid(OpSetID target) {
    return target.domain + "$" + std::to_string(target.version);
}

std::vector<std::string> VersionConverter::destringify_opsetid(std::string target) {
  std::stringstream ss(target);
  std::string segment;
  std::vector<std::string> seglist;
  while (std::getline(ss, segment, '$')) {
    seglist.push_back(segment);
  }
  return seglist;
}

OpSetID VersionConverter::operatorsetidproto_to_opsetid(
    ONNX_NAMESPACE::OperatorSetIdProto proto) {
  OpSetID retval;
  retval.domain = proto.domain();
  retval.version = proto.version();
  return retval;
}

ONNX_NAMESPACE::ModelProto ConvertVersion(
    const ONNX_NAMESPACE::ModelProto& mp_in,
    const ONNX_NAMESPACE::OperatorSetIdProto initial_version,
    const ONNX_NAMESPACE::OperatorSetIdProto target_version) {
  OpSetID initial_struct = _version_converter.operatorsetidproto_to_opsetid(initial_version);
  OpSetID target_struct = _version_converter.operatorsetidproto_to_opsetid(target_version);
  return _version_converter.convert_version(mp_in, initial_struct, target_struct);
}

}}
