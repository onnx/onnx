// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/version_converter/convert.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

static DefaultVersionConverter _version_converter;

ONNX_NAMESPACE::ModelProto ConvertVersion(
    const ONNX_NAMESPACE::ModelProto& mp_in,
    const int target_version) {
  // Get initial_opsetid from mp_in
  OpSetID* initial_struct;
  for (auto it = mp_in.opset_import().begin(); it != mp_in.opset_import().end(); ++it) {
    if (it->domain() == "" || it->domain() == "ai.onnx") {
      initial_struct = new OpSetID(*it);
    }
  }
  OpSetID* target_struct = new OpSetID(target_version);
  return _version_converter.convert_version(mp_in, *initial_struct, *target_struct);
}
}}
