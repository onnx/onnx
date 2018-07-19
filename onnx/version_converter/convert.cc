#include "onnx/version_converter/convert.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

namespace {
  static DefaultVersionConverter _version_converter;
}

ONNX_NAMESPACE::ModelProto ConvertVersion(
    const ONNX_NAMESPACE::ModelProto& mp_in,
    const int target_version) {
  // Get initial_opsetid from mp_in
  OpSetID initial_struct(0);
  for (auto it = mp_in.opset_import().begin(); it != mp_in.opset_import().end(); ++it) {
    if (it->domain() == "" || it->domain() == "ai.onnx") {
      initial_struct.setVersion(it->version());
      break;
    }
  }
  OpSetID target_struct = OpSetID(target_version);
  return _version_converter.convert_version(mp_in, initial_struct, target_struct);
}

}} // namespace ONNX_NAMESPACE::version_conversion
