// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/version_converter/convert.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

static DefaultVersionConverter _version_converter;

ONNX_NAMESPACE::ModelProto ConvertVersion(
    const ONNX_NAMESPACE::ModelProto& mp_in,
    const ONNX_NAMESPACE::OperatorSetIdProto initial_version,
    const ONNX_NAMESPACE::OperatorSetIdProto target_version) {
  OpSetID initial_struct = _version_converter.operatorsetidproto_to_opsetid(initial_version);
  OpSetID target_struct = _version_converter.operatorsetidproto_to_opsetid(target_version);
  return _version_converter.convert_version(mp_in, initial_struct, target_struct);
}

}}
