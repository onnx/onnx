// Default converter for ONNX models between different opset versions
// in the default domain ("" or "ai.onnx").

#pragma once

#include "onnx/version_converter/IntraDomainConverter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class DefaultVersionConverter : IntraDomainVersionConverter {
  public:
    DefaultVersionConverter() {
      // Register adapters to the version converter
    }

    ONNX_NAMESPACE::ModelProto convert_version(
        const ONNX_NAMESPACE::ModelProto& mp_in,
        const OpSetID& initial_version,
        const OpSetID& target_version) const {
      const char* initial_domain = initial_version.domain().c_str();
      const char* target_domain = target_version.domain().c_str();
      ONNX_ASSERTM((strcmp(initial_domain, "") == 0 || strcmp(initial_domain,
              "ai.onnx") == 0) && (strcmp(target_domain, "") == 0 || strcmp(
                target_domain, "ai.onnx") == 0),
          "Warning: default onnx version converter can only convert "
          " between default domain opset versions ('' or 'ai.onnx')\n"
          "Provided initial_domain: %s"
          ", provided target_domain: %s", initial_domain, target_domain);

      return IntraDomainVersionConverter::convert_version(mp_in, initial_version,
          target_version);
    }
};

ONNX_NAMESPACE::ModelProto ConvertVersion(
    const ONNX_NAMESPACE::ModelProto& mp_in,
    const int target_version);
}} // namespace ONNX_NAMESPACE::version_conversion
