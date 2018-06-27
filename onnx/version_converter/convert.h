// A converter for ONNX models between different opset versions

#pragma once

#include "onnx/common/ir.h"
#include "onnx/proto_utils.h"
#include "onnx/defs/schema.h"
#include <utility>

namespace ONNX_NAMESPACE { namespace version_conversion {

struct VersionConverter {
  VersionConverter() {
    // TODO: Should we use a similar registration structure to ops and optimizers for adapters?
  }

  virtual ~Converter() = default;

  ONNX_NAMESPACE::ModelProto convert_version(
      const ONNX_NAMESPACE::ModelProto& mp_in,
      const int target_version) {
    std::shared_ptr<ONNX_NAMESPACE::Graph> g(ONNX_NAMESPACE::ImportModelProto(mp_in));

    if (g.get() == nullptr) {
      std::cerr << "Warning: onnx optimizer is unable to parse input model. "
        << "(The IR version of the ONNX model may be too old.)" << std::endl;
      // If we can't parse the file, just return the input.
      return mp_in;
    }

    // Check if target_version is valid
    std::pair version_range = OpSchemaRegistry::DomainToVersionRange::Instance().Map()[""];
    if (target_version < version_range.first || target_version > version_range.second) {
      // Invalid target_version
      std::cerr << "Warning: invalid target_version (must be between "
        << version_range.first << " and " << version_range.second << std::endl;
      return mp_in;
    }

    // TODO: Get initial model version
    int initial_version;

    if (initial_version == target_version) {
      return mp_in;
    }

    // Iterate over all versions to target_version
    int curr_version = initial_version;
    while (curr_version != target_version) {

      if (target_version > initial_version) {
        curr_version++;
      } else {
        curr_version--;
      }
    }
  }
};

ONNX_NAMESPACE::ModelProto ConvertVersion(
    const ONNX_NAMESPACE::ModelProto& mp_in,
    const int target_version);
}}
