// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Adapter for Add in default domain from version 7 to 6

#pragma once

#include "onnx/version_converter/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct Add_7_6 final : public Adapter {
  explicit Add_7_6()
    : Adapter("Add", make_opsetid("", 7), make_opsetid("", 6)) {
    }

  OpSetID make_opsetid(std::string domain, int version) {
    OpSetID ret;
    ret.domain = domain;
    ret.version = version;
    return ret;
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion
