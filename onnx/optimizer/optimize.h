// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/common/stl_backports.h"
#include "onnx/optimizer/pass_manager.h"
#include "onnx/optimizer/pass_registry.h"
#include "onnx/proto_utils.h"

#include "vector"

namespace ONNX_NAMESPACE {
namespace optimization {

struct Optimizer {
  static GlobalPassRegistry passes;

 public:
  Optimizer(const std::vector<std::string>& names, const bool fixed_point);
  ~Optimizer();

  ModelProto optimize(const ModelProto& mp_in) {
    std::shared_ptr<Graph> g(ImportModelProto(mp_in));

    if (g.get() == nullptr) {
      std::cerr << "Warning: onnx optimizer is unable to parse input model. "
                << "(The IR version of the ONNX model may be too old.)"
                << std::endl;
      // If we can't parse the file, just return the input.
      return mp_in;
    }

    ModelProto mp_out = PrepareOutput(mp_in);
    this->pass_manager->run(*g);
    ExportModelProto(&mp_out, g);
    return mp_out;
  }

 private:
  std::shared_ptr<PassManager> pass_manager;
};

const std::vector<std::string> GetAvailablePasses();

ModelProto Optimize(
    const ModelProto& mp_in,
    const std::vector<std::string>& names);

ModelProto OptimizeFixed(
    const ModelProto& mp_in,
    const std::vector<std::string>& names);
} // namespace optimization
} // namespace ONNX_NAMESPACE
