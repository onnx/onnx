// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/optimizer/optimize.h"

namespace ONNX_NAMESPACE { namespace optimization {

void PrepareOutput(const ONNX_NAMESPACE::ModelProto& mp_in, ONNX_NAMESPACE::ModelProto& mp_out) {
  if (mp_in.has_producer_name()) {
    mp_out.set_ir_version(mp_in.ir_version());
  }
  if (mp_in.has_producer_name()) {
    mp_out.set_producer_name(mp_in.producer_name());
  }
  if (mp_in.has_producer_version()) {
    mp_out.set_producer_version(mp_in.producer_version());
  }
  if (mp_in.has_domain()) {
    mp_out.set_domain(mp_in.domain());
  }
  if (mp_in.has_model_version()) {
    mp_out.set_model_version(mp_in.model_version());
  }
  if (mp_in.has_doc_string()) {
    mp_out.set_doc_string(mp_in.doc_string());
  }
  for (int i = 0; i < mp_in.opset_import_size(); i++) {
    auto& oi_in = mp_in.opset_import(i);
    auto* oi_out = mp_out.add_opset_import();
    if (oi_in.has_domain()) {
      oi_out->set_domain(oi_in.domain());
    }
    if (oi_in.has_version()) {
      oi_out->set_version(oi_in.version());
    }
  }
  for (int i = 0; i < mp_in.metadata_props_size(); i++) {
    auto& pp_in = mp_in.metadata_props(i);
    auto* pp_out = mp_out.add_metadata_props();
    if (pp_in.has_key()) {
      pp_out->set_key(pp_in.key());
    }
    if (pp_in.has_value()) {
      pp_out->set_value(pp_in.value());
    }
  }
}

static Optimizer _optimizer;

ONNX_NAMESPACE::ModelProto Optimize(
    const ONNX_NAMESPACE::ModelProto& mp_in,
    const std::vector<std::string>& names) {
  return _optimizer.optimize(mp_in, names);

}

}}
