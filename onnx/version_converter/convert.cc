/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/version_converter/convert.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

ModelProto ConvertVersion(const ModelProto& mp_in, int target_version) {
  // Get initial_opsetid from mp_in
  OperatorSetIdProto initial_struct;
  for (auto it = mp_in.opset_import().begin(); it != mp_in.opset_import().end(); ++it) {
    if (it->domain() == "" || it->domain() == "ai.onnx") {
      initial_struct.set_version(it->version());
      break;
    }
  }
  OperatorSetIdProto target_struct;
  target_struct.set_version(target_version);
  DefaultVersionConverter v;
  return v.convert_version(mp_in, initial_struct, target_struct);
}

void DefaultVersionConverter::convert_graph(
    GraphProto* g,
    std::vector<OperatorSetIdProto>& opset_imports,
    const OperatorSetIdProto& initial_version,
    const OperatorSetIdProto& target_version) const {
  assertNonNull(g);

  // TODO: Move to Inter-Domain Converter
  // Get initial model versions
  // std::vector<OpSetID> initial_versions = g->opset_versions_mutable();

  // No conversion necessary if Model has single, equivalent opset version
  // if (initial_versions.size() == 1 && initial_versions[0].version ==
  //    target_version.version && initial_versions[0].domain ==
  //    target_version.domain) {
  //  return mp_in;
  // }

  // Check if versions are valid
  assertInVersionRange(initial_version.version());
  assertInVersionRange(target_version.version());

  // Iterate over all versions to target_version for specified
  int64_t curr_version = initial_version.version();
  int64_t step;
  if (target_version.version() > initial_version.version()) {
    step = 1;
  } else {
    step = -1;
  }
  // Identify index of this domain in g.opset_versions
  unsigned int domain_index = 0;
  for (unsigned int i = 0; i < opset_imports.size(); i++) {
    if (opset_imports[i].domain() == "") {
      domain_index = i;
    }
  }
  while (curr_version != target_version.version()) {
    debug(
        "curr_version: " + ONNX_NAMESPACE::to_string(curr_version) +
        ", next_version: " + ONNX_NAMESPACE::to_string(curr_version + step));
    Node* cur_op;
    // TODO: work with node in SSA order.
    for (NodeProto& cur_op : *g->mutable_node()) {
      // Iterate through and call adapter returned by adapter_lookup for ops from
      // current_version opset. We have to manipulate the iterator explicitly because cur_op
      // might change when applying the adapter (e.g. for deprecated ops)
      debug(std::string("Finding schema for ") + cur_op.op_type());
      if (cur_op.op_type() == "ConstantFill") {
        std::cerr
          << "Warning: skipping schema search for experimental op 'ConstantFill' and keeping the op as is. "
          "Please be advised the converted model may not be working properly if target runtime does not support this "
          "experimental op."
          << std::endl;
      }
      else if (cur_op.domain() != "" && cur_op.domain() != "ai.onnx") {
        std::cerr << "Warning: opset domain '" << cur_op.domain() << "' is not supported." << std::endl;
      }
      else if (cur_op.op_type() != "Undefined" && cur_op.op_type() != "Captured") {
        auto& op_domain_map = all_schemas.at(cur_op.op_type());
        OperatorSetIdProto curr_id = OpSetID(curr_version);
        OperatorSetIdProto next_id = OpSetID(curr_version + step);
        if (searchOpDomainMap(op_domain_map, curr_version, step)) {
          // Op is specifically defined for this domain and version
          auto& op_adapter = adapter_lookup(&cur_op, curr_id, next_id);
          // If adapter_lookup returns null, no adapter is present.
          // Error thrown by adapter_lookup
          if (DEBUG)
            std::cerr << "Applying adapter" << std::endl;
          // adapt should handle replacing node in graph. what to do with cur_op2?
          NodeProto* cur_op2 = op_adapter.adapt(g, &cur_op);
        }
        // Recursively convert any subgraph attributes
        for (auto& attr : *cur_op.mutable_attribute()) {
          if (attr.has_g()) {
            convert_graph(attr.mutable_g(), opset_imports, curr_id, next_id);
          }
        }
      }
    }
    // Update model version
    curr_version += step;
    opset_imports[domain_index].set_version(opset_imports[domain_index].version() + step);
  }
}

void make_model_proto_from_graph_and_opset_imports(ModelProto* p_m, GraphProto* g_p, std::vector<OperatorSetIdProto>& opset_imports) {
  *p_m->mutable_graph() = *g_p;
  // Add new opset_versions
  p_m->clear_opset_import();
  for (auto& opset : opset_imports) {
    *p_m->add_opset_import() = opset;
  }
}

GraphProto* make_graph_and_opset_imports_from_model_proto(const ModelProto& mp, std::vector<OperatorSetIdProto>& opset_imports) {
  if (!mp.has_ir_version()) {
    return nullptr;
  }
  else if (mp.ir_version() <= 1) {
    // ir_version=1 is not supported and ir_version=0 is illegal
    return nullptr;
  }

  GraphProto* g = new GraphProto(mp.graph());
  for (int i = 0; i < mp.opset_import_size(); i++) {
    opset_imports.push_back(mp.opset_import(i));
  }
  return g;
}

ModelProto DefaultVersionConverter::convert_version(
    const ModelProto& mp_in,
    const OperatorSetIdProto& initial_version,
    const OperatorSetIdProto& target_version) const {
  const std::string& initial_domain = initial_version.domain();
  const std::string& target_domain = target_version.domain();
  assertDefaultDomain(initial_domain, target_domain);

  for (auto it = mp_in.opset_import().begin(); it != mp_in.opset_import().end(); ++it) {
    if (it->domain() == initial_version.domain()) {
      ONNX_ASSERTM(
          initial_version.version() == it->version(), "initial_version does not reflect current state of model");
    }
  }

  std::vector<OperatorSetIdProto> opset_imports;
  GraphProto* g = make_graph_and_opset_imports_from_model_proto(mp_in, opset_imports);

  convert_graph(g, opset_imports, initial_version, target_version);

  // Export g as ModelProto
  debug("Finished conversion; returning model");
  ModelProto mp_out = PrepareOutput(mp_in);
  make_model_proto_from_graph_and_opset_imports(&mp_out, g, opset_imports);
  return mp_out;
}

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
