#include "onnx/py_utils.h" // Keep this at the top to avoid the compilation warning.

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>

#include "onnx/ir_pb_converter.h"
#include "onnx/optimizer/optimize.h"
#include "onnx/optimizer/wrapper.h"

namespace onnx { namespace optimization {

void PrepareOutput(const onnx::ModelProto& mp_in, onnx::ModelProto& mp_out) {
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

std::string Optimize(const std::string& content, std::list<std::string>& names) {
  onnx::ModelProto mp_in;
  ParseProtoFromBytes(&mp_in, content.c_str(), content.size());
  onnx::ModelProto mp_out;
  PrepareOutput(mp_in, mp_out);

  std::shared_ptr<onnx::Graph> g = onnx::ImportModelProto(mp_in);
  std::string out;
  if (g.get() == nullptr) {
    std::cerr << "Warning: onnx optimizer is unable to parse input model" << std::endl;
    // If we can't parse the file, just return the original content.
    out = content;
  } else {
    for (auto s : names) {
      if (s == "fuse_consecutive_transposes") {
        fuse_consecutive_transposes(g);
      } else if (s == "eliminate_nop_transpose") {
        eliminate_nop_transpose(g);
      } else if (s == "fuse_transpose_into_gemm") {
        fuse_transpose_into_gemm(g);
      } else{
        std::cerr << "Warning: onnx optimier doesn't recognize pass name "
          << s << std::endl;
      }
    }
    ExportModelProto(&mp_out, g);
    mp_out.SerializeToString(&out);
  }
  return out;
}

std::string Split(const std::string& content, bool init, bool predict) {
  onnx::ModelProto mp_in;
  ParseProtoFromBytes(&mp_in, content.c_str(), content.size());
  onnx::ModelProto mp_out;
  PrepareOutput(mp_in, mp_out);

  std::shared_ptr<onnx::Graph> g = onnx::ImportModelProto(mp_in);
  std::string out;
  if (g.get() == nullptr) {
    std::cerr << "Warning: onnx optimizer is unable to parse input model" << std::endl;

    // If we can't parse the file, give an empty init graph and copy
    // the input graph into the predict graph.
    if (init) {
      mp_out.SerializeToString(&out);
    } else {
      out = content;
    }
  } else {
    split_init_and_predict(g, init, predict);
    ExportModelProto(&mp_out, g);
    mp_out.SerializeToString(&out);
  }
  return out;
}

}} // namespace onnx::optimization
