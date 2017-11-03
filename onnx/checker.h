#pragma once

#include <stdexcept>
#include "onnx/onnx.pb.h"

namespace onnx {
namespace checker {
class ValidationError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

using IR_VERSION_TYPE = decltype(Version::IR_VERSION);
void check_value_info(const ValueInfoProto& value_info, int ir_version);
void check_tensor(const TensorProto& tensor, int ir_version);
void check_attribute(const AttributeProto& attr, int ir_version);
void check_node(const NodeProto& node, int ir_version);
void check_graph(const GraphProto& graph, int ir_version);
void check_model(const ModelProto& model, int ir_version);
} // namespace checker
} // namespace onnx
