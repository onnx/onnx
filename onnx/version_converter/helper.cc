/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Helper Methods for Adapters

#include "onnx/version_converter/helper.h"
#include "onnx/common/assertions.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

bool HasAttribute(NodeProto* node_proto, const std::string& name) {
  return std::find_if(node_proto->attribute().begin(), node_proto->attribute().end(), [name](const AttributeProto& attr) {
           return attr.name() == name;
         }) != node_proto->attribute().end();
}

template <typename T>
std::vector<T> protobuf_repeated_fields_to_vector(google::protobuf::RepeatedPtrField<T>& repeated) {
  std::vector<T> vector(repeated.size());
  std::copy(repeated.begin(), repeated.end(), vector.begin());
}

int check_numpy_unibroadcastable_and_require_broadcast(
    std::vector<TensorShapeProto::Dimension>& dim1,
    std::vector<TensorShapeProto::Dimension>& dim2) {
  if (dim1.size() < dim2.size())
    return -1;
  // Check that axis is input1_sizes.size()-input2_sizes.size()
  bool broadcast = false;
  int axis = (int)(dim1.size() - dim2.size());
  for (int i = 0; i < (int)dim2.size(); i++) {
    if (dim2[i].dim_value() != dim1[axis + i].dim_value() && dim2[i].dim_value() != 1)
      return -1;
    if (dim2[i].dim_value() != dim1[axis + i].dim_value())
      broadcast = true;
  }
  // Return true if broadcasting is required
  if (dim1.size() > dim2.size() || broadcast)
    return 1;
  else
    return 0;
}
  

void assert_numpy_multibroadcastable(
    const std::vector<TensorShapeProto::Dimension>& input1_sizes,
    const std::vector<TensorShapeProto::Dimension>& input2_sizes) {
  // Generalize above for multibroadcastable case
  const std::vector<TensorShapeProto::Dimension>* A_ptr;
  const std::vector<TensorShapeProto::Dimension>* B_ptr;
  int A;
  int B;
  if (input1_sizes.size() < input2_sizes.size()) {
    A_ptr = &input2_sizes;
    B_ptr = &input1_sizes;
    A = 2;
    B = 1;
  } else {
    A_ptr = &input1_sizes;
    B_ptr = &input2_sizes;
    A = 1;
    B = 2;
  }
  const std::vector<TensorShapeProto::Dimension>& A_sizes = *A_ptr;
  const std::vector<TensorShapeProto::Dimension>& B_sizes = *B_ptr;
  int axis = (int)(A_sizes.size() - B_sizes.size());
  for (int i = 0; i < (int)B_sizes.size(); i++) {
    ONNX_ASSERTM(
        B_sizes[i].dim_value() == A_sizes[axis + i].dim_value() || B_sizes[i].dim_value() == 1 || A_sizes[axis + i].dim_value() == 1,
        "Dimension %d of input %d does not match "
        "dimension %d of input %d, and neither's value is 1",
        i,
        B,
        axis + i,
        A);
  }
}

void assertNotParams(const std::vector<TensorShapeProto::Dimension>& sizes) {
  for (const TensorShapeProto::Dimension& dim : sizes) {
    ONNX_ASSERTM(dim.has_dim_value(), "%s Dimension is a param instead of an int.", dim.dim_param().c_str());
  }
}

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
