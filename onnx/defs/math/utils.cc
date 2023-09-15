/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/math/utils.h"

#include <string>

#include "onnx/defs/shape_inference.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {
template <typename T>
T GetScalarValueFromTensor(const TensorProto* t) {
  if (t == nullptr) {
    return T{};
  }

  auto data_type = t->data_type();
  switch (data_type) {
    case TensorProto::FLOAT:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<float>(t).at(0));
    case TensorProto::DOUBLE:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<double>(t).at(0));
    case TensorProto::INT32:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<int32_t>(t).at(0));
    case TensorProto::INT64:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<int64_t>(t).at(0));
    default:
      fail_shape_inference("Unsupported input data type of ", data_type);
  }
}
} // namespace ONNX_NAMESPACE
