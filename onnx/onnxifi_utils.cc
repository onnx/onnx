#include "onnxifi_utils.h"

namespace ONNX_NAMESPACE {
namespace testing {

onnxTensorDescriptorV1 ProtoToOnnxTensorDescriptor(
    const ONNX_NAMESPACE::TensorProto& proto_tensor) {
  onnxTensorDescriptorV1 onnx_tensor;
  onnx_tensor.tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
  onnx_tensor.name = proto_tensor.name().c_str();
  onnx_tensor.dataType = proto_tensor.data_type();
  onnx_tensor.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
  std::vector<uint64_t> shape_values(
      proto_tensor.dims().begin(), proto_tensor.dims().end());
  onnx_tensor.dimensions = (uint32_t)shape_values.size();
  onnx_tensor.shape = shape_values.data();
  onnx_tensor.buffer = (onnxPointer)proto_tensor.raw_data().data();
  return onnx_tensor;
}
} // namespace testing
} // namespace ONNX_NAMESPACE
