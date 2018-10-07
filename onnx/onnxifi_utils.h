#include <sstream>
#include <string>

#include "onnx/onnx_pb.h"
#include "onnx/onnxifi.h"
#include "onnx/proto_utils.h"

namespace ONNX_NAMESPACE {
namespace testing {

onnxTensorDescriptorV1 ProtoToOnnxTensorDescriptor(
    const ONNX_NAMESPACE::TensorProto& proto_tensor,
    std::vector<std::vector<uint64_t>>& shape_pool);
} // namespace testing
} // namespace ONNX_NAMESPACE
