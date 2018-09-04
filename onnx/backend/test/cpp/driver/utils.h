#include <sstream>
#include <string>

#include "onnx/onnx_pb.h"
#include "onnx/onnxifi.h"
#include "onnx/proto_utils.h"

namespace onnx {
namespace testing {

template <typename T>
std::string to_string(T value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

onnxTensorDescriptorV1 ProtoToOnnxTensorDescriptor(
    const TensorProto& proto_tensor);
} // namespace testing
} // namespace onnx
