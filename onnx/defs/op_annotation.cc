#include "onnx/defs/op_annotation.h"

namespace ONNX_NAMESPACE {
std::shared_ptr<OpAnnotationRegistry> OpAnnotationRegistry::instance_ =
    std::shared_ptr<OpAnnotationRegistry>(new OpAnnotationRegistry());
} // namespace ONNX_NAMESPACE