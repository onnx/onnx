#pragma once

#include <string>

namespace onnx { namespace optimization {

template <typename Proto>
bool ParseProtoFromBytes(Proto* proto, const char* buffer, size_t length);

std::string Optimize(const std::string& content, bool init, bool predict);

}} // namespace onnx::optimization
