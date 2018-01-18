#pragma once

#include <list>
#include <string>

namespace onnx { namespace optimization {

std::string Optimize(const std::string& content, std::list<std::string>& names);
std::string Split(const std::string& content, bool init, bool predict);

}} // namespace onnx::optimization
