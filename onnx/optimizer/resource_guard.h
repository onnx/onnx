#pragma once
#include <functional>

namespace onnx { namespace optimization {

class ResourceGuard {
  std::function<void()> _destructor;
  bool _released;

public:
  ResourceGuard(std::function<void()> destructor)
    : _destructor(std::move(destructor))
    , _released(false) {}

  ~ResourceGuard() {
    if (!_released) _destructor();
  }

  void release() {
    _released = true;
  }
};

}}
