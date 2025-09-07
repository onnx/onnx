// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>

#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {
namespace Common {

enum class StatusCategory : std::uint8_t {
  NONE = 0,
  CHECKER = 1,
  OPTIMIZER = 2,
};

enum class StatusCode : std::uint8_t {
  OK = 0,
  FAIL = 1,
  INVALID_ARGUMENT = 2,
  INVALID_PROTOBUF = 3,
};

class Status {
 public:
  Status() noexcept = default;

  Status(StatusCategory category, StatusCode code, const std::string& msg);

  Status(StatusCategory category, StatusCode code);

  Status(const Status& other) {
    *this = other;
  }

  Status& operator=(const Status& other) {
    if (&other != this) {
      if (nullptr == other.state_) {
        state_.reset();
      } else if (state_ != other.state_) {
        state_ = std::make_unique<State>(*other.state_);
      }
    }
    return *this;
  }

  Status(Status&&) = default;
  Status& operator=(Status&&) = default;
  ~Status() = default;

  ONNX_API bool IsOK() const noexcept;

  ONNX_API StatusCode Code() const noexcept;

  ONNX_API StatusCategory Category() const noexcept;

  ONNX_API const std::string& ErrorMessage() const;

  ONNX_API std::string ToString() const;

  bool operator==(const Status& other) const {
    return (this->state_ == other.state_) || (ToString() == other.ToString());
  }

  bool operator!=(const Status& other) const {
    return !(*this == other);
  }

  ONNX_API static const Status& OK() noexcept;

 private:
  struct State {
    State(StatusCategory cat_, StatusCode code_, std::string msg_)
        : category(cat_), code(code_), msg(std::move(msg_)) {}

    StatusCategory category = StatusCategory::NONE;
    StatusCode code{};
    std::string msg;
  };

  static const std::string& EmptyString();

  // state_ == nullptr when if status code is OK.
  std::unique_ptr<State> state_;
};

inline std::ostream& operator<<(std::ostream& out, const Status& status) {
  return out << status.ToString();
}

} // namespace Common
} // namespace ONNX_NAMESPACE
