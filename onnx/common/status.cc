// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "status.h"

#include <cassert>
#include <memory>

#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {
namespace Common {

Status::Status(StatusCategory category, StatusCode code, const std::string& msg) {
  assert(StatusCode::OK != code);
  state_ = std::make_unique<State>(category, code, msg);
}

Status::Status(StatusCategory category, StatusCode code) : Status(category, code, EmptyString()) {}

bool Status::IsOK() const noexcept {
  return (state_ == nullptr);
}

StatusCategory Status::Category() const noexcept {
  return IsOK() ? StatusCategory::NONE : state_->category;
}

StatusCode Status::Code() const noexcept {
  return IsOK() ? StatusCode::OK : state_->code;
}

const std::string& Status::ErrorMessage() const {
  return IsOK() ? EmptyString() : state_->msg;
}

std::string Status::ToString() const {
  if (state_ == nullptr) {
    return std::string("OK");
  }

  std::string result;

  if (StatusCategory::CHECKER == state_->category) {
    result += "[CheckerError]";
  } else if (StatusCategory::OPTIMIZER == state_->category) {
    result += "[OptimizerError]";
  }

  result += " : ";
  result += ONNX_NAMESPACE::to_string(static_cast<int>(Code()));
  std::string msg;

  switch (Code()) {
    case StatusCode::INVALID_ARGUMENT:
      msg = "INVALID_ARGUMENT";
      break;
    case StatusCode::INVALID_PROTOBUF:
      msg = "INVALID_PROTOBUF";
      break;
    case StatusCode::FAIL:
      msg = "FAIL";
      break;
    default:
      msg = "GENERAL ERROR";
      break;
  }
  result += " : ";
  result += msg;
  result += " : ";
  result += state_->msg;

  return result;
}

const Status& Status::OK() noexcept {
  static Status s_ok;
  return s_ok;
}

const std::string& Status::EmptyString() {
  static std::string empty_str;
  return empty_str;
}

} // namespace Common
} // namespace ONNX_NAMESPACE
