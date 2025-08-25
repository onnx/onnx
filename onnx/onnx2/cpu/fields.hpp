// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "onnx/onnx2/cpu/common_helpers.h"

namespace onnx2 {
namespace utils {

template <typename T>
std::vector<std::string> RepeatedField<T>::PrintToVectorString(utils::PrintOptions& options) const {
  std::vector<std::string> rows{"["};
  for (const auto& p : values_) {
    std::vector<std::string> r = p.PrintToVectorString(options);
    for (size_t i = 0; i < r.size(); ++i) {
      if (i + 1 == r.size()) {
        rows.push_back(common_helpers::MakeString("  ", r[i], ","));
      } else {
        rows.push_back(common_helpers::MakeString("  ", r[i]));
      }
    }
  }
  rows.push_back("],");
  return rows;
}

template <typename T>
void RepeatedProtoField<T>::clear() {
  for (auto& p : values_)
    p.reset();
  values_.clear();
}

template <typename T>
inline T& RepeatedProtoField<T>::operator[](size_t index) {
  return *values_[index];
}

template <typename T>
inline const T& RepeatedProtoField<T>::operator[](size_t index) const {
  return *values_[index];
}

template <typename T>
void RepeatedProtoField<T>::push_back(const T& v) {
  add().CopyFrom(v);
}

template <typename T>
void RepeatedProtoField<T>::extend(const RepeatedProtoField<T>& v) {
  values_.reserve(values_.size() + v.values_.size());
  for (size_t i = 0; i < v.size(); ++i)
    push_back(v[i]);
}

template <typename T>
void RepeatedProtoField<T>::extend(const RepeatedProtoField<T>&& v) {
  values_.reserve(values_.size() + v.values_.size());
  for (size_t i = 0; i < v.size(); ++i) {
    values_.emplace_back(simple_unique_ptr<T>(nullptr));
    values_.back().swap(v.get(i));
  }
  v.values_.clear();
}

template <typename T>
T& RepeatedProtoField<T>::add() {
  values_.emplace_back(simple_unique_ptr<T>(new T));
  return back();
}

template <typename T>
T& RepeatedProtoField<T>::back() {
  EXT_ENFORCE(!values_.empty(), "Cannot call back() on an empty RepeatedField.");
  return *values_.back();
}

template <typename T>
std::vector<std::string> RepeatedProtoField<T>::PrintToVectorString(utils::PrintOptions& options) const {
  std::vector<std::string> rows{"["};
  for (const auto& p : values_) {
    std::vector<std::string> r = p->PrintToVectorString(options);
    for (size_t i = 0; i < r.size(); ++i) {
      if (i + 1 == r.size()) {
        rows.push_back(common_helpers::MakeString("  ", r[i], ","));
      } else {
        rows.push_back(common_helpers::MakeString("  ", r[i]));
      }
    }
  }
  rows.push_back("],");
  return rows;
}

template <typename T>
void OptionalField<T>::reset() {
  value_.reset();
}

template <typename T>
void OptionalField<T>::set_empty_value() {
  value_.reset(new T);
}

template <typename T>
T& OptionalField<T>::operator*() {
  EXT_ENFORCE(has_value(), "Optional field has no value.");
  return *value_;
}

template <typename T>
const T& OptionalField<T>::operator*() const {
  EXT_ENFORCE(has_value(), "Optional field has no value.");
  return *value_;
}

template <typename T>
OptionalField<T>& OptionalField<T>::operator=(const T& v) {
  // We make a copy.
  set_empty_value();
  StringWriteStream stream;
  SerializeOptions opts;
  v.SerializeToStream(stream, opts);
  StringStream rstream(stream.data(), stream.size());
  ParseOptions ropts;
  value_->ParseFromStream(rstream, ropts);
  return *this;
}

template <typename T>
OptionalField<T>& OptionalField<T>::operator=(const OptionalField<T>& v) {
  // We make a copy.
  reset();
  if (v.has_value()) {
    set_empty_value();
    StringWriteStream stream;
    SerializeOptions opts;
    (*v).SerializeToStream(stream, opts);
    StringStream rstream(stream.data(), stream.size());
    ParseOptions ropts;
    value_->ParseFromStream(rstream, ropts);
  }
  return *this;
}

} // namespace utils
} // namespace onnx2
