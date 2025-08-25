// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "onnx/onnx2/cpu/common_helpers.h"
#include "onnx/onnx2/cpu/simple_string.h"

namespace onnx2 {
namespace utils {

struct PrintOptions {
  /** if true, raw data will not be printed but skipped, tensors are not valid in that case but the
   * model structure is still available */
  bool skip_raw_data = false;
  /** if skip_raw_data is true, raw data will be printed only if it is larger than the threshold */
  int64_t raw_data_threshold = 1024;
};

template <typename T>
class simple_unique_ptr {
 public:
  explicit inline simple_unique_ptr(T* ptr = nullptr) : ptr_(ptr) {}
  inline ~simple_unique_ptr() {
    delete ptr_;
  }
  inline simple_unique_ptr(simple_unique_ptr&& other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }
  inline simple_unique_ptr& operator=(simple_unique_ptr&& other) noexcept {
    if (this != &other) {
      delete ptr_;
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }
  inline bool isnull() const {
    return ptr_ == nullptr;
  }
  inline bool operator==(const simple_unique_ptr& other) const {
    return ptr_ == other.ptr_;
  }
  inline bool operator!=(const simple_unique_ptr& other) const {
    return ptr_ != other.ptr_;
  }
  simple_unique_ptr(const simple_unique_ptr&) {
    EXT_THROW("simple_unique_ptr cannot be copied, only moved (1).");
  }
  simple_unique_ptr& operator=(const simple_unique_ptr&) {
    EXT_THROW("simple_unique_ptr cannot be copied, only moved (2).");
  }
  inline T* get() const {
    return ptr_;
  }
  inline T& operator*() const {
    return *ptr_;
  }
  inline T* operator->() const {
    return ptr_;
  }
  inline T& reset_and(T* new_ptr) {
    EXT_ENFORCE(new_ptr != nullptr, "cannot you simple_unique_ptr::reset_and with a null pointer.");
    reset(new_ptr);
    return *this;
  }
  inline void reset(T* new_ptr = nullptr) {
    delete ptr_;
    ptr_ = new_ptr;
  }

 private:
  T* ptr_;
};

template <typename T>
class RepeatedField {
 public:
  explicit inline RepeatedField() {}
  inline void reserve(size_t n) {
    values_.reserve(n);
  }
  inline void clear() {
    values_.clear();
  }
  inline bool empty() const {
    return values_.empty();
  }
  inline size_t size() const {
    return values_.size();
  }
  inline T& operator[](size_t index) {
    return values_[index];
  }
  inline const std::vector<T>& values() const {
    return values_;
  }
  inline std::vector<T>& mutable_values() {
    return values_;
  }
  inline const T& operator[](size_t index) const {
    return values_[index];
  }
  inline void remove_range(size_t start, size_t stop, size_t step) {
    EXT_ENFORCE(step == 1, "remove_range not implemented for step=", static_cast<int>(step));
    EXT_ENFORCE(start == 0, "remove_range not implemented for start=", static_cast<int>(start));
    EXT_ENFORCE(
        stop == size(),
        "remove_range not implemented for stop=",
        static_cast<int>(stop),
        " and size=",
        static_cast<int>(size()));
    clear();
  }
  inline void push_back(const T& v) {
    values_.push_back(v);
  }
  inline void extend(const std::vector<T>& v) {
    values_.insert(values_.end(), v.begin(), v.end());
  }
  inline void extend(const RepeatedField<T>& v) {
    values_.insert(values_.end(), v.begin(), v.end());
  }
  inline T& add() {
    values_.emplace_back(T());
    return values_.back();
  }
  inline T& back() {
    return values_.back();
  }
  inline typename std::vector<T>::iterator begin() {
    return values_.begin();
  }
  inline typename std::vector<T>::iterator end() {
    return values_.end();
  }
  inline typename std::vector<T>::const_iterator begin() const {
    return values_.begin();
  }
  inline typename std::vector<T>::const_iterator end() const {
    return values_.end();
  }
  template <class... Args>
  inline void emplace_back(Args&&... args) {
    values_.emplace_back(std::forward<Args>(args)...);
  }
  std::vector<std::string> PrintToVectorString(PrintOptions& options) const;

 private:
  std::vector<T> values_;
};

template <typename T>
class RepeatedProtoField {
 public:
  explicit inline RepeatedProtoField() {}
  inline void reserve(size_t n) {
    values_.reserve(n);
  }
  inline bool empty() const {
    return values_.empty();
  }
  inline size_t size() const {
    return values_.size();
  }
  inline T& operator[](size_t index);
  inline const T& operator[](size_t index) const;
  inline simple_unique_ptr<T>& get(size_t index) {
    return values_[index];
  }
  inline void remove_range(size_t start, size_t stop, size_t step) {
    EXT_ENFORCE(step == 1, "remove_range not implemented for step=", static_cast<int>(step));
    EXT_ENFORCE(start == 0, "remove_range not implemented for start=", static_cast<int>(start));
    EXT_ENFORCE(
        stop == size(),
        "remove_range not implemented for stop=",
        static_cast<int>(stop),
        " and size=",
        static_cast<int>(size()));
    clear();
  }

  void clear();
  void push_back(const T& v);
  void extend(const std::vector<T>& v);
  void extend(const RepeatedProtoField<T>& v);
  void extend(const RepeatedProtoField<T>&& v);
  T& add();
  T& back();
  std::vector<std::string> PrintToVectorString(PrintOptions& options) const;

  class iterator {
   private:
    RepeatedProtoField<T>* parent_;
    size_t pos_;

   public:
    explicit iterator(RepeatedProtoField<T>* parent, size_t pos = 0) : parent_(parent), pos_(pos) {}
    iterator& operator++() {
      ++pos_;
      return *this;
    }
    bool operator==(const iterator& other) const {
      return pos_ == other.pos_ && parent_ == other.parent_;
    }
    bool operator!=(const iterator& other) const {
      return !(*this == other);
    }
    T& operator*() {
      return (*parent_)[pos_];
    }
  };
  inline iterator begin() {
    return iterator(this, 0);
  }
  inline iterator end() {
    return iterator(this, size());
  }

 private:
  std::vector<simple_unique_ptr<T>> values_;
};

template <typename T>
class OptionalField {
 public:
  explicit inline OptionalField() : value_(nullptr) {}
  explicit inline OptionalField(const OptionalField<T>& copy) : value_(nullptr) {
    *this = copy;
  }
  explicit inline OptionalField(OptionalField<T>&& move) : value_(move.value_) {
    move.reset();
  }
  inline bool has_value() const {
    return !value_.isnull();
  }
  inline void reset();
  T& operator*();
  const T& operator*() const;
  OptionalField<T>& operator=(const T& other);
  OptionalField<T>& operator=(const OptionalField<T>& other);
  inline OptionalField<T>& operator=(OptionalField<T>&& other) {
    value_ = other.value_;
    return *this;
  }
  void set_empty_value();

 private:
  simple_unique_ptr<T> value_;
};

template <typename T>
class _OptionalField {
 public:
  explicit inline _OptionalField() {}
  inline bool has_value() const {
    return value_.has_value();
  }
  inline void reset() {
    value_.reset();
  }
  inline const T& operator*() const {
    return *value_;
  }
  inline T& operator*() {
    return *value_;
  }
  inline bool operator==(const _OptionalField<T>& v) const {
    return value_ == v;
  }
  inline bool operator==(const T& v) const {
    return value_ == v;
  }
  inline _OptionalField<T>& operator=(const T& other) {
    value_ = other;
    return *this;
  }
  inline void set_empty_value() {
    value_ = static_cast<T>(0);
  }

 protected:
  std::optional<T> value_;
};

template <>
class OptionalField<int64_t> : public _OptionalField<int64_t> {
 public:
  explicit inline OptionalField() : _OptionalField<int64_t>() {}
  inline OptionalField<int64_t>& operator=(const int64_t& other) {
    value_ = other;
    return *this;
  }
};

template <>
class OptionalField<int32_t> : public _OptionalField<int32_t> {
 public:
  explicit inline OptionalField() : _OptionalField<int32_t>() {}
  inline OptionalField<int32_t>& operator=(const int32_t& other) {
    value_ = other;
    return *this;
  }
};

template <>
class OptionalField<float> : public _OptionalField<float> {
 public:
  explicit inline OptionalField() : _OptionalField<float>() {}
  inline OptionalField<float>& operator=(const float& other) {
    value_ = other;
    return *this;
  }
};

template <typename T>
class OptionalEnumField {
 public:
  explicit inline OptionalEnumField() {}
  inline bool has_value() const {
    return value_.has_value();
  }
  inline void reset() {
    value_.reset();
  }
  inline const T& operator*() const {
    return *value_;
  }
  inline T& operator*() {
    return *value_;
  }
  inline bool operator==(const OptionalEnumField<T>& v) const {
    return value_ == v;
  }
  inline bool operator==(const T& v) const {
    return value_ == v;
  }
  inline OptionalEnumField<T>& operator=(const T& other) {
    value_ = other;
    return *this;
  }
  inline void set_empty_value() {
    value_ = static_cast<T>(0);
  }

 protected:
  std::optional<T> value_;
};

} // namespace utils
} // namespace onnx2
