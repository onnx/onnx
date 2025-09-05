// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstring>
#include <string>
#include <vector>

#include "onnx/onnx2/cpu/common_helpers.h"

namespace ONNX_NAMESPACE {
namespace v2 {
namespace utils {

class String;

class RefString {
 private:
  const char* ptr_;
  size_t size_;

 public:
  explicit inline RefString(const RefString& copy) : ptr_(copy.ptr_), size_(copy.size_) {}
  explicit inline RefString(const char* ptr, size_t size) : ptr_(ptr), size_(size) {}
  inline RefString& operator=(const RefString& v) {
    ptr_ = v.ptr_;
    size_ = v.size_;
    return *this;
  }
  RefString& operator=(const String& v);
  inline size_t size() const {
    return size_;
  }
  inline const char* c_str() const {
    return ptr_;
  }
  inline const char* data() const {
    return ptr_;
  }
  inline bool empty() const {
    return size_ == 0;
  }
  inline char operator[](size_t i) const {
    return ptr_[i];
  }
  bool operator==(const RefString& other) const;
  bool operator==(const String& other) const;
  bool operator==(const std::string& other) const;
  bool operator==(const char* other) const;
  bool operator!=(const RefString& other) const;
  bool operator!=(const String& other) const;
  bool operator!=(const std::string& other) const;
  bool operator!=(const char* other) const;
  std::string as_string(bool quote = false) const;
  int64_t toint64() const;
};

class String {
 private:
  char* ptr_;
  size_t size_;

 public:
  inline ~String() {
    clear();
  }
  inline void clear() {
    if (ptr_ != nullptr) {
      delete[] ptr_;
      ptr_ = nullptr;
    }
    size_ = 0;
  }
  explicit inline String() : ptr_(nullptr), size_(0) {}
  explicit inline String(const RefString& s) {
    set(s.data(), s.size());
  }
  explicit inline String(const char* ptr, size_t size) {
    set(ptr, size);
  }
  explicit String(const std::string& s) {
    set(s.data(), s.size());
  }
  explicit String(const String& s) {
    set(s.data(), s.size());
  }
  explicit String(String&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  inline size_t size() const {
    return size_;
  }
  inline const char* data() const {
    return ptr_;
  }
  inline bool empty() const {
    return size_ == 0;
  }
  inline bool null() const {
    return size_ == 0 && ptr_ == nullptr;
  }
  inline char operator[](size_t i) const {
    return ptr_[i];
  }
  String& operator=(const char* s);
  String& operator=(const RefString& s);
  String& operator=(const String& s);
  String& operator=(const std::string& s);
  bool operator==(const std::string& other) const;
  bool operator==(const String& other) const;
  bool operator==(const RefString& other) const;
  bool operator==(const char* other) const;
  bool operator!=(const std::string& other) const;
  bool operator!=(const String& other) const;
  bool operator!=(const RefString& other) const;
  bool operator!=(const char* other) const;
  std::string as_string(bool quote = false) const;
  inline int64_t toint64() const {
    return RefString(ptr_, size_).toint64();
  }

 private:
  void set(const char* ptr, size_t size);
};

inline RefString& RefString::operator=(const String& v) {
  size_ = v.size();
  ptr_ = v.data();
  return *this;
}

std::string join_string(const std::vector<std::string>& rows, const char* delimiter = "\n");

} // namespace utils
} // namespace v2
} // namespace ONNX_NAMESPACE
