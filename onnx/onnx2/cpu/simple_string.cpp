#include "simple_string.h"
#include <sstream>

namespace onnx2 {
namespace utils {

bool RefString::operator==(const char *other) const {
  if (size_ == 0)
    return other == nullptr || other[0] == 0;
  if (other == nullptr)
    return false;
  size_t i;
  for (i = 0; i < size_ && ptr_[i] == other[i] && other[i] != 0; ++i)
    ;
  return i == size_ && other[i] == 0;
}

bool RefString::operator==(const RefString &other) const {
  if (size() != other.size())
    return false;
  if (size() == 0)
    return true;
  if (data() == other.data())
    return true;
  size_t i;
  for (i = 0; i < size_ && ptr_[i] == other[i]; ++i)
    ;
  return i == size_;
}

bool RefString::operator==(const std::string &other) const {
  return *this == RefString(other.data(), other.size());
}

bool RefString::operator==(const String &other) const {
  return *this == RefString(other.data(), other.size());
}

bool RefString::operator!=(const std::string &other) const { return !(*this == other); }
bool RefString::operator!=(const String &other) const { return !(*this == other); }
bool RefString::operator!=(const RefString &other) const { return !(*this == other); }
bool RefString::operator!=(const char *other) const { return !(*this == other); }

std::string RefString::as_string(bool quote) const {
  if (empty())
    return quote ? std::string("\"\"") : std::string();
  auto s = std::string(data(), size());
  return quote ? std::string("\"") + s + std::string("\"") : s;
}

int64_t RefString::toint64() const { return std::stoll(as_string()); }

void String::set(const char *ptr, size_t size) {
  if (size == SIZE_MAX) {
    if (ptr == nullptr) {
      size = 0;
    } else {
      const char *p = ptr;
      size = 0;
      for (; *p != 0; ++p, ++size)
        ;
    }
  }
  if (ptr == nullptr) {
    ptr_ = nullptr;
    size_ = 0;
  } else if (ptr[size - 1] == 0) {
    if (size == 0) {
      ptr_ = new char[1];
      *ptr_ = 0;
      size_ = 0;
    } else {
      ptr_ = new char[size - 1];
      memcpy(ptr_, ptr, size - 1);
      size_ = size - 1;
    }
  } else {
    ptr_ = new char[size];
    memcpy(ptr_, ptr, size);
    size_ = size;
  }
}

bool String::operator==(const char *other) const {
  if (size_ == 0)
    return other == nullptr || other[0] == 0;
  if (other == nullptr)
    return false;
  size_t i;
  for (i = 0; i < size_ && ptr_[i] == other[i] && other[i] != 0; ++i)
    ;
  return i == size_ && other[i] == 0;
}

bool String::operator==(const RefString &other) const {
  if (size() != other.size())
    return false;
  if (size() == 0)
    return true;
  size_t i;
  for (i = 0; i < size_ && ptr_[i] == other[i]; ++i)
    ;
  return i == size_;
}

bool String::operator==(const String &other) const {
  return *this == RefString(other.data(), other.size());
}

bool String::operator==(const std::string &other) const {
  return *this == RefString(other.data(), other.size());
}

bool String::operator!=(const std::string &other) const { return !(*this == other); }
bool String::operator!=(const String &other) const { return !(*this == other); }
bool String::operator!=(const RefString &other) const { return !(*this == other); }
bool String::operator!=(const char *other) const { return !(*this == other); }

std::string String::as_string(bool quote) const {
  if (empty())
    return quote ? std::string("\"\"") : std::string();
  auto s = std::string(data(), size());
  return quote ? std::string("\"") + s + std::string("\"") : s;
}

std::string join_string(const std::vector<std::string> &elements, const char *delimiter) {
  std::stringstream oss;
  auto it = elements.begin();
  if (it != elements.end()) {
    oss << *it;
    ++it;
  }
  while (it != elements.end()) {
    oss << delimiter << *it;
    ++it;
  }
  return oss.str();
}

String &String::operator=(const char *s) {
  EXT_ENFORCE(s != data(), "Cannot assign to self.");
  clear();
  set(s, SIZE_MAX);
  return *this;
}

String &String::operator=(const RefString &s) {
  if (ptr_ == s.data() && size_ == s.size())
    return *this; // no change
  EXT_ENFORCE(s.data() != data(), "Cannot assign to self when size is different.");
  clear();
  set(s.data(), s.size());
  return *this;
}

String &String::operator=(const String &s) {
  if (ptr_ == s.data() && size_ == s.size())
    return *this; // no change
  EXT_ENFORCE(s.data() != data(), "Cannot assign to self when size is different.");
  clear();
  set(s.data(), s.size());
  return *this;
}

String &String::operator=(const std::string &s) {
  clear();
  set(s.data(), s.size());
  return *this;
}

} // namespace utils
} // namespace onnx2
