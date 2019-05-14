// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <cmath>
#include <functional>
#include <numeric>
#include "onnx/common/assertions.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {

struct Tensor final {
private:
  bool is_segment_;
  int64_t segment_begin_;
  int64_t segment_end_;
  bool has_name_;
  std::string name_;
  int32_t elem_type_;
  std::vector<int64_t> sizes_;

  std::vector<float> float_data_;
  std::vector<double> double_data_;
  std::vector<int32_t> int32_data_;
  std::vector<int64_t> int64_data_;
  std::vector<uint64_t> uint64_data_;
  std::vector<std::string> string_data_;

  bool is_raw_data_;
  std::string raw_data_;

  template <typename F, typename T>
  void bin_func(const F& f, T* ptr, const T* a_ptr);

  template <typename F, typename T>
  void un_func(const F& f, T* ptr);

  template <typename T>
  void scale_dim(T* ptr, const T* s_ptr);

 public:
  Tensor()
  : is_segment_(false)
  , segment_begin_(0)
  , segment_end_(0)
  , has_name_(false)
  , elem_type_(ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED)
  , is_raw_data_(false)
  {}

  const std::vector<int64_t>& sizes() const {
    return sizes_;
  }
  std::vector<int64_t>& sizes() {
    return sizes_;
  }

  int64_t size_from_dim(int dim) const {
    if (dim < 0) {
      dim += (int)sizes_.size();
    }
    ONNX_ASSERT(dim >= 0 && (size_t)dim < sizes_.size());
    return std::accumulate(sizes_.begin() + dim, sizes_.end(), (int64_t)1, std::multiplies<int64_t>{});
  }

  int32_t elem_type() const {
    return elem_type_;
  }

  int32_t& elem_type() {
    return elem_type_;
  }

  std::vector<std::string>& strings() {
    return string_data_;
  }

  const std::vector<std::string>& strings() const {
    return string_data_;
  }

  std::vector<float>& floats() {
    return float_data_;
  }

  const std::vector<float>& floats() const {
    return float_data_;
  }

  std::vector<double>& doubles() {
    return double_data_;
  }

  const std::vector<double>& doubles() const {
    return double_data_;
  }

  std::vector<int32_t>& int32s() {
    return int32_data_;
  }

  const std::vector<int32_t>& int32s() const {
    return int32_data_;
  }

  std::vector<int64_t>& int64s() {
    return int64_data_;
  }

  const std::vector<int64_t>& int64s() const {
    return int64_data_;
  }

  std::vector<uint64_t>& uint64s() {
    return uint64_data_;
  }

  const std::vector<uint64_t>& uint64s() const {
    return uint64_data_;
  }

  const std::string& raw() const {
    return raw_data_;
  }

  void set_raw_data(std::string raw_data) {
    is_raw_data_ = true;
    raw_data_ = std::move(raw_data);
  }

  template <typename T>
  T* data();

  template <typename T>
  const T* data() const;

  bool is_segment() const {
    return is_segment_;
  }

  int64_t segment_begin() const {
    return segment_begin_;
  }

  int64_t segment_end() const {
    return segment_end_;
  }

  void set_segment_begin_and_end(int64_t begin, int64_t end) {
    is_segment_ = true;
    segment_begin_ = begin;
    segment_end_ = end;
  }

  bool hasName() const {
    return has_name_;
  }

  const std::string& name() const {
    return name_;
  }

  void setName(std::string name) {
    has_name_ = true;
    name_ = std::move(name);
  }

  bool is_raw_data() const {
    return is_raw_data_;
  }

  //this += a
  //Supported for
  //FLOAT, BOOL, INT8, INT16, INT32, UINT8, UINT16, INT64,
  //UINT32, UINT64, DOUBLE,
  //TODO: Support for FLOAT16, COMPLEX64, COMPLEX128
  void add(const Tensor& a);

  //this -= a
  //Supported for
  //FLOAT, BOOL, INT8, INT16, INT32, UINT8, UINT16, INT64,
  //UINT32, UINT64, DOUBLE
  //TODO: Support for FLOAT16, COMPLEX64, COMPLEX128
  void subtract(const Tensor& a);

  //this *= a
  //Supported for
  //FLOAT, BOOL, INT8, INT16, INT32, UINT8, UINT16, INT64,
  //UINT32, UINT64, DOUBLE
  //TODO: Support for FLOAT16, COMPLEX64, COMPLEX128
  void multiply(const Tensor& a);

  //this /= a
  //Supported for
  //FLOAT, INT8, INT16, INT32, UINT8, UINT16, INT64,
  //UINT32, UINT64, DOUBLE
  //TODO: Support for FLOAT16, COMPLEX64, COMPLEX128
  void divide(const Tensor& a);

  //Element-wise square root of This
  //Supported for
  //FLOAT, DOUBLE,
  //TODO: Support for FLOAT16
  void sqrt();

  //Element wise scaling of tensor s
  //s is one dimensional, has size M, where M is size of first dimension of tensor
  //s must have has data type corresponding to this
  //Supported for
  //FLOAT16, FLOAT, DOUBLE
  void scale_by_first_dim(const Tensor& s);
};

#define define_data(type, field)                  \
  template <>                                     \
  inline type* Tensor::data<type>() {             \
    if (is_raw_data_) {                           \
      return (type*)&raw_data_.data()[0];         \
    } else {                                      \
      return field.data();                        \
    }                                             \
  }                                               \
                                                  \
  template <>                                     \
  inline const type* Tensor::data<type>() const { \
    if (is_raw_data_) {                           \
      return (type*)(raw_data_.data());           \
    } else {                                      \
      return field.data();                        \
    }                                             \
  }

define_data(float, float_data_);
define_data(double, double_data_);
define_data(int32_t, int32_data_);
define_data(int64_t, int64_data_);
define_data(uint64_t, uint64_data_);
define_data(std::string, string_data_);
#undef define_data

template <typename F, typename T>
inline void Tensor::bin_func(const F& f, T* ptr, const T* a_ptr) {
  const int64_t num_elements = size_from_dim(0);
  for (int64_t i = 0; i < num_elements; ++i) {
    ptr[i] = f(ptr[i], a_ptr[i]);
  }
}

template <typename F, typename T>
inline void Tensor::un_func(const F& f, T* ptr) {
  const int64_t num_elements = size_from_dim(0);
  for (int64_t i = 0; i < num_elements; ++i) {
    ptr[i] = f(ptr[i]);
  }
}

template <typename T>
inline void Tensor::scale_dim(T* ptr, const T* s_ptr) {
  int64_t elems_per_first_dim = size_from_dim(1);
  int64_t first_dim_size = sizes_[0];
  int64_t counter = 0;
  for (int64_t i = 0; i < first_dim_size; ++i) {
    for (int64_t j = 0; j < elems_per_first_dim; ++j) {
      ptr[counter++] *= s_ptr[i];
    }
  }
}

#define APPLY_BINARY_FUNCTION(op_name, f)                                  \
  inline void Tensor::op_name(const Tensor& other) {                       \
    TENSOR_ASSERTM(                                                        \
        other.elem_type() == elem_type_,                                   \
        "Tensor types do not match: %s != %s",                             \
        to_string(elem_type_).c_str(),                                     \
        " vs. ",                                                           \
        to_string(other.elem_type()).c_str());                             \
    TENSOR_ASSERTM(other.sizes() == sizes_, "Tensor sizes do not match."); \
    switch (elem_type_) {                                                  \
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {                   \
        bin_func(f<float>(), data<float>(), other.data<float>());          \
        break;                                                             \
      }                                                                    \
      case ONNX_NAMESPACE::TensorProto_DataType_BOOL:                      \
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:                      \
      case ONNX_NAMESPACE::TensorProto_DataType_INT16:                     \
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:                     \
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:                     \
      case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {                  \
        bin_func(f<int32_t>(), data<int32_t>(), other.data<int32_t>());    \
        break;                                                             \
      }                                                                    \
      case ONNX_NAMESPACE::TensorProto_DataType_INT64: {                   \
        bin_func(f<int64_t>(), data<int64_t>(), other.data<int64_t>());    \
        break;                                                             \
      }                                                                    \
      case ONNX_NAMESPACE::TensorProto_DataType_UINT32:                    \
      case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {                  \
        bin_func(f<uint64_t>(), data<uint64_t>(), other.data<uint64_t>()); \
        break;                                                             \
      }                                                                    \
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {                  \
        bin_func(f<double>(), data<double>(), other.data<double>());       \
        break;                                                             \
      }                                                                    \
      default:                                                             \
        TENSOR_ASSERTM(                                                    \
            false,                                                         \
            "Operation %s not supported for data type %s",                 \
            #op_name,                                                      \
            " not supported for data type ",                               \
            to_string(elem_type_).c_str());                                \
    }                                                                      \
  }

APPLY_BINARY_FUNCTION(add, std::plus)
APPLY_BINARY_FUNCTION(subtract, std::minus)
APPLY_BINARY_FUNCTION(multiply, std::multiplies)
APPLY_BINARY_FUNCTION(divide, std::divides)

#undef APPLY_BINARY_FUNCTION

inline void Tensor::sqrt() {
  switch(elem_type_) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
      un_func<float (*)(float), float>(std::sqrt, data<float>());
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
      un_func<double (*)(double), double>(std::sqrt, data<double>());
      break;
    }
    default:
      TENSOR_ASSERTM(
          false,
          "Operation sqrt not supported for data type %s",
          to_string(elem_type_).c_str());
  }
}

inline void Tensor::scale_by_first_dim(const Tensor& other) {
  ONNX_ASSERT(
      sizes_.size() > 1 && other.sizes().size() == 1 &&
      other.sizes()[0] == sizes_[0]);
  ONNX_ASSERT(other.elem_type() == elem_type_);

  switch(elem_type_) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
      scale_dim(data<float>(), other.data<float>());
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
      scale_dim(data<int32_t>(), other.data<int32_t>());
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
      scale_dim(data<double>(), other.data<double>());
      break;
    }
    default:
      TENSOR_ASSERTM(
          false,
          "Operation scale_by_first_dim not supported for data type %s",
          to_string(elem_type_).c_str());
  }
}

} // namespace ONNX_NAMESPACE
