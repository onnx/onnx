// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/onnx_pb.h"
#include <math.h>
#include <numeric>
#include <functional>
#include <stdexcept>


namespace ONNX_NAMESPACE {

class TensorError final : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Tensor final {
private:
  bool is_segment_;
  int64_t segment_begin_;
  int64_t segment_end_;
  bool has_name_;
  std::string name_;
  ONNX_NAMESPACE::TensorProto_DataType elem_type_;
  std::vector<int64_t> sizes_;

  std::vector<float> float_data_;
  std::vector<double> double_data_;
  std::vector<int32_t> int32_data_;
  std::vector<int64_t> int64_data_;
  std::vector<uint64_t> uint64_data_;
  std::vector<std::string> string_data_;

  bool is_raw_data_;
  std::string raw_data_;

  template<typename F, typename T>
  void bin_func(F f, T* ptr, const T* a_ptr);

  template<typename F, typename T>
  void un_func(F f, T* ptr);

  template<typename T>
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

  ONNX_NAMESPACE::TensorProto_DataType elem_type() const {
    return elem_type_;
  }

  ONNX_NAMESPACE::TensorProto_DataType& elem_type() {
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

  void set_raw_data(const char* raw_data, size_t size) {
    set_raw_data({raw_data, size});
  }

  void set_raw_data(const char* raw_data) {
    set_raw_data(raw_data, raw_data_.size());
  }

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

#define CONST_DATA(owner, type, vec)                                           \
  const type* owner##_const_data_ptr;                                          \
  if (owner->is_raw_data())  {                                                 \
    owner##_const_data_ptr = (const type*) owner->raw().c_str();               \
  } else {                                                                     \
    owner##_const_data_ptr = (const type*) owner->vec().data();                \
  }                                                                            \


#define DATA(owner, type, vec)                                                 \
  type* owner##_data_ptr;                                                      \
  std::vector<type> vals;                                                      \
  if (owner->is_raw_data())  {                                                 \
    for (size_t i = 0; i < raw_data_.size(); i += sizeof(type))  {             \
        vals.push_back(*((const type*)(owner->raw().c_str() + i)));            \
    }                                                                          \
    owner##_data_ptr = (type*) vals.data();                                    \
  } else {                                                                     \
    owner##_data_ptr = (type*) owner->vec().data();                            \
  }                                                                            \


#define SET_RAW_DATA(ptr)                                                      \
  if (is_raw_data_)  {                                                         \
    set_raw_data((const char*)(ptr));                                          \
  }                                                                            \



template<typename F, typename T>
inline void Tensor::bin_func(F f, T* ptr, const T* a_ptr) {
  int64_t num_elements = std::accumulate(sizes_.begin(), sizes_.end(),
                                        (int64_t) 1, std::multiplies<int64_t>());
  for (int64_t i = 0; i < num_elements; ++i) {
    ptr[i] = f(ptr[i], a_ptr[i]);
  }
  SET_RAW_DATA(ptr)
}

template<typename F, typename T>
inline void Tensor::un_func(F f, T* ptr)  {
  int64_t num_elements = std::accumulate(sizes_.begin(), sizes_.end(),
                                        (int64_t) 1, std::multiplies<int64_t>());
  for (int64_t i = 0; i < num_elements; ++i) {
    ptr[i] = f(ptr[i]);
  }
  SET_RAW_DATA(ptr)
}

template<typename T>
inline void Tensor::scale_dim(T* ptr, const T* s_ptr)  {
  int64_t elems_per_first_dim = std::accumulate(sizes_.begin() + 1, sizes_.end(),
                                              (int64_t) 1, std::multiplies<int64_t>());
  int64_t first_dim_size = sizes_[0];
  int64_t counter = 0;
  for (int64_t i = 0; i < first_dim_size; ++i)  {
    for (int64_t j = 0; j < elems_per_first_dim; ++j) {
      ptr[counter++] *= s_ptr[i];
    }
  }
  SET_RAW_DATA(ptr)
}

#define CALL_BIN_FUNC(type, vec, f)                                            \
  DATA(this, type, vec)                                                        \
  CONST_DATA(a, type, vec)                                                     \
  bin_func(f<type>(), this_data_ptr, a_const_data_ptr);                        \



#define APPLY_BINARY_FUNCTION(op_name, f)                                      \
  inline void Tensor::op_name(const Tensor& a_tensor) {                        \
    const Tensor* a = &a_tensor;                                               \
    if (a->elem_type() != elem_type_) {                                        \
      throw TensorError(std::string("Tensor types do not match.\nType ") +     \
      ONNX_NAMESPACE::to_string(elem_type_) + std::string(" != Type ") +       \
      ONNX_NAMESPACE::to_string(a->elem_type()));                              \
    }                                                                          \
    if (a->sizes() != sizes_) {                                                \
      throw TensorError(std::string("Tensor sizes do not match."));            \
    }                                                                          \
    switch(elem_type_) {                                                       \
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:  {                      \
        CALL_BIN_FUNC(float, floats, f)                                        \
        break;                                                                 \
      }                                                                        \
      case ONNX_NAMESPACE::TensorProto_DataType_BOOL:                          \
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:                          \
      case ONNX_NAMESPACE::TensorProto_DataType_INT16:                         \
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:                         \
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:                         \
      case ONNX_NAMESPACE::TensorProto_DataType_UINT16:  {                     \
        CALL_BIN_FUNC(int32_t, int32s, f)                                      \
        break;                                                                 \
      }                                                                        \
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:  {                      \
        CALL_BIN_FUNC(int64_t, int64s, f)                                      \
        break;                                                                 \
      }                                                                        \
      case ONNX_NAMESPACE::TensorProto_DataType_UINT32:                        \
      case ONNX_NAMESPACE::TensorProto_DataType_UINT64:  {                     \
        CALL_BIN_FUNC(uint64_t, uint64s, f)                                    \
        break;                                                                 \
      }                                                                        \
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {                      \
        CALL_BIN_FUNC(double, doubles, f)                                      \
        break;                                                                 \
      }                                                                        \
      default:                                                                 \
        throw TensorError(std::string("Operation ") + std::string(#op_name) +  \
        std::string(" not supported for data type ") +                         \
        ONNX_NAMESPACE::to_string(elem_type_));                                \
    }                                                                          \
  }                                                                            \

APPLY_BINARY_FUNCTION(add, std::plus)
APPLY_BINARY_FUNCTION(subtract, std::minus)
APPLY_BINARY_FUNCTION(multiply, std::multiplies)
APPLY_BINARY_FUNCTION(divide, std::divides)

#undef CALL_BIN_FUNC
#undef APPLY_BINARY_FUNCTION

inline void Tensor::sqrt() {
  switch(elem_type_) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:  {
      DATA(this, float, floats)
      un_func<float (*)(float), float>(std::sqrt, this_data_ptr);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
      DATA(this, double, doubles)
      un_func<double (*)(double), double>(std::sqrt, this_data_ptr);
      break;
    }
    default:
      throw TensorError(std::string("Operation sqrt not supported for data type ") +
      ONNX_NAMESPACE::to_string(elem_type_));
    }
}

inline void Tensor::scale_by_first_dim(const Tensor& s_tensor) {
  const Tensor* s = &s_tensor;
  ONNX_ASSERT(sizes_.size() > 1 && s->sizes().size() == 1 && s->sizes()[0] == sizes_[0]);
  ONNX_ASSERT(s->elem_type() == elem_type_);

  switch(elem_type_) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:  {
      DATA(this, float, floats)
      CONST_DATA(s, float, floats)
      scale_dim<float>(this_data_ptr, s_const_data_ptr);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
      DATA(this, int32_t, int32s)
      CONST_DATA(s, int32_t, int32s)
      scale_dim<int32_t>(this_data_ptr, s_const_data_ptr);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
      DATA(this, double, doubles)
      CONST_DATA(s, double, doubles)
      scale_dim<double>(this_data_ptr, s_const_data_ptr);
      break;
    }
    default:
      throw TensorError(std::string("Operation scale_by_first_dim not supported for data type ") +
      ONNX_NAMESPACE::to_string(elem_type_));
    }
}
#undef CONST_DATA
#undef DATA
#undef SET_RAW_DATA

} // namespace ONNX_NAMESPACE
