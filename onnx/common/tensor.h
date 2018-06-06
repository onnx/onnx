// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {

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

  //Element wise scaling of tensor by scale_by_channel
  //s is one dimensional, has size C, where C is number of channels
  //s must have data as raw_data and has data type corresponding to this
  void scale_by_channel(Tensor* s) {
    ONNX_ASSERT(sizes_.size() > 2 && s.sizes().size() == 1 && s.sizes()[0] == sizes_[1]);
    ONNX_ASSERT(s.is_raw_data() && s.elem_type() == this.elem_type_);
    int64_t dim_per_data = 1;
    for (int i = 2; i < sizes_.size(); i++) {
      dim_per_data *= sizes_[i];
    }
    int64_t feature_maps = sizes_[0];
    int64_t num_channels = sizes_[1];
    switch(this.elem_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {
        float* float_ptr;
        float* channel_scales = s.raw();
        if (this.is_raw_data_)  {
          float_ptr = (float*) raw_data_;
        } else {
          float_ptr = (float*) &this.float_data_[0];
        }
        int counter = 0;
        for (int i = 0; i < feature_maps; i++)  {
          for(int j = 0; j < num_channels; j++) {
            for (int k = 0; k < dim_per_data; k++)  {
              float_ptr[counter++] *= channel_scales[j];
            }
          }
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        int32_t* int32_ptr;
        int32_t* channel_scales = s.raw();
        if (this.is_raw_data_)  {
          int32_ptr = (int32_t*) raw_data_;
        } else {
          int32_ptr = (int32_t*) &this.int32_data_[0];
        }
        int counter = 0;
        for (int i = 0; i < feature_maps; i++)  {
          for(int j = 0; j < num_channels; j++) {
            for (int k = 0; k < dim_per_data; k++)  {
              int32_ptr[counter++] *= channel_scales[j];
            }
          }
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128: {
        double* double_ptr;
        double* channel_scales = s.raw();
        if (this.is_raw_data_)  {
          double_ptr = (double*) raw_data_;
        } else {
          double_ptr = (double*) &this.double_data_[0];
        }
        int counter = 0;
        for (int i = 0; i < feature_maps; i++)  {
          for(int j = 0; j < num_channels; j++) {
            for (int k = 0; k < dim_per_data; k++)  {
              double_ptr[counter++] *= channel_scales[j];
            }
          }
        }
        break;
      }
      default:
        throw("Incompatible data type: FLOAT, COMPLEX64, and FLOAT16 supported");
    }
  }



  //applies function f element-wise to this and a, storing result in this
  //WARNING: does not type check, so ensure that the tensors have the correct
  //types before using
  void apply_binary_function(void (*f)(void*, void*), Tensor a)  {
    if (a.elem_type() != this.elem_type_) {
      throw("Type of tensors do not match");
    }
    if (a.sizes() != this.sizes_) {
      throw("Tensor shapes are incompatible");
    }

    switch(this.elem_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {
        float* float_ptr;
        float* a_ptr;
        if (this.is_raw_data_)  {
          float_ptr = (float*) raw_data_;
        } else {
          float_ptr = (float*) &this.float_data_[0];
        }
        if (a.is_raw_data())  {
          a_ptr = (float*) a.raw();
        } else {
          a_ptr = (float*) &a.floats()[0];
        }
        for (int i = 0; i < this.float_data_.size(); i++) {
          f((void*)(float_ptr + i), (void*)(a_ptr + i));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
        int32_t* int32_ptr;
        int32_t* a_ptr;
        if (this.is_raw_data_)  {
          int32_ptr = (int32_t*) raw_data_;
        } else {
          int32_ptr = (int32_t*) &this.int32_data_[0];
        }
        if (a.is_raw_data())  {
          a_ptr = (int32_t*) a.raw();
        } else {
          a_ptr = (int32_t*) &a.int32s()[0];
        }
        for (int i = 0; i < this.int32_data_.size(); i++) {
          f((void*)(int32_ptr + i), (void*)(a_ptr + i));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
        int64_t* int64_ptr;
        int64_t* a_ptr;
        if (this.is_raw_data_)  {
          int64_ptr = (int64_t*) raw_data_;
        } else {
          int64_ptr = (int64_t*) &this.int64_data_[0];
        }
        if (a.is_raw_data())  {
          a_ptr = (int64_t*) a.raw();
        } else {
          a_ptr = (int64_t*) &a.int64s()[0];
        }
        for (int i = 0; i < this.int64_data_.size(); i++) {
          f((void*)(int64_ptr + i), (void*)(a_ptr + i));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {
        uint64_t* uint64_ptr;
        uint64_t* a_ptr;
        if (this.is_raw_data_)  {
          uint64_ptr = (uint64_t*) raw_data_;
        } else {
          uint64_ptr = (uint64_t*) &this.uint64_data_[0];
        }
        if (a.is_raw_data())  {
          a_ptr = (uint64_t*) a.raw();
        } else {
          a_ptr = (uint64_t*) &a.uint64s()[0];
        }
        for (int i = 0; i < this.uint64_data_.size(); i++) {
          f((void*)(uint64_ptr + i), (void*)(a_ptr + i));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128: {
        double* double_ptr;
        double* a_ptr;
        if (this.is_raw_data_)  {
          double_ptr = (double*) raw_data_;
        } else {
          double_ptr = (double*) &this.double_data_[0];
        }
        if (a.is_raw_data())  {
          a_ptr = (double*) a.raw();
        } else {
          a_ptr = (double*) &a.doubles()[0];
        }
        for (int i = 0; i < this.double_data_.size(); i++) {
          f((void*)(double_ptr + i), (void*)(a_ptr + i));
        }
        break;
      }
      default:
        throw("Operation not supported for this data type");
    }
  }

  //applies function f element-wise to this, storing result in this
  //WARNING: does not type check, so ensure that this tensor has the correct
  //type before using
  void apply_unary_function(void (*f)(void*))  {
    switch(this.elem_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {
        float* float_ptr;
        if (this.is_raw_data_)  {
          float_ptr = (float*) raw_data_;
        } else {
          float_ptr = (float*) &this.float_data_[0];
        }
        for (int i = 0; i < this.float_data_.size(); i++) {
          f((void*)(float_ptr + i));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
        int32_t* int32_ptr;
        if (this.is_raw_data_)  {
          int32_ptr = (int32_t*) raw_data_;
        } else {
          int32_ptr = (int32_t*) &this.int32_data_[0];
        }
        for (int i = 0; i < this.int32_data_.size(); i++) {
          f((void*)(int32_ptr + i));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
        int64_t* int64_ptr;
        if (this.is_raw_data_)  {
          int64_ptr = (int64_t*) raw_data_;
        } else {
          int64_ptr = (int64_t*) &this.int64_data_[0];
        }
        for (int i = 0; i < this.int64_data_.size(); i++) {
          f((void*)(int64_ptr + i));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {
        uint64_t* uint64_ptr;
        if (this.is_raw_data_)  {
          uint64_ptr = (uint64_t*) raw_data_;
        } else {
          uint64_ptr = (uint64_t*) &this.uint64_data_[0];
        }
        for (int i = 0; i < this.uint64_data_.size(); i++) {
          f((void*)(uint64_ptr + i));
        }
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128: {
        double* double_ptr;
        if (this.is_raw_data_)  {
          double_ptr = (double*) raw_data_;
        } else {
          double_ptr = (double*) &this.double_data_[0];
        }
        for (int i = 0; i < this.double_data_.size(); i++) {
          f((void*)(double_ptr + i));
        }
        break;
      }
      default:
        throw("Operation not supported for this data type");
    }
  }
};

} // namespace ONNX_NAMESPACE
