// SPDX-License-Identifier: Apache-2.0
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_classifier.cc.

#include "onnx/reference/c_ops/c_op_common.h"

namespace onnx_c_ops {

namespace py = pybind11;

template <typename T>
static void Im2colWithEqualPadding(
    int64_t output_h,
    int64_t output_w,
    const T* data_im,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t pad_t,
    int64_t pad_l,
    int64_t stride_h,
    int64_t stride_w,
    T* data_col,
    T padding_value) {
  // From Intel, https://github.com/BVLC/caffe/pull/3536
  int64_t pad_h = pad_t;
  int64_t pad_w = pad_l;
  int64_t channel_size = height * width;
  for (int64_t channel = channels; channel--; data_im += channel_size) {
    for (int64_t kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int64_t kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int64_t input_row = -pad_h + kernel_row * dilation_h;
        for (int64_t output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            std::fill_n(data_col, output_w, padding_value);
            data_col += output_w;
          } else {
            int64_t input_col = -pad_w + kernel_col * dilation_w;
            const T* rdptr = data_im + input_row * width + input_col;
            for (int64_t i = 0; i != output_w; ++i) {
              *data_col = is_a_ge_zero_and_a_lt_b(input_col, width) ? rdptr[i * stride_w] : padding_value;
              input_col += stride_w;
              ++data_col;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

template <typename T>
void Im2colNd_NCHW(
    const T* data_img,
    const int64_t* im_shape,
    const int64_t* col_shape,
    int64_t /*img_size*/,
    int64_t /*col_size*/,
    const int64_t* kernel_shape,
    const int64_t* stride,
    const int64_t* dilation,
    const int64_t* pad,
    int64_t N,
    T* data_col,
    bool accumulate_output = false,
    T padding_value = 0) {
  int64_t kernel_size = 1;
  for (int64_t i = 0; i < N; ++i)
    kernel_size *= kernel_shape[i];

  int64_t channels_col = col_shape[0];
  std::vector<int64_t> d_offset(N, 0);
  std::vector<int64_t> d_iter(N, 0);

  for (int64_t c_col = 0; c_col < channels_col; ++c_col) {
    // Loop over spatial axes in reverse order to compute a per-axis offset.
    int64_t offset = c_col;
    for (int64_t d_i = N - 1; d_i >= 0; --d_i) {
      if (d_i < N - 1)
        offset /= kernel_shape[d_i + 1];
      d_offset[d_i] = offset % kernel_shape[d_i];
    }
    for (bool incremented = true; incremented;) {
      // Loop over spatial axes in forward order to compute the indices in the
      // image and column, and whether the index lies in the padding.
      int64_t index_col = c_col;
      int64_t index_im = c_col / kernel_size;
      bool is_padding = false;
      for (int64_t d_i = 0; d_i < N; ++d_i) {
        int64_t d = d_iter[d_i];
        int64_t d_im = d * stride[d_i] - pad[d_i] + d_offset[d_i] * dilation[d_i];
        is_padding |= d_im < 0 || d_im >= im_shape[d_i + 1];
        index_col *= col_shape[d_i + 1];
        index_col += d;
        index_im *= im_shape[d_i + 1];
        index_im += d_im;
      }
      if (!accumulate_output) {
        if (is_padding)
          data_col[index_col] = padding_value;
        else
          data_col[index_col] = data_img[index_im];
      } else if (!is_padding) // col2im
        data_col[index_im] += data_img[index_col];

      // Loop over spatial axes in reverse order to choose an index,
      // like counting.
      incremented = false;
      for (int64_t d_i = N - 1; d_i >= 0; --d_i) {
        int64_t d_max = col_shape[d_i + 1];
        // ORT_ENFORCE(d_iter[d_i] < d_max);
        if (d_iter[d_i] == d_max - 1)
          d_iter[d_i] = 0;
        else { // d_iter[d_i] < d_max - 1
          ++d_iter[d_i];
          incremented = true;
          break;
        }
      }
    } // while(incremented) {
  } // for (int c = 0; c < channels_col; ++c) {
}

template <typename T>
void Im2col_NCHW(
    const T* data_im,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t pad_t,
    int64_t pad_l,
    int64_t pad_b,
    int64_t pad_r,
    int64_t stride_h,
    int64_t stride_w,
    T* data_col,
    T padding_value = 0) {
  const int64_t output_h = (height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int64_t output_w = (width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  // Fast path for zero padding and no dilation
  // From Torch, THNN_(unfolded_copy)
  if (dilation_h == 1 && dilation_w == 1 && pad_l == 0 && pad_r == 0 && pad_t == 0 && pad_b == 0) {
    for (auto k = 0; k < channels * kernel_h * kernel_w; k++) {
      const auto nip = k / (kernel_h * kernel_w);
      const auto rest = k % (kernel_h * kernel_w);
      const auto kh = rest / kernel_w;
      const auto kw = rest % kernel_w;
      auto* dst = data_col + nip * (kernel_h * kernel_w * output_h * output_w) + kh * (kernel_w * output_h * output_w) +
          kw * (output_h * output_w);
      const auto* src = data_im + nip * (height * width);
      for (auto y = 0; y < output_h; y++) {
        const auto iy = y * stride_h + kh;
        const auto ix = kw;
        if (stride_w == 1) {
          memcpy(dst + (y * output_w), src + (iy * width + ix), sizeof(T) * output_w);
        } else {
          for (auto x = 0; x < output_w; x++) {
            memcpy(dst + (y * output_w + x), src + (iy * width + ix + x * stride_w), sizeof(T));
          }
        }
      }
    }
    return;
  }

  // Fast path for equal padding
  if (pad_l == pad_r && pad_t == pad_b) {
    Im2colWithEqualPadding(
        output_h,
        output_w,
        data_im,
        channels,
        height,
        width,
        kernel_h,
        kernel_w,
        dilation_h,
        dilation_w,
        pad_t,
        pad_l,
        stride_h,
        stride_w,
        data_col,
        padding_value);
    return;
  }

  // Baseline
  const int64_t dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int64_t dkernel_w = dilation_w * (kernel_w - 1) + 1;

  int64_t height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  int64_t width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;

  int64_t channels_col = channels * kernel_h * kernel_w;
  for (int64_t c = 0; c < channels_col; ++c) {
    int64_t w_offset = c % kernel_w;
    int64_t h_offset = (c / kernel_w) % kernel_h;
    int64_t c_im = c / kernel_h / kernel_w;
    for (int64_t h = 0; h < height_col; ++h) {
      for (int64_t w = 0; w < width_col; ++w) {
        int64_t h_pad = h * stride_h - pad_t + h_offset * dilation_h;
        int64_t w_pad = w * stride_w - pad_l + w_offset * dilation_w;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] = data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = padding_value;
      }
    }
  }
}

// Loop over spatial axes in reverse order to choose an index, like counting.
inline bool NextPosition(int64_t N, const int64_t* shape, int64_t* dims) {
  bool has_next_output = false;
  for (int64_t d_i = N - 1; d_i >= 0; --d_i) {
    int64_t d_max = shape[d_i];
    // assert dims[d_i] < d_max
    if (dims[d_i] == d_max - 1) {
      dims[d_i] = 0;
    } else { // dims[d_i] < d_max - 1
      ++dims[d_i];
      has_next_output = true;
      break;
    }
  }
  return has_next_output;
}

template <typename T>
void Im2col_NCHW(
    const T* data_im,
    int64_t group_channels,
    int64_t input_channels,
    const int64_t* im_shape,
    const int64_t* output_shape,
    const int64_t* kernel_shape,
    const int64_t* stride,
    const int64_t* dilation,
    const int64_t* pad,
    ptrdiff_t rank,
    T* data_col,
    T padding_value) {
  // iterate dimensions on output image shape (without Batch and Channel)
  std::vector<int64_t> d_output(rank, 0);
  // inner iterate dimensions on kernel shape (without output channel and input channel)
  std::vector<int64_t> d_kernel(rank, 0);

  // Loop over spatial axes along the output image shape
  do {
    // Loop over spatial axes in reverse order to choose an index on kernel dimensions
    do {
      // Loop over spatial axes in forward order to compute the indices in the image
      // and the inner col, and whether the index lies in the padding.
      int64_t index_im = 0;
      bool is_padding = false;
      for (ptrdiff_t d_i = 0; d_i < rank; ++d_i) {
        int64_t d_im = d_output[d_i] * stride[d_i] - pad[d_i] + d_kernel[d_i] * dilation[d_i];
        is_padding |= !is_a_ge_zero_and_a_lt_b(d_im, im_shape[d_i]);
        index_im *= im_shape[d_i];
        index_im += d_im;
      }
      index_im *= input_channels;

      if (is_padding) {
        data_col = std::fill_n(data_col, group_channels, padding_value);
      } else {
        data_col = std::copy_n(data_im + index_im, group_channels, data_col);
      }
    } while (NextPosition(rank, kernel_shape, d_kernel.data()));
  } while (NextPosition(rank, output_shape, d_output.data()));
}

template <typename T>
void Im2col_NHWC(
    const T* data_im,
    int64_t input_channels,
    const int64_t* input_shape,
    const int64_t* output_shape,
    const int64_t* kernel_shape,
    const int64_t* stride,
    const int64_t* dilation,
    const int64_t* pad,
    ptrdiff_t rank,
    int64_t output_start,
    int64_t output_count,
    T const** data_indirection,
    const T* padding_ptr) {
  if (rank == 1) {
    int64_t stride_w = stride[0];
    int64_t kernel_w = kernel_shape[0];
    int64_t dilation_w = dilation[0];
    int64_t pad_l = pad[0];
    int64_t input_w = input_shape[0];

    int64_t ow = output_start * stride_w;

    while (output_count--) {
      int64_t iw = ow - pad_l;
      for (int64_t kw = 0; kw < kernel_w; kw++) {
        const T* data_ptr = data_im + iw * input_channels;
        data_indirection[kw] = (is_a_ge_zero_and_a_lt_b(iw, input_w) ? data_ptr : padding_ptr);
        iw += dilation_w;
      }
      data_indirection += kernel_w;
      ow += stride_w;
    }
  } else if (rank == 2) {
    int64_t stride_h = stride[0];
    int64_t stride_w = stride[1];
    int64_t kernel_h = kernel_shape[0];
    int64_t kernel_w = kernel_shape[1];
    int64_t dilation_h = dilation[0];
    int64_t dilation_w = dilation[1];
    int64_t pad_t = pad[0];
    int64_t pad_l = pad[1];
    int64_t input_h = input_shape[0];
    int64_t input_w = input_shape[1];
    int64_t output_w = output_shape[1];

    int64_t oh = (output_start / output_w) * stride_h;
    int64_t ow = (output_start % output_w) * stride_w;
    int64_t ow_end = output_w * stride_w;

    while (output_count--) {
      for (int64_t kh = 0; kh < kernel_h; kh++) {
        int64_t ih = kh * dilation_h + oh - pad_t;
        if (is_a_ge_zero_and_a_lt_b(ih, input_h)) {
          int64_t ihw = ih * input_w;
          int64_t iw = ow - pad_l;
          for (int64_t kw = 0; kw < kernel_w; kw++) {
            const T* data_ptr = data_im + (ihw + iw) * input_channels;
            data_indirection[kw] = (is_a_ge_zero_and_a_lt_b(iw, input_w) ? data_ptr : padding_ptr);
            iw += dilation_w;
          }
        } else {
          std::fill_n(data_indirection, kernel_w, padding_ptr);
        }
        data_indirection += kernel_w;
      }
      ow += stride_w;
      if (ow == ow_end) {
        oh += stride_h;
        ow = 0;
      }
    }
  } else {
    // iterate dimensions on output image shape (without Batch and Channel)
    std::vector<int64_t> d_output(rank, 0);
    // inner iterate dimensions on kernel shape (without output channel and input channel)
    std::vector<int64_t> d_kernel(rank, 0);

    // Skip ahead to the starting output index.
    for (ptrdiff_t d_i = rank - 1; d_i >= 0; --d_i) {
      d_output[d_i] = output_start % output_shape[d_i];
      output_start /= output_shape[d_i];
    }

    while (output_count--) {
      // Loop over spatial axes in reverse order to choose an index on kernel dimensions
      do {
        // Loop over spatial axes in forward order to compute the indices in the image
        // and the inner col, and whether the index lies in the padding.
        int64_t index_im = 0;
        bool is_padding = false;
        for (ptrdiff_t d_i = 0; d_i < rank; ++d_i) {
          int64_t d_input = d_output[d_i] * stride[d_i] - pad[d_i] + d_kernel[d_i] * dilation[d_i];
          is_padding |= !is_a_ge_zero_and_a_lt_b(d_input, input_shape[d_i]);
          index_im *= input_shape[d_i];
          index_im += d_input;
        }
        const T* data_ptr = data_im + index_im * input_channels;
        *data_indirection++ = is_padding ? padding_ptr : data_ptr;
      } while (NextPosition(rank, kernel_shape, d_kernel.data()));
      // Loop over spatial axes along the output image shape
      NextPosition(rank, output_shape, d_output.data());
    }
  }
}

template <typename T>
void Im2col_NHWC(
    const T* data_im,
    int64_t group_channels,
    int64_t input_channels,
    int64_t input_h,
    int64_t input_w,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t pad_t,
    int64_t pad_l,
    int64_t stride_h,
    int64_t stride_w,
    int64_t output_w,
    int64_t output_start,
    int64_t output_count,
    T* data_col,
    T padding_value) {
  int64_t mh = output_start / output_w;
  int64_t mw = output_start % output_w;
  for (int64_t mz = output_start; mz < output_start + output_count; mz++) {
    int64_t oh = mh * stride_h;
    int64_t ow = mw * stride_w;

    for (int64_t kh = 0; kh < kernel_h; kh++) {
      int64_t ih = kh * dilation_h + oh - pad_t;

      if (is_a_ge_zero_and_a_lt_b(ih, input_h)) {
        int64_t iw = ow - pad_l;
        if (dilation_w == 1 && group_channels == input_channels) {
          int64_t kw = kernel_w;
          while (kw > 0) {
            if (is_a_ge_zero_and_a_lt_b(iw, input_w)) {
              // Increase the copy count size to reduce the number of copy calls.
              int64_t batch_w = std::min(kw, input_w - iw);
              std::memcpy(
                  data_col,
                  data_im + (ih * input_w + iw) * group_channels,
                  static_cast<size_t>(sizeof(T) * batch_w * group_channels));
              data_col += batch_w * group_channels;
              iw += batch_w;
              kw -= batch_w;
            } else {
              data_col = std::fill_n(data_col, group_channels, padding_value);
              iw++;
              kw--;
            }
          }
        } else {
          for (int64_t kw = 0; kw < kernel_w; kw++) {
            if (is_a_ge_zero_and_a_lt_b(iw, input_w)) {
              // N.B. Using std::memcpy helped here over std::copy_n when doing a
              // transform for an image with a small number of group channels.
              std::memcpy(
                  data_col,
                  data_im + (ih * input_w + iw) * input_channels,
                  static_cast<size_t>(sizeof(T) * group_channels));
              data_col += group_channels;
            } else {
              data_col = std::fill_n(data_col, group_channels, padding_value);
            }
            iw += dilation_w;
          }
        }
      } else {
        data_col = std::fill_n(data_col, kernel_w * group_channels, padding_value);
      }
    }

    if (++mw == output_w) {
      ++mh;
      mw = 0;
    }
  }
}

void ComputePadAndOutputShape(
    int64_t in_dim,
    int64_t stride,
    int64_t kernel,
    int64_t dilation,
    AutoPadType pad_type,
    int64_t* pad_head,
    int64_t* pad_tail,
    int64_t* out_dim,
    bool ForceSymmetricAutoPadding) {
  const int64_t dkernel = dilation * (kernel - 1) + 1;

  if (pad_type == AutoPadType::NOTSET) {
    *out_dim = static_cast<int64_t>(static_cast<float>(in_dim + *pad_head + *pad_tail - dkernel) / stride + 1);
  } else {
    switch (pad_type) {
      case AutoPadType::VALID:
        *pad_head = 0;
        *pad_tail = 0;
        *out_dim = (in_dim - dkernel) / stride + 1;
        break;
      case AutoPadType::SAME_UPPER:
      case AutoPadType::SAME_LOWER: {
        if (dilation != 1)
          throw std::invalid_argument("Dilation not supported for AutoPadType::SAME_UPPER or AutoPadType::SAME_LOWER.");
        int64_t legacy_target_size = (in_dim + stride - 1) / stride;
        int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_dim;
        *out_dim = (in_dim + pad_needed - dkernel) / stride + 1;

        // make sure padding is symmetric
        if (ForceSymmetricAutoPadding)
          pad_needed = roundUpPow2<int64_t, 2>(pad_needed);

        *pad_head = (pad_type == AutoPadType::SAME_LOWER) ? (pad_needed + 1) / 2 : pad_needed / 2;
        *pad_tail = pad_needed - *pad_head;
      } break;
      default:
        throw std::invalid_argument("Invalid argument in ComputePadAndOutputShape.");
    }
  }
}

void conv_infer_output_shape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& kernel_shape,
    const std::vector<int64_t>& strides_p,
    const std::vector<int64_t>& dilations_p,
    std::vector<int64_t>& pads_p,
    std::vector<int64_t>& output_shape,
    bool ForceSymmetricAutoPadding,
    AutoPadType auto_pad) {
  size_t rank = input_shape.size();
  int64_t dim_size;

  for (size_t dim = 0; dim < rank; ++dim) {
    if (dim >= strides_p.size() || dim >= kernel_shape.size() || dim >= dilations_p.size() || dim >= pads_p.size() ||
        rank + dim >= pads_p.size())
      throw std::invalid_argument(MakeString(
          "Failure in infer_output_shape, one of these conditions should be True:",
          "dim >= strides.size(), dim >= kernel_shape.size(), ",
          "dim >= dilations.size(), dim >= padding.size(), dim=",
          dim,
          ", strides.size()=",
          strides_p.size(),
          ", kernel_shape.size()=",
          kernel_shape.size(),
          ", dilations.size()=",
          dilations_p.size(),
          ", padding.size()=",
          pads_p.size(),
          "."));

    dim_size = 0;
    ComputePadAndOutputShape(
        input_shape[dim],
        strides_p[dim],
        kernel_shape[dim],
        dilations_p[dim],
        auto_pad,
        &pads_p.at(dim),
        &pads_p.at(input_shape.size() + dim),
        &dim_size,
        ForceSymmetricAutoPadding);
    if (dim_size <= 0)
      throw std::invalid_argument(MakeString(
          "Invalid argument in infer_output_shape, ComputePadAndOutputShape returned dim_size=", dim_size, "."));
    output_shape.push_back(dim_size);
  }
}

template <typename T>
void ComputeTransposePadAndOutputShape(
    int64_t in_size,
    int64_t stride,
    int64_t kernel,
    int64_t dilation,
    int64_t adj,
    AutoPadType pad_type,
    int64_t* pad_head,
    int64_t* pad_tail,
    int64_t* out_size) {
  if (*out_size != -1) {
    // total padding size
    int64_t paddings = std::max<int64_t>(0, (in_size - 1) * stride + adj + (kernel - 1) * dilation + 1 - *out_size);
    if (pad_type == AutoPadType::SAME_UPPER) { // pad more on head when paddings are odd.
      *pad_head = paddings - paddings / 2;
      *pad_tail = paddings / 2;
    } else {
      // for pad_type is NOTSET, SAME_LOWER or VALID
      // set pad_head as paddings/2, pad_tail as paddings-paddings/2.
      // That said, we pad more on tail when paddings are odd.
      *pad_head = paddings / 2;
      *pad_tail = paddings - paddings / 2;
    }
    return;
  }
  if (pad_type != AutoPadType::NOTSET) {
    switch (pad_type) {
        // We handle cases of AutoPadType::VALID and AutoPadType::SAME_UPPER/LOWER,
        // the same way
      case AutoPadType::VALID:
      case AutoPadType::SAME_UPPER:
      case AutoPadType::SAME_LOWER:
        *pad_head = 0;
        *pad_tail = 0;
        *out_size = (in_size - 1) * stride + adj + (kernel - 1) * dilation + 1;
        break;
      default:
        throw std::invalid_argument("pad type not supported");
    }
  } else {
    *out_size = (in_size - 1) * stride + adj + (kernel - 1) * dilation + 1 - *pad_head - *pad_tail;
  }
}

class ConvPoolCommonShape {
 protected:
  AutoPadType auto_pad_;
  std::vector<int64_t> kernel_shape_;

 public:
  ConvPoolCommonShape() {
    auto_pad_ = AutoPadType::NOTSET;
  }

  void init(const std::string& auto_pad, py_array_t<int64_t> kernel_shape);
  void initcpp(const std::string& auto_pad, std::vector<int64_t> kernel_shape);
  void compute_kernel_shape(const std::vector<int64_t>& weight_shape, std::vector<int64_t>& kernel_shape) const;

  void infer_output_shape(
      const std::vector<int64_t>& input_shape,
      const std::vector<int64_t>& kernel_shape,
      const std::vector<int64_t>& strides_p,
      const std::vector<int64_t>& dilations_p,
      std::vector<int64_t>& pads_p,
      std::vector<int64_t>& output_shape,
      bool ForceSymmetricAutoPadding) const;
};

class ConvPoolCommon : public ConvPoolCommonShape {
 protected:
  std::vector<int64_t> dilations_;
  int64_t group_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> strides_;

 public:
  void init(
      const std::string& auto_pad,
      py_array_t<int64_t> dilations,
      int64_t group,
      py_array_t<int64_t> kernel_shape,
      py_array_t<int64_t> pads,
      py_array_t<int64_t> strides);

  void initcpp(
      const std::string& auto_pad,
      std::vector<int64_t> dilations,
      int64_t group,
      std::vector<int64_t> kernel_shape,
      std::vector<int64_t> pads,
      std::vector<int64_t> strides);
};

void ConvPoolCommonShape::init(const std::string& auto_pad, py_array_t<int64_t> kernel_shape) {
  auto_pad_ = to_AutoPadType(auto_pad);
  array2vector(kernel_shape_, kernel_shape, int64_t);
}

void ConvPoolCommonShape::initcpp(const std::string& auto_pad, std::vector<int64_t> kernel_shape) {
  auto_pad_ = to_AutoPadType(auto_pad);
  kernel_shape_ = kernel_shape;
}

void ConvPoolCommonShape::compute_kernel_shape(
    const std::vector<int64_t>& weight_shape,
    std::vector<int64_t>& kernel_shape) const {
  if (kernel_shape_.size() > 0) {
    kernel_shape = kernel_shape_;
    if (kernel_shape.size() + 2 != weight_shape.size())
      throw std::invalid_argument("kernel_shape num_dims is not compatible with W num_dims (1).");

    for (size_t i = 0; i < kernel_shape.size(); ++i)
      if (kernel_shape[i] != weight_shape[i + 2])
        throw std::invalid_argument("kernel_shape num_dims is not compatible with W num_dims (2).");
  } else {
    auto& weight_dims = weight_shape;
    kernel_shape = std::vector<int64_t>(weight_dims.begin() + 2, weight_dims.end());
  }
}

void ConvPoolCommonShape::infer_output_shape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& kernel_shape,
    const std::vector<int64_t>& strides_p,
    const std::vector<int64_t>& dilations_p,
    std::vector<int64_t>& pads_p,
    std::vector<int64_t>& output_shape,
    bool ForceSymmetricAutoPadding) const {
  conv_infer_output_shape(
      input_shape, kernel_shape, strides_p, dilations_p, pads_p, output_shape, ForceSymmetricAutoPadding, auto_pad_);
}

void ConvPoolCommon::init(
    const std::string& auto_pad,
    py_array_t<int64_t> dilations,
    int64_t group,
    py_array_t<int64_t> kernel_shape,
    py_array_t<int64_t> pads,
    py_array_t<int64_t> strides) {
  ConvPoolCommonShape::init(auto_pad, kernel_shape);
  array2vector(dilations_, dilations, int64_t);
  group_ = group;
  array2vector(pads_, pads, int64_t);
  array2vector(strides_, strides, int64_t);
}

void ConvPoolCommon::initcpp(
    const std::string& auto_pad,
    std::vector<int64_t> dilations,
    int64_t group,
    std::vector<int64_t> kernel_shape,
    std::vector<int64_t> pads,
    std::vector<int64_t> strides) {
  ConvPoolCommonShape::initcpp(auto_pad, kernel_shape);
  dilations_ = dilations;
  group_ = group;
  pads_ = pads;
  strides_ = strides;
}

template <typename T>
class Conv : public ConvPoolCommon {
 public:
  Conv();

  py::array_t<T> compute(
      py::array_t<T, py::array::c_style | py::array::forcecast> X,
      py::array_t<T, py::array::c_style | py::array::forcecast> W,
      py::array_t<T, py::array::c_style | py::array::forcecast> B) const;

 protected:
  void compute_gil_free(
      py::array_t<T, py::array::c_style | py::array::forcecast> X,
      py::array_t<T, py::array::c_style | py::array::forcecast> W,
      py::array_t<T, py::array::c_style | py::array::forcecast> B,
      py::array_t<T, py::array::c_style | py::array::forcecast>& Y,
      const std::vector<int64_t>& input_shape,
      const std::vector<int64_t>& output_shape,
      const std::vector<int64_t>& kernel_shape,
      const std::vector<int64_t>& pads,
      const std::vector<int64_t>& dilations,
      const std::vector<int64_t>& strides,
      const std::vector<int64_t>& x_dims,
      const std::vector<int64_t>& y_dims,
      const std::vector<int64_t>& w_dims) const;
};

template <typename T>
Conv<T>::Conv() : ConvPoolCommon() {}

template <typename T>
py::array_t<T> Conv<T>::compute(
    py::array_t<T, py::array::c_style | py::array::forcecast> X,
    py::array_t<T, py::array::c_style | py::array::forcecast> W,
    py::array_t<T, py::array::c_style | py::array::forcecast> B) const {
  std::vector<int64_t> x_dims;
  arrayshape2vector(x_dims, X);
  std::vector<int64_t> w_dims;
  arrayshape2vector(w_dims, W);

  const int64_t N = x_dims[0];
  const int64_t M = w_dims[0];

  std::vector<int64_t> kernel_shape;
  compute_kernel_shape(w_dims, kernel_shape);

  std::vector<int64_t> pads(pads_);
  if (pads.empty())
    pads.resize(kernel_shape.size() * 2, 0);

  std::vector<int64_t> dilations(dilations_);
  if (dilations.empty())
    dilations.resize(kernel_shape.size(), 1);

  std::vector<int64_t> strides(strides_);
  if (strides.empty())
    strides.resize(kernel_shape.size(), 1);

  std::vector<int64_t> y_dims;
  y_dims.insert(y_dims.begin(), {N, M});
  std::vector<int64_t> input_shape(x_dims.begin() + 2, x_dims.end());
  infer_output_shape(input_shape, kernel_shape, strides, dilations, pads, y_dims, false);
  std::vector<int64_t> output_shape(y_dims.begin() + 2, y_dims.end());

  py::array_t<T, py::array::c_style | py::array::forcecast> Y(y_dims);
  {
    py::gil_scoped_release release;
    compute_gil_free(
        X, W, B, Y, input_shape, output_shape, kernel_shape, pads, dilations, strides, x_dims, y_dims, w_dims);
  }
  return Y;
}

template <typename T>
void Conv<T>::compute_gil_free(
    py::array_t<T, py::array::c_style | py::array::forcecast> X,
    py::array_t<T, py::array::c_style | py::array::forcecast> W,
    py::array_t<T, py::array::c_style | py::array::forcecast> B,
    py::array_t<T, py::array::c_style | py::array::forcecast>& Y,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& output_shape,
    const std::vector<int64_t>& kernel_shape,
    const std::vector<int64_t>& pads,
    const std::vector<int64_t>& dilations,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& x_dims,
    const std::vector<int64_t>& y_dims,
    const std::vector<int64_t>& w_dims) const {
  std::vector<int64_t> b_dims;
  arrayshape2vector(b_dims, B);

  const int64_t N = x_dims[0];
  const int64_t C = x_dims[1];
  const int64_t M = w_dims[0];

  const int64_t input_image_size = flattened_dimension(input_shape);
  const int64_t output_image_size = flattened_dimension(output_shape);
  const int64_t y_size = flattened_dimension(y_dims);
  const int64_t kernel_size = flattened_dimension(kernel_shape);
  const int64_t X_offset = C / group_ * input_image_size;
  const int64_t Y_offset = flattened_dimension(y_dims) / y_dims[0] / group_;
  const int64_t W_offset = flattened_dimension(w_dims) / group_;
  const int64_t kernel_dim = C / group_ * kernel_size;
  const int64_t col_buffer_size = kernel_dim * output_image_size;

  std::vector<T> _col_data(col_buffer_size);
  auto col_buffer_data = &_col_data[0];

  const T* Xdata = X.data(0);
  T* Ydata = (T*)Y.data(0);
  T* yptr;
  size_t k2;

  std::fill(Ydata, Ydata + y_size, (T)0);

  std::vector<int64_t> image_shape(x_dims.begin() + 1, x_dims.end());
  std::vector<int64_t> col_buffer_shape{kernel_dim};
  col_buffer_shape.insert(col_buffer_shape.end(), output_shape.begin(), output_shape.end());

  const size_t kernel_rank = kernel_shape.size();

  for (int image_id = 0; image_id < N; ++image_id) {
    for (int group_id = 0; group_id < group_; ++group_id) {
      if (kernel_rank == 2) {
        Im2col_NCHW<T>(
            Xdata + group_id * X_offset,
            C / group_,
            input_shape[0],
            input_shape[1],
            kernel_shape[0],
            kernel_shape[1],
            dilations[0],
            dilations[1],
            pads[0],
            pads[1],
            pads[2],
            pads[3],
            strides[0],
            strides[1],
            col_buffer_data);
      } else {
        Im2colNd_NCHW<T>(
            Xdata + group_id * X_offset,
            &image_shape[0],
            col_buffer_shape.data(),
            C * input_image_size,
            col_buffer_size,
            &kernel_shape[0],
            strides.data(),
            &dilations[0],
            &pads[0],
            static_cast<int>(kernel_shape.size()),
            col_buffer_data);
      }

      gemm<T>(
          false,
          false,
          (size_t)(M / group_), // m
          (size_t)(output_image_size), // n
          (size_t)kernel_dim, // k
          (T)1, // alpha
          (const T*)W.data(0) + group_id * W_offset, // *a
          (const T*)col_buffer_data, // *b
          (T)0, // beta
          (T*)Ydata + group_id * Y_offset // *c
      );
    }

    if (b_dims.size() != 0 && b_dims[0] != 0) {
      const T* ptrb = B.data(0);
      for (size_t k = 0; k < (size_t)M; ++k, ++ptrb) {
        yptr = Ydata + output_image_size * k;
        for (k2 = 0; k2 < (size_t)output_image_size; ++k2, ++yptr)
          *yptr += *ptrb;
      }
    }

    Xdata += X_offset * group_;
    Ydata += Y_offset * group_;
  }
}

class ConvFloat : public Conv<float> {
 public:
  ConvFloat() : Conv<float>() {}
};

class ConvDouble : public Conv<double> {
 public:
  ConvDouble() : Conv<double>() {}
};

}; // namespace onnx_c_ops
