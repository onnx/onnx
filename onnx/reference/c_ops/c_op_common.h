#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace onnx_c_ops {

namespace py = pybind11;

#define py_array_t py::array_t
#define py_array_style py::array::c_style | py::array::forcecast

#define is_a_ge_zero_and_a_lt_b(a, b) (static_cast<uint64_t>(a) < static_cast<uint64_t>(b))

#define array2vector(vec, arr, dtype)     \
  {                                       \
    if (arr.size() > 0) {                 \
      auto n = arr.size();                \
      auto p = (dtype*)arr.data(0);       \
      vec = std::vector<dtype>(p, p + n); \
    }                                     \
  }

#define arrayshape2vector(vec, arr)           \
  {                                           \
    if (arr.size() > 0) {                     \
      vec.resize(arr.ndim());                 \
      for (size_t i = 0; i < vec.size(); ++i) \
        vec[i] = (int64_t)arr.shape(i);       \
    }                                         \
  }

template <typename T, T b>
constexpr T roundUpPow2(T a) {
  return (a + (b - 1)) & (~(b - 1));
}

template <class NTYPE>
NTYPE flattened_dimension(const std::vector<NTYPE>& values) {
  NTYPE r = 1;
  for (auto it = values.begin(); it != values.end(); ++it)
    r *= *it;
  return r;
}

template <class NTYPE>
NTYPE flattened_dimension(const std::vector<NTYPE>& values, int64_t first) {
  NTYPE r = 1;
  auto end = values.begin() + first;
  for (auto it = values.begin(); it != end; ++it)
    r *= *it;
  return r;
}

template <typename T>
inline void MakeStringInternal(std::ostringstream& ss, const T& t) noexcept {
  ss << t;
}

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<int32_t>& t) noexcept {
  for (auto it : t)
    ss << "x" << it;
}

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<uint32_t>& t) noexcept {
  for (auto it : t)
    ss << "x" << it;
}

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<int64_t>& t) noexcept {
  for (auto it : t)
    ss << "x" << it;
}

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<uint64_t>& t) noexcept {
  for (auto it : t)
    ss << "x" << it;
}

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<int16_t>& t) noexcept {
  for (auto it : t)
    ss << "x" << it;
}

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<uint16_t>& t) noexcept {
  for (auto it : t)
    ss << "x" << it;
}

template <typename T, typename... Args>
inline void MakeStringInternal(std::ostringstream& ss, const T& t, const Args&... args) noexcept {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

template <typename... Args>
inline std::string MakeString(const Args&... args) {
  std::ostringstream ss;
  MakeStringInternal(ss, args...);
  return std::string(ss.str());
}

enum StorageOrder {
  UNKNOWN = 0,
  NHWC = 1,
  NCHW = 2,
};

StorageOrder to_StorageOrder(const std::string& value);

enum class AutoPadType {
  NOTSET = 0,
  VALID = 1,
  SAME_UPPER = 2,
  SAME_LOWER = 3,
};

AutoPadType to_AutoPadType(const std::string& value);

StorageOrder to_StorageOrder(const std::string& input) {
  if (input == "UNKNOWN")
    return StorageOrder::UNKNOWN;
  if (input == "NHWC")
    return StorageOrder::NHWC;
  if (input == "NCHW")
    return StorageOrder::NCHW;
  throw std::invalid_argument(std::string("StorageOrder '") + input + std::string("' is not defined."));
}

AutoPadType to_AutoPadType(const std::string& input) {
  if (input == "NOTSET")
    return AutoPadType::NOTSET;
  if (input == "VALID")
    return AutoPadType::VALID;
  if (input == "SAME_UPPER")
    return AutoPadType::SAME_UPPER;
  if (input == "SAME_LOWER")
    return AutoPadType::SAME_LOWER;
  throw std::invalid_argument(std::string("AutoPadType '") + input + std::string("' is not defined."));
}

// The function adds value to C, assuming this array
// was initialized.
template <typename NTYPE>
void gemm(
    bool transA,
    bool transB,
    size_t M,
    size_t N,
    size_t K,
    NTYPE alpha,
    const NTYPE* A,
    const NTYPE* B,
    NTYPE beta,
    NTYPE* C) {
  if (transA) {
    if (transB) {
    } else {
      // a A B + b C, dimension = M * N
      NTYPE* begin;
      NTYPE val;
      NTYPE val0;
      size_t i, j, k, maxc = 0;
      const NTYPE *pA, *pB;
      for (i = 0, begin = C; i < M; ++i) {
        for (j = 0; j < N; ++j, ++begin) {
          val0 = *begin * beta;
          val = 0;
          pA = A + i;
          pB = B + j;
          for (k = K; k > 0; --k, pA += K, pB += N)
            val += *pA * *pB;
          *begin = val0 + val * alpha;
          maxc = maxc > (size_t)(begin - C) ? maxc : (size_t)(begin - C);
          if (maxc > M * N)
            throw std::invalid_argument("gemm10: maxc > M * N");
        }
      }
      return;
    }
  } else {
    if (transB) {
    } else {
      // a A B + b C, dimension = M * N
      NTYPE* begin;
      NTYPE val;
      NTYPE val0;
      size_t i, j, k, maxc = 0;
      const NTYPE *pA, *pB;
      for (i = 0, begin = C; i < M; ++i) {
        for (j = 0; j < N; ++j, ++begin) {
          val0 = *begin * beta;
          val = 0;
          pA = A + i * K;
          pB = B + j;
          for (k = K; k > 0; --k, ++pA, pB += N)
            val += *pA * *pB;
          *begin = val0 + val * alpha;
          maxc = maxc > (size_t)(begin - C) ? maxc : (size_t)(begin - C);
          if (maxc > M * N)
            throw std::invalid_argument("gemm00: maxc > M * N");
        }
      }
      return;
    }
  }
  throw std::invalid_argument("Not implemented for transposed matrices (Gemm<T>).");
}

}; // namespace onnx_c_ops
