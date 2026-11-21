// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/checker.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <filesystem> // NOLINT(build/c++17)
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "onnx/common/path.h"
#include "onnx/common/proto_util.h"
#include "onnx/common/safe_math.h"
#include "onnx/common/scoped_resource.h"
#include "onnx/defs/tensor_proto_util.h"
#include "onnx/shape_inference/implementation.h"

#ifdef _WIN32
#include <Windows.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
// path.h undefines the Windows SDK's OPTIONAL macro because it conflicts with
// ONNX enum members. winternl.h still uses the annotation internally, so make
// it available only while that SDK header is parsed.
#ifndef OPTIONAL
#define OPTIONAL
#define ONNX_DEFINED_WINDOWS_OPTIONAL
#endif
#include <winternl.h>
#ifdef ONNX_DEFINED_WINDOWS_OPTIONAL
#undef OPTIONAL
#undef ONNX_DEFINED_WINDOWS_OPTIONAL
#endif
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif // _WIN32

namespace ONNX_NAMESPACE::checker {

#define enforce_has_field(proto, field)                                                                          \
  do {                                                                                                           \
    if (!(proto).has_##field()) {                                                                                \
      fail_check_with_code(                                                                                      \
          error_code::kRequiredFieldMissing, "Field '", #field, "' of '", #proto, "' is required but missing."); \
    }                                                                                                            \
  } while (0)

#define enforce_non_empty_field(proto, field)                                                                      \
  do {                                                                                                             \
    if ((proto).field().empty()) {                                                                                 \
      fail_check_with_code(                                                                                        \
          error_code::kRequiredFieldEmpty, "Field '", #field, "' of '", #proto, "' is required to be non-empty."); \
    }                                                                                                              \
  } while (0)

// Returns the IR version that introduced the given tensor data type, or 0 for
// data types that have been available since the first IR version.
static int min_ir_version_for_data_type(int32_t data_type) {
  switch (data_type) {
    case TensorProto::BFLOAT16:
      return 0x00000004;
    case TensorProto::FLOAT8E4M3FN:
    case TensorProto::FLOAT8E4M3FNUZ:
    case TensorProto::FLOAT8E5M2:
    case TensorProto::FLOAT8E5M2FNUZ:
      return 0x00000009;
    case TensorProto::UINT4:
    case TensorProto::INT4:
      return 0x0000000A;
    case TensorProto::FLOAT4E2M1:
      return 0x0000000B;
    case TensorProto::FLOAT8E8M0:
      return 0x0000000C;
    case TensorProto::UINT2:
    case TensorProto::INT2:
      return 0x0000000D;
    case TensorProto::FLOAT6E2M3:
    case TensorProto::FLOAT6E3M2:
      return 0x0000000E;
    default:
      return 0;
  }
}

static void check_data_type_ir_version(int32_t data_type, std::string_view name, const CheckerContext& ctx) {
  const int min_ir_version = min_ir_version_for_data_type(data_type);
  // A non-positive IR version means the context does not carry one (e.g. a
  // default-constructed CheckerContext), so there is nothing to check against.
  if (ctx.get_ir_version() > 0 && ctx.get_ir_version() < min_ir_version) {
    fail_check_with_code(
        error_code::kDataTypeRequiresNewerIrVersion,
        "Data type ",
        Utils::DataTypeUtils::ToDataTypeString(data_type),
        " used by '",
        name,
        "' requires IR version >= ",
        min_ir_version,
        ", but the IR version is ",
        ctx.get_ir_version(),
        ".");
  }
}

static void check_type_ir_version(const TypeProto& type, std::string_view name, const CheckerContext& ctx) {
  switch (type.value_case()) {
    case TypeProto::kTensorType:
      check_data_type_ir_version(type.tensor_type().elem_type(), name, ctx);
      break;
    case TypeProto::kSparseTensorType:
      check_data_type_ir_version(type.sparse_tensor_type().elem_type(), name, ctx);
      break;
    case TypeProto::kSequenceType:
      check_type_ir_version(type.sequence_type().elem_type(), name, ctx);
      break;
    case TypeProto::kOptionalType:
      check_type_ir_version(type.optional_type().elem_type(), name, ctx);
      break;
    case TypeProto::kMapType:
      check_data_type_ir_version(type.map_type().key_type(), name, ctx);
      check_type_ir_version(type.map_type().value_type(), name, ctx);
      break;
    default:
      break;
  }
}

static void check_attribute_tensors_ir_version(const AttributeProto& attr, const CheckerContext& ctx) {
  if (attr.has_t()) {
    check_data_type_ir_version(attr.t().data_type(), attr.name(), ctx);
  }
  for (const auto& tensor : attr.tensors()) {
    check_data_type_ir_version(tensor.data_type(), attr.name(), ctx);
  }
  if (attr.has_sparse_tensor()) {
    check_data_type_ir_version(attr.sparse_tensor().values().data_type(), attr.name(), ctx);
  }
  for (const auto& sparse_tensor : attr.sparse_tensors()) {
    check_data_type_ir_version(sparse_tensor.values().data_type(), attr.name(), ctx);
  }
}

void check_value_info(const ValueInfoProto& value_info, const CheckerContext& ctx) {
  enforce_non_empty_field(value_info, name);
  check_type_ir_version(value_info.type(), value_info.name(), ctx);
  // Relax constraint for subgraph input/output.
  if (!ctx.is_main_graph())
    return;
  enforce_has_field(value_info, type);
  const auto value_case = value_info.type().value_case();
  switch (value_case) {
    case TypeProto::kTensorType: {
      const auto& type = value_info.type().tensor_type();
      enforce_has_field(type, elem_type);
      enforce_has_field(type, shape);
    } break;
    case TypeProto::kOptionalType: {
      const auto& type = value_info.type().optional_type();
      enforce_has_field(type, elem_type);
    } break;
    case TypeProto::kSequenceType: {
      const auto& type = value_info.type().sequence_type();
      enforce_has_field(type, elem_type);
    } break;
    case TypeProto::kMapType: {
      const auto& type = value_info.type().map_type();
      enforce_has_field(type, key_type);
      enforce_has_field(type, value_type);
    } break;
    case TypeProto::kOpaqueType: {
      const auto& type = value_info.type().opaque_type();
      enforce_non_empty_field(type, name);
    } break;
    case TypeProto::kSparseTensorType: {
      const auto& type = value_info.type().sparse_tensor_type();
      enforce_has_field(type, elem_type);
      enforce_has_field(type, shape);
    } break;

    default:
      fail_check("Unrecognized type value case (value_info name: ", value_info.name(), "): ", value_case);
  }
}

// Returns the number of bytes the tensor's data must have for its declared shape and
// type: either a packed sub-byte type (2, 4 or 4/3 elements per byte) or a regular,
// byte-aligned type. Returns 0 for types without a fixed size, such as STRING.
static int64_t expected_raw_data_bytes(const TensorProto& tensor, int64_t nelem) {
  const int64_t bit_width = ElementBitWidth(tensor.data_type());
  if (bit_width < 0) {
    return 0;
  }
  if (bit_width == 2 || bit_width == 4) {
    const int64_t elements_per_byte = 8 / bit_width;
    return (nelem / elements_per_byte) + (nelem % elements_per_byte != 0);
  }
  if (bit_width == 6) {
    int64_t expected_bits = 0;
    if (checked_mul_overflow(nelem, bit_width, &expected_bits)) {
      fail_check(
          "TensorProto (tensor name: ", tensor.name(), ") has a shape too large to validate against its data type.");
    }
    return (expected_bits / 8) + (expected_bits % 8 != 0);
  }
  int64_t expected_bytes = 0;
  if (checked_mul_overflow(nelem, bit_width / 8, &expected_bytes)) {
    fail_check(
        "TensorProto (tensor name: ", tensor.name(), ") has a shape too large to validate against its data type.");
  }
  return expected_bytes;
}

// Parses the value of an external_data "offset" or "length" entry, which must be a
// non-negative decimal integer.
static int64_t parse_external_data_integer(const StringStringEntryProto& entry, std::string_view tensor_name) {
  const std::string& value = entry.value();
  const char* end = value.data() + value.size();
  int64_t result = -1;
  const auto [ptr, ec] = std::from_chars(value.data(), end, result);
  if (ec != std::errc() || ptr != end || result < 0) {
    fail_check_with_code(
        error_code::kExternalDataEntryInvalid,
        "External data ",
        entry.key(),
        " of TensorProto (tensor name: ",
        tensor_name,
        ") must be a non-negative integer, but it is '",
        value,
        "'.");
  }
  return result;
}

// Checks offset and length against the size of the external data file, as the
// Python loader does before reading it, and that the data they describe is large
// enough for the tensor's declared shape and type.
static void check_external_data_size(
    const TensorProto& tensor,
    int64_t nelem,
    std::uintmax_t file_size,
    std::optional<int64_t> offset,
    std::optional<int64_t> length) {
  const auto start = static_cast<std::uintmax_t>(offset.value_or(0));
  if (start > file_size) {
    fail_check_with_code(
        error_code::kExternalDataOutOfRange,
        "External data offset (",
        start,
        ") exceeds file size (",
        file_size,
        ") for tensor ",
        tensor.name(),
        ".");
  }
  std::uintmax_t available = file_size - start;
  if (length.has_value()) {
    const auto requested = static_cast<std::uintmax_t>(*length);
    if (requested > available) {
      fail_check_with_code(
          error_code::kExternalDataOutOfRange,
          "External data length (",
          requested,
          ") exceeds available data (",
          available,
          " bytes from offset ",
          start,
          ") for tensor ",
          tensor.name(),
          ".");
    }
    available = requested;
  }
  const int64_t expected_bytes = expected_raw_data_bytes(tensor, nelem);
  if (expected_bytes > 0 && available < static_cast<std::uintmax_t>(expected_bytes)) {
    fail_check_with_code(
        error_code::kExternalDataOutOfRange,
        "External data of TensorProto (tensor name: ",
        tensor.name(),
        ") is ",
        available,
        " bytes, which is too small for the declared shape and type (",
        expected_bytes,
        " bytes required).");
  }
}

void check_tensor(const TensorProto& tensor, const CheckerContext& ctx) {
  enforce_has_field(tensor, data_type);
  if (!TensorProto_DataType_IsValid(tensor.data_type())) {
    fail_check("Unrecognized data_type (tensor name: ", tensor.name(), "): ", tensor.data_type());
  }
  if (tensor.data_type() == TensorProto::UNDEFINED) {
    fail_check("setting data_type field (tensor name: ", tensor.name(), ") to UNDEFINED is not allowed");
  }
  check_data_type_ir_version(tensor.data_type(), tensor.name(), ctx);

  int num_value_fields = 0;

  const char* value_field = nullptr;

#define check_data_field(field)               \
  bool has_##field = !tensor.field().empty(); \
  if (has_##field) {                          \
    ++num_value_fields;                       \
    value_field = #field;                     \
  }

  check_data_field(float_data);
  check_data_field(int32_data);
  check_data_field(string_data);
  check_data_field(int64_data);
  check_data_field(raw_data);
  check_data_field(double_data);
  check_data_field(uint64_data);

#undef check_data_field

  const int64_t full_nelem =
      safe_dim_product(tensor.dims(), [&](const char* msg) { fail_check(msg, " (tensor name: ", tensor.name(), ")"); });
  // Dims describe the full tensor, while segment describes the stored chunk for every payload type.
  int64_t nelem = full_nelem;
  if (tensor.has_segment()) {
    const auto& seg = tensor.segment();
    if (!seg.has_begin() || !seg.has_end()) {
      fail_check("TensorProto segment must set begin and end (tensor name: ", tensor.name(), ")");
    }
    const int64_t begin = seg.begin();
    const int64_t end = seg.end();
    if (begin < 0 || end < begin || end > full_nelem) {
      fail_check(
          "Invalid TensorProto segment (tensor name: ",
          tensor.name(),
          "): [",
          begin,
          ", ",
          end,
          ") out of [0, ",
          full_nelem,
          ")");
    }
    nelem = end - begin;
  }
  bool stored_externally = tensor.has_data_location() && tensor.data_location() == TensorProto::EXTERNAL;
  if (stored_externally) {
    if (num_value_fields != 0) {
      fail_check_with_code(
          error_code::kExternalDataEntryInvalid,
          "Data of TensorProto ( tensor name: ",
          tensor.name(),
          ") is stored externally and should not have data field.",
          value_field);
    }
    // External data holds the bytes raw_data would hold, and raw_data cannot
    // represent strings.
    if (tensor.data_type() == TensorProto::STRING) {
      fail_check("STRING data (tensor name: ", tensor.name(), ") should not be stored externally");
    }

    // Every location is validated, but, as in the Python loader, the last value
    // of each key is the one that describes the data.
    bool has_location = false;
    std::optional<std::uintmax_t> external_data_file_size;
    bool location_is_in_memory = false;
    const StringStringEntryProto* offset_entry = nullptr;
    const StringStringEntryProto* length_entry = nullptr;
    for (const StringStringEntryProto& entry : tensor.external_data()) {
      if (!entry.has_key()) {
        continue;
      }
      if (entry.key() == "location") {
        has_location = true;
        // An in-memory model has no model directory, so an external data
        // location must not be resolved against the current working
        // directory. (A '#' location refers to in-memory data, not a
        // file, and is unaffected.)
        // Classify the location after the same lexical normalization
        // resolve_external_data_location applies: '#unused/../data.bin'
        // normalizes to 'data.bin' and refers to a file, so only a
        // location that still starts with '#' once normalized is
        // in-memory data.
        const auto normalized_location = path_to_utf8(utf8_to_path(entry.value()).lexically_normal().make_preferred());
        location_is_in_memory = !normalized_location.empty() && normalized_location[0] == '#';
        if (!ctx.model_dir_from_path() && !normalized_location.empty() && normalized_location[0] != '#') {
          fail_check(
              "Data of TensorProto ( tensor name: ",
              tensor.name(),
              ") is stored externally, but the model was not checked from a file path, so its external data location '",
              entry.value(),
              "' cannot be resolved. Pass a model path to check_model to check external data.");
        }
        if (!location_is_in_memory) {
          const auto& external_data_context = ctx.get_external_data_context();
          if (external_data_context == nullptr) {
            fail_check("External data context is unavailable for tensor ", tensor.name(), ".");
          }
          // Validate every location entry, while retaining the last value as
          // the one that describes the tensor, matching the Python loader.
          external_data_file_size = external_data_context->file_size(entry.value(), tensor.name());
        } else {
          external_data_file_size.reset();
        }
      } else if (entry.key() == "offset") {
        offset_entry = &entry;
      } else if (entry.key() == "length") {
        length_entry = &entry;
      }
    }
    if (!has_location) {
      fail_check_with_code(
          error_code::kExternalDataEntryInvalid,
          "TensorProto ( tensor name: ",
          tensor.name(),
          ") is stored externally but doesn't have a location.");
    }
    std::optional<int64_t> offset;
    if (offset_entry != nullptr) {
      offset = parse_external_data_integer(*offset_entry, tensor.name());
    }
    std::optional<int64_t> length;
    if (length_entry != nullptr) {
      length = parse_external_data_integer(*length_entry, tensor.name());
    }
    if (!location_is_in_memory) {
      check_external_data_size(tensor, nelem, *external_data_file_size, offset, length);
    }
    return;
  }
  if (nelem == 0 && num_value_fields != 0) {
    fail_check("TensorProto (tensor name: ", tensor.name(), ") is 0-element but contains data!");
  }
  if (nelem != 0 && num_value_fields != 1) {
    fail_check("TensorProto (tensor name: ", tensor.name(), ") should contain one and only one value field.");
  }
  if (has_raw_data) {
    if (tensor.data_type() == TensorProto::STRING) {
      fail_check("STRING data (tensor name: ", tensor.name(), ") should not be stored in raw_data field");
    }
    const int64_t expected_bytes = expected_raw_data_bytes(tensor, nelem);
    if (expected_bytes > 0 && static_cast<int64_t>(tensor.raw_data().size()) < expected_bytes) {
      fail_check(
          "TensorProto (tensor name: ",
          tensor.name(),
          ") raw_data size (",
          tensor.raw_data().size(),
          " bytes) is too small for the declared shape and type (",
          expected_bytes,
          " bytes required).");
    }
    if (tensor.data_type() == TensorProto::FLOAT6E2M3 || tensor.data_type() == TensorProto::FLOAT6E3M2) {
      // expected_raw_data_bytes() already rejected counts where nelem * 6 overflows.
      const auto used_bits = static_cast<uint8_t>((nelem * 6) % 8);
      if (expected_bytes > 0 && used_bits != 0) {
        const auto last_byte = static_cast<uint8_t>(tensor.raw_data()[expected_bytes - 1]);
        const auto unused_bits_mask = static_cast<uint8_t>(0xFFU << used_bits);
        if ((last_byte & unused_bits_mask) != 0) {
          fail_check(
              "TensorProto (tensor name: ",
              tensor.name(),
              ") has non-zero padding bits in its packed FLOAT6 raw_data.");
        }
      }
    }
    return;
  } else {
#define check_field(field)               \
  if (nelem != 0 && !has_##field) {      \
    fail_check(                          \
        "values of data_type '",         \
        tensor.data_type(),              \
        "' should be stored in field '", \
        #field,                          \
        "' instead of '",                \
        value_field,                     \
        "'");                            \
  }

    switch (tensor.data_type()) {
      case TensorProto::FLOAT:
      case TensorProto::COMPLEX64:
        check_field(float_data);
        if (nelem > 0) {
          // COMPLEX64 stores interleaved real/imaginary parts: 2 float_data entries per element.
          int64_t expected_floats = 0;
          if (checked_mul_overflow(nelem, tensor.data_type() == TensorProto::COMPLEX64 ? 2 : 1, &expected_floats)) {
            fail_check(
                "TensorProto (tensor name: ",
                tensor.name(),
                ") has a shape too large to validate against its data type.");
          }
          if (static_cast<int64_t>(tensor.float_data().size()) < expected_floats) {
            fail_check(
                "TensorProto (tensor name: ",
                tensor.name(),
                ") float_data size (",
                tensor.float_data().size(),
                ") is too small for the declared shape (",
                expected_floats,
                " float values required).");
          }
        }
        break;

      case TensorProto::DOUBLE:
      case TensorProto::COMPLEX128:
        check_field(double_data);
        if (nelem > 0) {
          // COMPLEX128 stores interleaved real/imaginary parts: 2 double_data entries per element.
          int64_t expected_doubles = 0;
          if (checked_mul_overflow(nelem, tensor.data_type() == TensorProto::COMPLEX128 ? 2 : 1, &expected_doubles)) {
            fail_check(
                "TensorProto (tensor name: ",
                tensor.name(),
                ") has a shape too large to validate against its data type.");
          }
          if (static_cast<int64_t>(tensor.double_data().size()) < expected_doubles) {
            fail_check(
                "TensorProto (tensor name: ",
                tensor.name(),
                ") double_data size (",
                tensor.double_data().size(),
                ") is too small for the declared shape (",
                expected_doubles,
                " double values required).");
          }
        }
        break;

      case TensorProto::INT32:
      case TensorProto::UINT8:
      case TensorProto::INT8:
      case TensorProto::UINT16:
      case TensorProto::INT16:
      case TensorProto::BOOL:
      case TensorProto::FLOAT16:
      case TensorProto::BFLOAT16:
      case TensorProto::FLOAT8E4M3FN:
      case TensorProto::FLOAT8E4M3FNUZ:
      case TensorProto::FLOAT8E5M2:
      case TensorProto::FLOAT8E5M2FNUZ:
      case TensorProto::FLOAT8E8M0:
      case TensorProto::FLOAT6E2M3:
      case TensorProto::FLOAT6E3M2:
        check_field(int32_data);
        if (nelem > 0) {
          // These types are not packed: each element occupies one int32_data entry.
          const int64_t expected_int32s = nelem;
          if (static_cast<int64_t>(tensor.int32_data().size()) < expected_int32s) {
            fail_check(
                "TensorProto (tensor name: ",
                tensor.name(),
                ") int32_data size (",
                tensor.int32_data().size(),
                ") is too small for the declared shape (",
                expected_int32s,
                " int32 values required).");
          }
          if (tensor.data_type() == TensorProto::FLOAT6E2M3 || tensor.data_type() == TensorProto::FLOAT6E3M2) {
            for (const auto value : tensor.int32_data()) {
              if (value < 0 || value > 0x3F) {
                fail_check(
                    "TensorProto (tensor name: ", tensor.name(), ") FLOAT6 int32_data values must use only bits 0-5.");
              }
            }
          }
        }
        break;

      case TensorProto::UINT4:
      case TensorProto::INT4:
      case TensorProto::FLOAT4E2M1:
        check_field(int32_data);
        if (nelem > 0) {
          // Each int32 stores 4 bytes = 8 4-bit elements.
          const int64_t expected_int32s = (nelem / 8) + (nelem % 8 != 0);
          if (static_cast<int64_t>(tensor.int32_data().size()) < expected_int32s) {
            fail_check(
                "TensorProto (tensor name: ",
                tensor.name(),
                ") int32_data size (",
                tensor.int32_data().size(),
                ") is too small for the declared shape and packed type (",
                expected_int32s,
                " int32 values required).");
          }
        }
        break;
      case TensorProto::UINT2:
      case TensorProto::INT2:
        check_field(int32_data);
        if (nelem > 0) {
          // Each int32 stores 4 bytes = 16 2-bit elements.
          const int64_t expected_int32s = (nelem / 16) + (nelem % 16 != 0);
          if (static_cast<int64_t>(tensor.int32_data().size()) < expected_int32s) {
            fail_check(
                "TensorProto (tensor name: ",
                tensor.name(),
                ") int32_data size (",
                tensor.int32_data().size(),
                ") is too small for the declared shape and packed type (",
                expected_int32s,
                " int32 values required).");
          }
        }
        break;

      case TensorProto::INT64:
        check_field(int64_data);
        if (nelem > 0 && static_cast<int64_t>(tensor.int64_data().size()) < nelem) {
          fail_check(
              "TensorProto (tensor name: ",
              tensor.name(),
              ") int64_data size (",
              tensor.int64_data().size(),
              ") is too small for the declared shape (",
              nelem,
              " int64 values required).");
        }
        break;

      case TensorProto::UINT32:
      case TensorProto::UINT64:
        check_field(uint64_data);
        if (nelem > 0 && static_cast<int64_t>(tensor.uint64_data().size()) < nelem) {
          fail_check(
              "TensorProto (tensor name: ",
              tensor.name(),
              ") uint64_data size (",
              tensor.uint64_data().size(),
              ") is too small for the declared shape (",
              nelem,
              " uint64 values required).");
        }
        break;

      case TensorProto::STRING:
        check_field(string_data);
        if (nelem > 0 && static_cast<int64_t>(tensor.string_data_size()) < nelem) {
          fail_check(
              "TensorProto (tensor name: ",
              tensor.name(),
              ") string_data size (",
              tensor.string_data_size(),
              ") is too small for the declared shape (",
              nelem,
              " string values required).");
        }
        break;

      default:
        fail_check("Unrecognized data_type (tensor name: ", tensor.name(), "): ", tensor.data_type());
    }
  }

#undef check_field
}

void check_sequence(const SequenceProto& sequence, const CheckerContext& ctx) {
  enforce_has_field(sequence, elem_type);
  if (sequence.elem_type() == SequenceProto::TENSOR) {
    for (const TensorProto& tensor : sequence.tensor_values()) {
      check_tensor(tensor, ctx);
    }
  } else if (sequence.elem_type() == SequenceProto::SPARSE_TENSOR) {
    for (const SparseTensorProto& sparse_tensor : sequence.sparse_tensor_values()) {
      check_sparse_tensor(sparse_tensor, ctx);
    }
  } else if (sequence.elem_type() == SequenceProto::SEQUENCE) {
    for (const SequenceProto& seq : sequence.sequence_values()) {
      check_sequence(seq, ctx);
    }
  } else if (sequence.elem_type() == SequenceProto::MAP) {
    for (const MapProto& map : sequence.map_values()) {
      check_map(map, ctx);
    }
  } else {
    fail_check(
        "Sequence ( Structure name: ",
        sequence.name(),
        ", elem_type: ",
        sequence.elem_type(),
        ") is not have a valid element type.");
  }
}

void check_optional(const OptionalProto& optional, const CheckerContext& ctx) {
  enforce_has_field(optional, elem_type);
  if (optional.elem_type() == OptionalProto::UNDEFINED) {
    return;
  } else if (optional.elem_type() == OptionalProto::TENSOR) {
    if (optional.has_tensor_value())
      check_tensor(optional.tensor_value(), ctx);
  } else if (optional.elem_type() == OptionalProto::SPARSE_TENSOR) {
    if (optional.has_sparse_tensor_value())
      check_sparse_tensor(optional.sparse_tensor_value(), ctx);
  } else if (optional.elem_type() == OptionalProto::SEQUENCE) {
    if (optional.has_sequence_value())
      check_sequence(optional.sequence_value(), ctx);
  } else if (optional.elem_type() == OptionalProto::MAP) {
    if (optional.has_map_value())
      check_map(optional.map_value(), ctx);
  } else {
    fail_check(
        "Optional ( Structure name: ",
        optional.name(),
        ", elem_type: ",
        optional.elem_type(),
        ") is not have a valid element type.");
  }
}

void check_map(const MapProto& map, const CheckerContext& ctx) {
  enforce_has_field(map, key_type);
  if (map.key_type() == TensorProto::UNDEFINED) {
    fail_check("setting key_type field (map name: ", map.name(), ") to UNDEFINED is not allowed");
  }
  // Check if key is a valid type, specifically INT8, INT16, INT32, INT64,
  // UINT8, UINT16, UINT32, UINT64, or STRING.
  if ((map.key_type() == TensorProto::FLOAT) || (map.key_type() == TensorProto::BOOL) ||
      (map.key_type() == TensorProto::FLOAT16) || (map.key_type() == TensorProto::COMPLEX64) ||
      (map.key_type() == TensorProto::COMPLEX128)) {
    fail_check(
        "setting key_type field (map name: ",
        map.name(),
        ") to invalid TensorProto key_type ",
        map.key_type(),
        " is not allowed");
  }

  // MapProto will use either keys or string_keys, so only one should be > 0.
  if ((map.keys_size() > 0) && (map.string_keys_size() > 0)) {
    fail_check("Map (name: ", map.name(), ") should not contain more than one keys field.");
  }

  int num_keys = map.keys_size() + map.string_keys_size();
  int num_values = 0;

  enforce_has_field(map, values);
  check_sequence(map.values(), ctx);

  if (map.values().elem_type() == SequenceProto::TENSOR) {
    num_values = map.values().tensor_values_size();
  } else if (map.values().elem_type() == SequenceProto::SPARSE_TENSOR) {
    num_values = map.values().sparse_tensor_values_size();
  } else if (map.values().elem_type() == SequenceProto::SEQUENCE) {
    num_values = map.values().sequence_values_size();
  } else if (map.values().elem_type() == SequenceProto::MAP) {
    num_values = map.values().map_values_size();
  }

  if (num_keys != num_values) {
    fail_check("Length of map keys and map values are not the same (map name: ", map.name(), ")");
  }
}

// Check that the index data stored in a SparseTensorProto is valid.
// indices: a 1-dimensional tensor; indices[i] represents the
// linearized index value for the i-th nonzero value.
static void
check_sparse_tensor_indices_1(const TensorProto& indices, const SparseTensorProto& sparse_tensor_proto, size_t nnz) {
  int dense_rank = sparse_tensor_proto.dims_size();
  int64_t dense_size = 1;
  for (int i = 0; i < dense_rank; ++i)
    dense_size *= sparse_tensor_proto.dims(i);
  if (static_cast<size_t>(indices.dims(0)) != nnz) {
    fail_check("Sparse tensor indices (", indices.name(), ") has ", indices.dims(0), " values, but NNZ is ", nnz);
  }

  // Check if indices appear in ascending order, and if they have valid
  // values. The i-th value in index_data is the linear index of the i-th
  // non-zero value.
  const std::vector<int64_t> index_data = ParseData<int64_t>(&indices);

  int64_t prev_index = -1;
  for (size_t i = 0; i < nnz; ++i) {
    int64_t curr_index = index_data[i]; // linearized index of i-th value
    if (curr_index < 0 || curr_index >= dense_size) {
      fail_check(
          "Sparse tensor (",
          indices.name(),
          ") index value at position [",
          i,
          "] out of range [0, ",
          dense_size - 1,
          "]");
    }
    if (curr_index <= prev_index) {
      fail_check("Sparse tensor (", indices.name(), ") index value at position [", i, "] not in sorted order.");
    }
    prev_index = curr_index;
  }
}

// Check that the index data stored in a SparseTensorProto is valid.
// indices: a 2-dimensional tensor; indices[i,j] represents the j-th
// index value for the i-th nonzero value.
static void
check_sparse_tensor_indices_2(const TensorProto& indices, const SparseTensorProto& sparse_tensor_proto, size_t nnz) {
  int dense_rank = sparse_tensor_proto.dims_size();
  if (static_cast<size_t>(indices.dims(0)) != nnz) {
    fail_check("Sparse tensor indices (", indices.name(), ") first dimension size does not equal NNZ.");
  }
  if (indices.dims(1) != dense_rank) {
    fail_check("Sparse tensor indices (", indices.name(), ") second dimension size does not match rank of tensor.");
  }

  // Check if indices appear in ascending order, and if they have valid
  // values.
  const std::vector<int64_t> index_data = ParseData<int64_t>(&indices);
  int64_t prev_index = -1;
  for (size_t i = 0; i < nnz; ++i) {
    int64_t curr_index = 0; // linearized index of i-th value
    for (int j = 0; j < dense_rank; ++j) {
      auto index_ij = index_data[(i * dense_rank) + j];
      if ((index_ij < 0) || (index_ij >= sparse_tensor_proto.dims(j))) {
        fail_check("Sparse tensor (", indices.name(), ") index value at position [", i, ",", j, "] out of range.");
      }
      curr_index = (curr_index * sparse_tensor_proto.dims(j)) + index_ij;
    }
    if (curr_index <= prev_index) {
      fail_check(
          "Sparse tensor (", indices.name(), ") index value at position [", i, "] not in lexicographic sorted order.");
    }
    prev_index = curr_index;
  }
}

void check_sparse_tensor(const SparseTensorProto& sparse_tensor_proto, const CheckerContext& ctx) {
  enforce_has_field(sparse_tensor_proto, values);

  const TensorProto& values = sparse_tensor_proto.values();
  check_tensor(values, ctx);

  // values must be a tensor of shape [NNZ]
  // Currently we restrict the value associated with a particular index-tuple
  // to be a single value. In the future, if there is a requirement,
  // we may extend this to permit the value to be a "sub-tensor", in which
  // case values will have dimension > 1.
  if (values.dims_size() != 1) {
    fail_check("Sparse tensor values (", values.name(), ") must have rank 1.");
  }
  size_t nnz = static_cast<size_t>(values.dims(0));
  int dense_rank = sparse_tensor_proto.dims_size();
  if (dense_rank == 0) {
    fail_check("Sparse tensor (", values.name(), ") must have a dense-rank > 0");
  }
  for (int i = 0; i < dense_rank; ++i) {
    if (sparse_tensor_proto.dims(i) <= 0) {
      fail_check("Sparse tensor (", values.name(), ") dimensions are not positive.");
    }
  }

  if (sparse_tensor_proto.has_indices()) {
    const TensorProto& indices = sparse_tensor_proto.indices();
    check_tensor(indices, ctx);
    if (indices.data_type() != TensorProto::INT64) {
      fail_check("Sparse tensor indices (", indices.name(), ") must have INT64 type.");
    }
    switch (indices.dims().size()) {
      case 1:
        // Indices in linearized format
        check_sparse_tensor_indices_1(indices, sparse_tensor_proto, nnz);
        return;
      case 2:
        // Check COO-style index. E.g., an index for a 3D tensor is a 3-tuple.
        check_sparse_tensor_indices_2(indices, sparse_tensor_proto, nnz);
        return;
      default:
        fail_check("Sparse tensor indices (", indices.name(), ") must have rank 1 or 2.");
    }
  } else if (nnz != 0) {
    fail_check("Sparse tensor (", values.name(), ") has no index values.");
  }
}

// NB: This is a generic "attribute well-formedness" check, it doesn't
// actually test if an attribute is valid per a schema
void check_attribute(const AttributeProto& attr, const CheckerContext& ctx, const LexicalScopeContext& lex_ctx) {
  enforce_non_empty_field(attr, name);

  if (ctx.get_ir_version() >= 0x00000002) {
    enforce_has_field(attr, type);
  }

  int used_fields = 0;

#define check_type(expected_type)                                                     \
  if (attr.has_type() && attr.type() != (expected_type)) {                            \
    fail_check("type field and data field mismatch in attribute ", attr.name(), "."); \
  }

#define check_singular_field(field, type) \
  if (attr.has_##field()) {               \
    ++used_fields;                        \
    check_type(type);                     \
  }

#define check_repeated_field(field, type) \
  if (attr.field##_size() > 0) {          \
    ++used_fields;                        \
    check_type(type);                     \
  }

  check_singular_field(f, AttributeProto::FLOAT);
  check_singular_field(i, AttributeProto::INT);
  check_singular_field(s, AttributeProto::STRING);
  check_singular_field(t, AttributeProto::TENSOR);
  check_singular_field(g, AttributeProto::GRAPH);
  check_singular_field(tp, AttributeProto::TYPE_PROTO);
  check_singular_field(sparse_tensor, AttributeProto::SPARSE_TENSOR);
  check_repeated_field(floats, AttributeProto::FLOATS);
  check_repeated_field(ints, AttributeProto::INTS);
  check_repeated_field(strings, AttributeProto::STRINGS);
  check_repeated_field(tensors, AttributeProto::TENSORS);
  check_repeated_field(graphs, AttributeProto::GRAPHS);
  check_repeated_field(sparse_tensors, AttributeProto::SPARSE_TENSORS);
  check_repeated_field(type_protos, AttributeProto::TYPE_PROTOS);

#undef check_type
#undef check_singular_field
#undef check_repeated_field

  // Normally, used_fields is expected to be 1.
  // In proto3, when the value to be set is type default value (say 0 for
  // int), used_fields may be 0.
  if (used_fields > 1) {
    fail_check("Attribute (name: ", attr.name(), ") should not contain more than one value field.");
  }

  if ((!ctx.is_main_graph()) && (attr.has_ref_attr_name() && used_fields != 0))
  // It's an attribute of a node in function body.
  {
    // The attribute proto is supposed to refer to data outside and does not
    // have its own value field set.
    fail_check("Attribute (name: ", attr.name(), ") should refer to attribute in parent node.");
  }

  if (attr.has_t()) {
    check_tensor(attr.t(), ctx);
  }

  if (attr.has_sparse_tensor()) {
    check_sparse_tensor(attr.sparse_tensor(), ctx);
  }

  if (attr.has_g()) {
    CheckerContext subgraph_ctx(ctx);
    subgraph_ctx.set_is_main_graph(false);
    check_graph(attr.g(), subgraph_ctx, lex_ctx);
  }

  for (const auto& tensor : attr.tensors()) {
    check_tensor(tensor, ctx);
  }
  for (const auto& sparse_tensor : attr.sparse_tensors()) {
    check_sparse_tensor(sparse_tensor, ctx);
  }
  if (!attr.graphs().empty()) {
    CheckerContext subgraph_ctx(ctx);
    subgraph_ctx.set_is_main_graph(false);
    for (const auto& graph : attr.graphs()) {
      check_graph(graph, subgraph_ctx, lex_ctx);
    }
  }
}

static void print_warning_if_has_experimental(const std::unordered_set<std::string>& used_experimental_ops) {
  if (!used_experimental_ops.empty()) {
    std::string all_experimental_ops;
    for (const auto& op : used_experimental_ops) {
      all_experimental_ops += " " + op + ",";
    }
    // Remove the last comma which is unnecessary
    all_experimental_ops.pop_back();
    std::cout << "Warning: Model contains experimental ops:" + all_experimental_ops << '\n';
  }
}

void check_node(const NodeProto& node, const CheckerContext& ctx, const LexicalScopeContext& lex_ctx) {
  enforce_non_empty_field(node, op_type);

  if (node.input().empty() && node.output().empty()) {
    fail_check("NodeProto (name: ", node.name(), ", type: ", node.op_type(), ") has zero input and zero output.");
  }

  // Resolve domain for node
  const auto& opset_imports = ctx.get_opset_imports();
  auto dit = opset_imports.find(node.domain());
  if (dit == opset_imports.end()) {
    // Both "" (ONNX_DOMAIN) and "ai.onnx" (AI_ONNX_DOMAIN) refer to the default ONNX domain
    if (node.domain() == ONNX_DOMAIN) {
      dit = opset_imports.find(AI_ONNX_DOMAIN);
    }
    if (dit == opset_imports.end()) {
      fail_check_with_code(error_code::kOpsetImportMissing, "No opset import for domain '" + node.domain() + "'");
    }
  }
  auto domain_version = dit->second;

  // for ops referencing local functions, there is no schema to verify it.
  // will add a check to verify consistency between these ops and local functions.
  std::unordered_set<std::string> seen_attr_names{};
  for (const auto& attr : node.attribute()) {
    if (!seen_attr_names.insert(attr.name()).second) {
      fail_check_with_code(error_code::kNodeSchemaMismatch, "Attribute '", attr.name(), "' appeared multiple times.");
    }

    check_attribute(attr, ctx, lex_ctx);
  }

  // This issue will be caught by check_graph instead
  if (check_is_experimental_op(node)) {
    return;
  }

  const auto* const schema = ctx.get_schema_registry()->GetSchema(node.op_type(), domain_version, node.domain());
  if (!schema) {
    if (node.domain() == ONNX_DOMAIN || node.domain() == AI_ONNX_ML_DOMAIN || node.domain() == "ai.onnx" ||
        node.domain() == AI_ONNX_TRAINING_DOMAIN || ctx.check_custom_domain()) {
      // fail the checker if op is in built-in domains or if it has no schema when `check_custom_domain` is true
      fail_check_with_code(
          error_code::kOperatorNotRegistered,
          "No Op registered for " + node.op_type() + " with domain_version of " +
              ONNX_NAMESPACE::to_string(domain_version));
    }
  } else if (schema->Deprecated()) {
    fail_check_with_code(
        error_code::kOperatorDeprecated,
        "Op registered for " + node.op_type() + " is deprecated in domain_version of " +
            ONNX_NAMESPACE::to_string(domain_version));
  } else {
    schema->Verify(node);
  }
}

void check_graph(const GraphProto& graph, const CheckerContext& ctx, const LexicalScopeContext& parent_lex) {
  enforce_non_empty_field(graph, name);

  for (const auto& value_info : graph.input()) {
    check_value_info(value_info, ctx);
  }
  for (const auto& value_info : graph.output()) {
    check_value_info(value_info, ctx);
  }
  // Intermediate value_info entries are optional annotations and are not
  // otherwise validated, but they must not use data types newer than the IR version.
  for (const auto& value_info : graph.value_info()) {
    check_type_ir_version(value_info.type(), value_info.name(), ctx);
  }

  // Inherit values available in outer scope
  // Note that we do not allow shadowing, so the presence of an already-defined
  // name is always an error.
  LexicalScopeContext lex_ctx{parent_lex};

  for (const auto& value_info : graph.input()) {
    // TODO(ONNX): If shadowing isn't allowed, this should maybe use
    // this_or_ancestor_graph_has
    if (lex_ctx.this_graph_has(value_info.name())) {
      fail_check_with_code(
          error_code::kNotSsa,
          "Graph must be in single static assignment (SSA) form, however '",
          value_info.name(),
          "' has been used as graph input names multiple times.");
    }
    lex_ctx.add(value_info.name());
  }

  std::unordered_set<std::string> initializer_name_checker;

  for (const auto& init : graph.initializer()) {
    enforce_has_field(init, name);
    const auto& name = init.name();
    if (name.empty()) {
      fail_check_with_code(error_code::kRequiredFieldEmpty, "Tensor initializers must have a non-empty name");
    }

    if (!initializer_name_checker.emplace(name).second) {
      fail_check_with_code(error_code::kNotSsa, name + " initializer name is not unique");
    }

    check_tensor(init, ctx);

    if (ctx.get_ir_version() <= 0x00000003) {
      // Initializers are a subset of graph inputs for IR_VERSION <= 3
      if (!lex_ctx.this_graph_has(name)) {
        fail_check(name + " in initializer but not in graph input");
      }
    } else {
      // An initializer is allowed to have the same name as an input,
      // but is not required to (for IR_VERSION >= 4)
      lex_ctx.add(name);
    }
  }

  for (const auto& sparse_init : graph.sparse_initializer()) {
    const auto& values = sparse_init.values();
    enforce_has_field(values, name);
    const auto& name = values.name();
    if (name.empty()) {
      fail_check_with_code(error_code::kRequiredFieldEmpty, "Sparse tensor initializers must have a non-empty name");
    }
    if (!initializer_name_checker.insert(name).second) {
      fail_check_with_code(
          error_code::kNotSsa,
          name + " sparse initializer name is not unique across initializers and sparse_initializers");
    }
    check_sparse_tensor(sparse_init, ctx);
    lex_ctx.add(name);
  }
  std::unordered_set<std::string> used_experimental_ops;
  for (const auto& node : graph.node()) {
    // nodes must be in topologically sorted order
    for (const auto& input : node.input()) {
      // explicit optional input
      if (input.empty()) {
        continue;
      }
      if (!lex_ctx.this_or_ancestor_graph_has(input)) {
        fail_check_with_code(
            error_code::kInputNotDefined,
            "Nodes in a graph must be topologically sorted, however input '",
            input,
            "' of node: \n",
            "name: ",
            node.name(),
            " OpType: ",
            node.op_type(),
            "\n is not output of any previous nodes.");
      }
    }

    if (check_is_experimental_op(node)) {
      used_experimental_ops.insert(node.op_type());
    }

    // This needs to happen before SSA check since we don't want to recurse and
    // find that outputs from control flow ops are colliding with names in the
    // inner block

    ONNX_TRY {
      check_node(node, ctx, lex_ctx);
    }
    ONNX_CATCH(ValidationError & ex) {
      ONNX_HANDLE_EXCEPTION([&]() {
        ex.AppendContext("Bad node spec for node. Name: " + node.name() + " OpType: " + node.op_type());
        // Rethrow without copying to avoid triggering
        // bugprone-exception-copy-constructor-throws.
        // The in-place AppendContext modification is preserved because
        // `ex` is a reference to the active exception object.
        throw;
      });
    }
    // check for SSA form
    for (const auto& output : node.output()) {
      // optional output
      if (output.empty()) {
        continue;
      }

      if (lex_ctx.this_or_ancestor_graph_has(output)) {
        fail_check_with_code(
            error_code::kNotSsa,
            "Graph must be in single static assignment (SSA) form, however '",
            output,
            "' has been used as output names multiple times.");
      }
      lex_ctx.add(output);
    }
  }
  for (const auto& value_info : graph.output()) {
    if (!lex_ctx.this_graph_has(value_info.name())) {
      fail_check_with_code(
          error_code::kGraphOutputNotProduced,
          "Graph output '",
          value_info.name(),
          "' is not an output of any node in graph.");
    }
  }

  print_warning_if_has_experimental(used_experimental_ops);
}

// Converts a proto opset version to int, rejecting values that do not fit.
static int checked_opset_version(int64_t version) {
  if (version < std::numeric_limits<int>::min() || version > std::numeric_limits<int>::max()) {
    fail_check("Opset import version ", version, " is out of supported range");
  }
  return static_cast<int>(version);
}

// Utilify function to get the imported version of domain from opset imports
// Returns -1 if requested domain is not found in the opset_imports
static int get_version_for_domain(
    const std::string& domain,
    const std::unordered_map<std::string, int>& opset_imports) {
  auto it = opset_imports.find(domain);
  if (it == opset_imports.end()) {
    return -1;
  }

  return it->second;
}

void check_opset_compatibility(
    const NodeProto& node,
    const CheckerContext& ctx,
    const std::unordered_map<std::string, int>& func_opset_imports,
    const std::unordered_map<std::string, int>& model_opset_imports) {
  auto func_opset_version = get_version_for_domain(node.domain(), func_opset_imports);
  auto model_opset_version = get_version_for_domain(node.domain(), model_opset_imports);

  if (func_opset_version == -1) {
    fail_check_with_code(error_code::kOpsetImportMissing, "No Opset registered for domain " + node.domain());
  }

  if (model_opset_version == -1) {
    // model does not include opset import for a node present in function body.
    // This is ok as along as the opset import is present in function level opset imports.
    return;
  }

  if (func_opset_version == model_opset_version) {
    // both versions are same, no need to verify schema.
    return;
  }

  const auto* const schema_for_model_import =
      ctx.get_schema_registry()->GetSchema(node.op_type(), model_opset_version, node.domain());

  const auto* const schema_for_function_import =
      ctx.get_schema_registry()->GetSchema(node.op_type(), func_opset_version, node.domain());

  if (!schema_for_model_import && !schema_for_function_import) {
    // the op belongs to a custom domain so we cannot verify schema
    return;
  }

  // if schema is present for 1 but not other or the schema since versions do not match then raise an error
  if (!schema_for_model_import || !schema_for_function_import ||
      schema_for_function_import->since_version() != schema_for_model_import->since_version()) {
    fail_check(
        "Opset import for domain " + node.domain() + " in function op " + node.op_type() +
        "is not compatible with the version imported by model. FunctionOp imports version " +
        ONNX_NAMESPACE::to_string(func_opset_version) + " whereas model imports version " +
        ONNX_NAMESPACE::to_string(model_opset_version));
  }
}

namespace {

using FuncPtr = const FunctionProto*;
using CallGraph = std::unordered_map<FuncPtr, std::unordered_set<FuncPtr>>;

// Limits to reject obviously malicious models early.
constexpr int kMaxModelLocalFunctions = 10000;
constexpr size_t kMaxFunctionCallDepth = 100;

enum class VisitState : uint8_t { Unvisited, InPath, Done };

// Iterative DFS to detect cycles in the function call graph.
// Uses an explicit stack to avoid stack overflow on deeply-chained models.
void DetectCycleDFS(
    FuncPtr root,
    const CallGraph& call_graph,
    std::unordered_map<FuncPtr, VisitState>& state,
    std::vector<FuncPtr>& path) {
  // Each frame tracks the current function and an iterator into its callees.
  using Iter = std::unordered_set<FuncPtr>::const_iterator;
  struct Frame {
    FuncPtr func;
    Iter cur;
    Iter end;
  };
  std::vector<Frame> stack;

  auto push = [&](FuncPtr func) {
    state[func] = VisitState::InPath;
    path.push_back(func);
    if (path.size() > kMaxFunctionCallDepth) {
      fail_check(
          "Function call chain depth exceeds limit (",
          kMaxFunctionCallDepth,
          "). The model may be malformed or malicious.");
    }
    auto it = call_graph.find(func);
    if (it != call_graph.end()) {
      stack.push_back({func, it->second.begin(), it->second.end()});
    } else {
      stack.push_back({func, {}, {}});
    }
  };

  push(root);

  while (!stack.empty()) {
    auto& frame = stack.back();

    if (frame.cur == frame.end) {
      // All callees processed — backtrack.
      path.pop_back();
      state[frame.func] = VisitState::Done;
      stack.pop_back();
      continue;
    }

    FuncPtr callee = *frame.cur;
    ++frame.cur;

    auto s = state[callee];
    if (s == VisitState::InPath) {
      auto start = std::find(path.begin(), path.end(), callee);
      std::string cycle;
      for (auto cit = start; cit != path.end(); ++cit)
        cycle += (cit == start ? "" : " -> ") + GetFunctionImplId(**cit);
      fail_check(
          "Cycle detected in model-local function references: ",
          cycle,
          " -> ",
          GetFunctionImplId(*callee),
          ". Model-local functions must not be recursive.");
    } else if (s == VisitState::Unvisited) {
      push(callee);
    }
  }
}

// Collect callee edges from a list of nodes into the callee set. A node in a
// function body may carry subgraphs (via GRAPH / GRAPHS attributes on control-flow
// ops such as If/Loop/Scan) whose own nodes can also call model-local functions, so
// we descend recursively to detect cycles hidden at any nesting level. This follows the
// same subgraph descent as the existing checker recursion (check_graph -> check_node ->
// check_attribute -> check_graph) and does not add a new unbounded-recursion characteristic.
void CollectCalleeEdges(
    const google::protobuf::RepeatedPtrField<NodeProto>& nodes,
    const std::unordered_map<std::string, FuncPtr>& func_by_key,
    std::unordered_set<FuncPtr>& callees) {
  for (const auto& node : nodes) {
    auto it = func_by_key.find(GetCalleeId(node));
    if (it != func_by_key.end()) {
      callees.insert(it->second);
    }
    // has_g() guards the singular GRAPH attribute; graphs() is the repeated GRAPHS field.
    for (const auto& attr : node.attribute()) {
      if (attr.has_g()) {
        CollectCalleeEdges(attr.g().node(), func_by_key, callees);
      }
      for (const auto& subgraph : attr.graphs()) {
        CollectCalleeEdges(subgraph.node(), func_by_key, callees);
      }
    }
  }
}

} // namespace

void check_function_call_cycles(const ModelProto& model) {
  if (model.functions_size() > kMaxModelLocalFunctions) {
    fail_check(
        "Model contains ",
        model.functions_size(),
        " local functions, exceeding the limit of ",
        kMaxModelLocalFunctions,
        ". The model may be malformed or malicious.");
  }

  // Build function map: callee key -> FunctionProto pointer.
  // Duplicate keys could mask cycles, so reject them here.
  std::unordered_map<std::string, FuncPtr> func_by_key;
  for (const auto& f : model.functions()) {
    const auto function_impl_id = GetFunctionImplId(f);
    const bool inserted = func_by_key.emplace(function_impl_id, &f).second;
    if (!inserted) {
      fail_check("Model contains multiple local functions with the same implementation id '", function_impl_id, "'.");
    }
  }

  // Build adjacency list of FuncPtr edges. FuncPtr points into model.functions(), which
  // outlives this local map, so storing raw pointers as graph nodes is safe here.
  CallGraph call_graph;
  for (const auto& entry : func_by_key) {
    const auto* func = entry.second;
    CollectCalleeEdges(func->node(), func_by_key, call_graph[func]);
  }

  std::unordered_map<FuncPtr, VisitState> state;
  std::vector<FuncPtr> path;
  for (const auto& entry : func_by_key) {
    if (state[entry.second] == VisitState::Unvisited) {
      DetectCycleDFS(entry.second, call_graph, state, path);
    }
  }
}

void check_model_local_functions(
    const ModelProto& model,
    const CheckerContext& ctx,
    const LexicalScopeContext& parent_lex) {
  // make a copy of model opset imports to maintain a main copy of opset imports across the model and
  // all model local functions to verify opset compatibility
  std::unordered_map<std::string, int> model_opset_imports(ctx.get_opset_imports());

  // merge the opset imports from every function in model_opset_imports
  // only add the opset import if an entry for it does not exist in model_opset_imports
  // if there is an entry then the compatibility will be checked later on in check_opset_compatibility
  // called by check_function.
  for (const auto& function_proto : model.functions()) {
    for (const auto& opset_import : function_proto.opset_import()) {
      if (get_version_for_domain(opset_import.domain(), model_opset_imports) == -1) {
        model_opset_imports[opset_import.domain()] = checked_opset_version(opset_import.version());
      }
    }
  }

  check_function_call_cycles(model);

  CheckerContext ctx_copy = ctx;
  ctx_copy.set_opset_imports(model_opset_imports);

  for (const auto& function_proto : model.functions()) {
    check_function(function_proto, ctx_copy, parent_lex);
  }
}

void check_function(const FunctionProto& function, const CheckerContext& ctx, const LexicalScopeContext& parent_lex) {
  enforce_non_empty_field(function, name);

  if (ctx.get_ir_version() >= 0x00000008) {
    enforce_has_field(function, domain);
  }

  const auto& model_opset_imports = ctx.get_opset_imports();
  CheckerContext ctx_copy = ctx;

  std::unordered_map<std::string, int> func_opset_imports;
  for (const auto& relied_opset : function.opset_import()) {
    func_opset_imports[relied_opset.domain()] = checked_opset_version(relied_opset.version());
  }

  ctx_copy.set_opset_imports(func_opset_imports);

  LexicalScopeContext lex_ctx{parent_lex};

  for (const auto& input : function.input()) {
    // TODO(ONNX): If shadowing isn't allowed, this should maybe use
    // this_or_ancestor_graph_has
    if (lex_ctx.this_graph_has(input)) {
      fail_check_with_code(
          error_code::kNotSsa,
          "Graph must be in single static assignment (SSA) form, however '",
          input,
          "' has been used multiple times.");
    }
    lex_ctx.add(input);
  }

  std::unordered_set<std::string> outputs;
  for (const auto& output : function.output()) {
    if (!outputs.insert(output).second) {
      fail_check("function (", function.name(), ") should not have duplicate outputs specified.");
    }
  }

  std::unordered_set<std::string> attrs;
  for (const auto& attr : function.attribute()) {
    if (!attrs.insert(attr).second) {
      fail_check("function (", function.name(), ") should not have duplicate attributes specified.");
    }
  }
  for (const auto& attr : function.attribute_proto()) {
    check_attribute_tensors_ir_version(attr, ctx);
  }
  // Function inputs and outputs are untyped names; their types, if any, are given here.
  for (const auto& value_info : function.value_info()) {
    check_type_ir_version(value_info.type(), value_info.name(), ctx);
  }
  std::unordered_set<std::string> used_experimental_ops;
  for (const auto& node : function.node()) {
    // nodes must be in topologically sorted order
    for (const auto& input : node.input()) {
      // explicit optional input
      if (input.empty()) {
        continue;
      }
      if (!lex_ctx.this_graph_has(input)) {
        fail_check_with_code(
            error_code::kInputNotDefined,
            "Nodes in a function must be topologically sorted, however input '",
            input,
            "' of node: \n",
            "Name: ",
            node.name(),
            " OpType: ",
            node.op_type(),
            "\n is neither output of any previous nodes nor input of the function.");
      }
    }

    // check whether the opset version imported for a domain by function and model are
    // compatible
    if (!ctx_copy.skip_opset_compatibility_check())
      check_opset_compatibility(node, ctx_copy, func_opset_imports, model_opset_imports);
    if (check_is_experimental_op(node)) {
      used_experimental_ops.insert(node.op_type());
    }
    check_node(node, ctx_copy, lex_ctx);

    // check for SSA form
    for (const auto& output : node.output()) {
      // optional output
      if (output.empty()) {
        continue;
      }
      if (lex_ctx.this_or_ancestor_graph_has(output)) {
        fail_check_with_code(
            error_code::kNotSsa,
            "Function must be in single static assignment (SSA) form, however '",
            output,
            "' has been used as output names multiple times.");
      }
      lex_ctx.add(output);
    }
  }
  print_warning_if_has_experimental(used_experimental_ops);
}

static void check_model(const ModelProto& model, CheckerContext& ctx) {
  if (!model.ir_version()) {
    fail_check_with_code(error_code::kIrVersionMissing, "The model does not have an ir_version set properly.");
  }
  if (model.ir_version() > IR_VERSION) {
    fail_check_with_code(
        error_code::kIrVersionUnsupported,
        "Your model ir_version ",
        model.ir_version(),
        " is higher than the checker's (",
        IR_VERSION,
        ").");
  }
  if (model.metadata_props_size() > 1) {
    std::unordered_set<std::string> keys;
    for (const StringStringEntryProto& entry : model.metadata_props()) {
      auto i = keys.insert(entry.key());
      if (!i.second) {
        fail_check_with_code(
            error_code::kMetadataPropsDuplicateKey, "Your model has duplicate keys in metadata_props.");
      }
    }
  }
  ctx.set_ir_version(static_cast<int>(model.ir_version()));
  std::unordered_map<std::string, int> opset_imports;
  for (const auto& opset_import : model.opset_import()) {
    opset_imports[opset_import.domain()] = checked_opset_version(opset_import.version());
  }
  if (model.ir_version() >= 3) {
    if (opset_imports.empty()) {
      fail_check_with_code(
          error_code::kOpsetImportMissing, "model with IR version >= 3 must specify opset_import for ONNX");
    }
  } else {
    if (opset_imports.empty()) {
      opset_imports[ONNX_DOMAIN] = 1;
    } else {
      fail_check("model with IR version < 3 cannot have opset_import specified");
    }
  }
  ctx.set_opset_imports(opset_imports);
  LexicalScopeContext lex_ctx;
  check_graph(model.graph(), ctx, lex_ctx);

  if (ctx.get_ir_version() >= 0x00000008) {
    check_model_local_functions(model, ctx, lex_ctx);
    // TODO(ONNX): check consistency between local functions and ops referencing it.
  }
}

void check_model(
    const std::string& model_path,
    bool full_check,
    bool skip_opset_compatibility_check,
    bool check_custom_domain) {
  CheckerContext ctx;
  const auto native_model_path = utf8_to_path(model_path);
  const std::string model_dir = path_to_utf8(native_model_path.parent_path());
  const std::string model_filename = path_to_utf8(native_model_path.filename());
  ctx.set_model_dir(model_dir);
  const auto raw_model_fd = ctx.get_external_data_context()->open_model_file(model_filename, true);
  ONNX_NAMESPACE::ScopedFd model_fd(static_cast<int>(raw_model_fd));
  std::string model_bytes;
#ifdef _WIN32
  struct _stat64 model_stat{};
  if (_fstat64(model_fd.get(), &model_stat) == 0 && model_stat.st_size > 0 &&
      static_cast<std::uintmax_t>(model_stat.st_size) <= model_bytes.max_size()) {
    model_bytes.reserve(static_cast<size_t>(model_stat.st_size));
  }
#else
  struct stat model_stat{};
  if (fstat(model_fd.get(), &model_stat) == 0 && model_stat.st_size > 0 &&
      static_cast<std::uintmax_t>(model_stat.st_size) <= model_bytes.max_size()) {
    model_bytes.reserve(static_cast<size_t>(model_stat.st_size));
  }
#endif
  std::array<char, 64 * 1024> buffer{};
  while (true) {
#ifdef _WIN32
    const int count = _read(model_fd.get(), buffer.data(), static_cast<unsigned int>(buffer.size()));
#else
    const ssize_t count = read(model_fd.get(), buffer.data(), buffer.size());
#endif
    if (count < 0) {
      fail_check("Unable to read proto file: ", model_path, ". Please check if it is a valid proto. ");
    }
    if (count == 0) {
      break;
    }
    model_bytes.append(buffer.data(), static_cast<size_t>(count));
  }
  ModelProto model;
  if (!ParseProtoFromBytes(&model, model_bytes.data(), model_bytes.size())) {
    fail_check(
        "Unable to parse proto from file: ", model_path, ". Please check if it is a valid protobuf file of proto. ");
  }
  ctx.set_skip_opset_compatibility_check(skip_opset_compatibility_check);
  ctx.set_check_custom_domain(check_custom_domain);
  check_model(model, ctx);

  if (full_check) {
    ShapeInferenceOptions options{true, 1, false};
    ONNX_NAMESPACE::shape_inference::InferShapes(model, ctx.get_schema_registry(), options);
  }
}

void check_model(
    const ModelProto& model,
    bool full_check,
    bool skip_opset_compatibility_check,
    bool check_custom_domain) {
  CheckerContext ctx;
  ctx.set_skip_opset_compatibility_check(skip_opset_compatibility_check);
  ctx.set_check_custom_domain(check_custom_domain);
  check_model(model, ctx);
  if (full_check) {
    ShapeInferenceOptions options{true, 1, false};
    // Do not update the model in place by the check from shape inference
    // because checker should not modify the original model
    ModelProto copy = model;
    ONNX_NAMESPACE::shape_inference::InferShapes(copy, ctx.get_schema_registry(), options);
  }
}

using ONNX_NAMESPACE::ScopedFd;
using ONNX_NAMESPACE::utf8_to_path;
#ifdef _WIN32
using ONNX_NAMESPACE::ScopedHandle;
#endif

static std::filesystem::path validate_external_data_relative_path(
    const std::string& location,
    const std::string& tensor_name) {
  auto file_path = utf8_to_path(location);
  if (file_path.empty() || file_path.is_absolute() || file_path.has_root_name() || file_path.has_root_directory()) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid,
        "External data location for tensor ",
        tensor_name,
        " is empty or absolute: ",
        location);
  }
  auto relative_path = file_path.lexically_normal().make_preferred();
  if (relative_path == std::filesystem::path(".") ||
      relative_path.native().find(std::filesystem::path("..").native()) !=
          std::filesystem::path::string_type::npos) { // NOSONAR — preserve the existing conservative check
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid,
        "External data location for tensor ",
        tensor_name,
        " contains '..' or does not name a file: ",
        location);
  }
  return relative_path;
}

static std::vector<std::filesystem::path> external_data_path_components(
    const std::string& location,
    const std::string& tensor_name) {
  auto relative_path = validate_external_data_relative_path(location, tensor_name);
  std::vector<std::filesystem::path> components;
  for (const auto& component : relative_path) {
    if (component.empty() || component == std::filesystem::path(".") || component == std::filesystem::path("..")) {
      fail_check_with_code(
          error_code::kExternalDataLocationInvalid,
          "External data location for tensor ",
          tensor_name,
          " has an invalid path component: ",
          location);
    }
    components.push_back(component);
  }
  if (components.empty()) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid,
        "External data location for tensor ",
        tensor_name,
        " does not name a file: ",
        location);
  }
  return components;
}

static std::vector<std::filesystem::path> model_file_path_components(const std::string& location) {
  auto file_path = utf8_to_path(location);
  if (file_path.empty() || file_path.is_absolute() || file_path.has_root_name() || file_path.has_root_directory()) {
    fail_check("Model filename is empty or absolute: ", location);
  }
  auto relative_path = file_path.lexically_normal().make_preferred();
  std::vector<std::filesystem::path> components;
  for (const auto& component : relative_path) {
    if (component.empty() || component == std::filesystem::path(".") || component == std::filesystem::path("..")) {
      fail_check("Model filename has an invalid path component: ", location);
    }
    components.push_back(component);
  }
  if (components.empty()) {
    fail_check("Model filename does not name a file: ", location);
  }
  return components;
}

// Canonicalize data_path, verify containment within base_dir.
static std::filesystem::path verify_path_containment(
    const std::filesystem::path& data_path,
    const std::string& base_dir,
    const std::string& tensor_name) {
  std::error_code ec;
  auto canonical_data = std::filesystem::weakly_canonical(data_path, ec);
  if (ec) {
    fail_check("Tensor ", tensor_name, " external data path could not be canonicalized: ", ec.message());
  }
  // An empty base_dir means the current working directory. Passing an empty
  // path to weakly_canonical can leave the containment prefix empty, which
  // would accept every external-data path.
  const auto normalized_base_dir = base_dir.empty() ? std::string(".") : base_dir;
  auto canonical_base = std::filesystem::weakly_canonical(utf8_to_path(normalized_base_dir), ec);
  if (ec) {
    fail_check("Tensor ", tensor_name, " base directory could not be canonicalized: ", ec.message());
  }
  auto base_str = canonical_base.native();
  if (!base_str.empty() && base_str.back() != std::filesystem::path::preferred_separator) {
    base_str += std::filesystem::path::preferred_separator;
  }
  if (canonical_data.native().find(base_str) != 0 && canonical_data != canonical_base) { // NOSONAR
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid,
        "Tensor ",
        tensor_name,
        " external data resolves outside model directory.");
  }
  return canonical_data;
}

std::filesystem::path resolve_external_data_location(
    const std::string& base_dir,
    const std::string& location,
    const std::string& tensor_name) {
  auto base_dir_path = utf8_to_path(base_dir);
  auto relative_path = validate_external_data_relative_path(location, tensor_name);
  auto data_path = base_dir_path / relative_path;
  auto data_path_str = path_to_utf8(data_path);
  // Do not allow symlinks or directories.
  if (data_path.empty() || std::filesystem::is_symlink(data_path)) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid,
        "Data of TensorProto ( tensor name: ",
        tensor_name,
        ") should be stored in ",
        data_path_str,
        ", but it is a symbolic link.");
  }
  // Verify canonical containment (catches parent-dir symlinks).
  if (data_path_str[0] != '#') {
    verify_path_containment(data_path, base_dir, tensor_name);
    // One status call tells a missing file apart from a directory or other file type.
    std::error_code ec;
    const auto data_status = std::filesystem::status(data_path, ec);
    if (!std::filesystem::is_regular_file(data_status)) {
      fail_check_with_code(
          data_status.type() == std::filesystem::file_type::not_found ? error_code::kExternalDataFileMissing
                                                                      : error_code::kExternalDataLocationInvalid,
          "Data of TensorProto ( tensor name: ",
          tensor_name,
          ") should be stored in ",
          data_path_str,
          ", but it is not regular file.");
    }
  }
  // Do not allow hardlinks, as they can be used to read arbitrary files.
  if (data_path_str[0] != '#' && std::filesystem::hard_link_count(data_path) > 1) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid,
        "Data of TensorProto ( tensor name: ",
        tensor_name,
        ") should be stored in ",
        data_path_str,
        ", but it has multiple hard links, indicating a potential hardlink attack.");
  }
  return data_path;
}

struct ExternalDataContext::Impl {
#ifdef _WIN32
  HANDLE base_handle = INVALID_HANDLE_VALUE;
#else
  int base_fd = -1;
#endif
};

#ifdef _WIN32

using NtCreateFileFn = NTSTATUS(NTAPI*)(
    PHANDLE,
    ACCESS_MASK,
    POBJECT_ATTRIBUTES,
    PIO_STATUS_BLOCK,
    PLARGE_INTEGER,
    ULONG,
    ULONG,
    ULONG,
    ULONG,
    PVOID,
    ULONG);
using RtlNtStatusToDosErrorFn = ULONG(NTAPI*)(NTSTATUS);

static NtCreateFileFn get_nt_create_file() {
  static const auto nt_create_file =
      reinterpret_cast<NtCreateFileFn>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtCreateFile"));
  if (nt_create_file == nullptr) {
    fail_check("Secure external-data opening is unavailable on this Windows system.");
  }
  return nt_create_file;
}

static ULONG nt_status_to_dos_error(NTSTATUS status) {
  static const auto convert = reinterpret_cast<RtlNtStatusToDosErrorFn>(
      GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlNtStatusToDosError"));
  return convert == nullptr ? ERROR_GEN_FAILURE : convert(status);
}

static HANDLE open_windows_component(
    HANDLE root,
    const std::filesystem::path& component,
    ACCESS_MASK access,
    ULONG disposition,
    ULONG options,
    ULONG share_access,
    const std::string& tensor_name) {
  const auto& name = component.native();
  if (name.size() > std::numeric_limits<USHORT>::max() / sizeof(wchar_t)) {
    fail_check("External data path component is too long for tensor ", tensor_name, ".");
  }
  UNICODE_STRING unicode_name{};
  unicode_name.Buffer = const_cast<PWSTR>(name.data());
  unicode_name.Length = static_cast<USHORT>(name.size() * sizeof(wchar_t));
  unicode_name.MaximumLength = unicode_name.Length;
  OBJECT_ATTRIBUTES attributes{};
  attributes.Length = sizeof(attributes);
  attributes.RootDirectory = root;
  attributes.ObjectName = &unicode_name;
  attributes.Attributes = OBJ_CASE_INSENSITIVE;
  IO_STATUS_BLOCK io_status{};
  HANDLE handle = INVALID_HANDLE_VALUE;
  const NTSTATUS status = get_nt_create_file()(
      &handle,
      access,
      &attributes,
      &io_status,
      nullptr,
      FILE_ATTRIBUTE_NORMAL,
      share_access,
      disposition,
      options,
      nullptr,
      0);
  if (status < 0 || handle == INVALID_HANDLE_VALUE) {
    const ULONG error = nt_status_to_dos_error(status);
    fail_check_with_code(
        error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND ? error_code::kExternalDataFileMissing
                                                                       : error_code::kExternalDataLocationInvalid,
        "Cannot open external data path component for tensor ",
        tensor_name,
        ".");
  }
  return handle;
}

static BY_HANDLE_FILE_INFORMATION
validate_windows_handle(HANDLE handle, const std::string& tensor_name, bool expect_directory) {
  BY_HANDLE_FILE_INFORMATION info{};
  if (!GetFileInformationByHandle(handle, &info)) {
    fail_check("Tensor ", tensor_name, " external data: cannot query file information.");
  }
  if (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid,
        "Tensor ",
        tensor_name,
        " external data path contains a reparse point.");
  }
  const bool is_directory = (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
  if (is_directory != expect_directory) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid,
        "Tensor ",
        tensor_name,
        " external data path has an unexpected file type.");
  }
  return info;
}

static HANDLE open_windows_leaf(
    HANDLE base_handle,
    const std::vector<std::filesystem::path>& components,
    ACCESS_MASK access,
    ULONG disposition,
    const std::string& tensor_name,
    bool open_reparse_point = true) {
  std::vector<HANDLE> directories;
  ONNX_NAMESPACE::ScopeExit close_directories([&]() noexcept {
    for (HANDLE handle : directories) {
      CloseHandle(handle);
    }
  });
  HANDLE parent = base_handle;
  for (size_t i = 0; i + 1 < components.size(); ++i) {
    HANDLE directory = open_windows_component(
        parent,
        components[i],
        FILE_TRAVERSE | FILE_READ_ATTRIBUTES | SYNCHRONIZE,
        FILE_OPEN,
        FILE_DIRECTORY_FILE | FILE_OPEN_REPARSE_POINT | FILE_SYNCHRONOUS_IO_NONALERT,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        tensor_name);
    ScopedHandle directory_guard(directory);
    validate_windows_handle(directory, tensor_name, true);
    directories.push_back(directory);
    directory_guard.release();
    parent = directory;
  }
  return open_windows_component(
      parent,
      components.back(),
      access,
      disposition,
      FILE_NON_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT | (open_reparse_point ? FILE_OPEN_REPARSE_POINT : 0),
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
      tensor_name);
}

#else // POSIX

static int directory_open_flags() {
  int flags = O_DIRECTORY;
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
#ifdef O_PATH
  flags |= O_PATH;
#elif defined(O_SEARCH)
  flags |= O_SEARCH;
#else
  flags |= O_RDONLY;
#endif
  return flags;
}

static int open_posix_parent(
    int base_fd,
    const std::vector<std::filesystem::path>& components,
    const std::string& tensor_name,
    std::vector<int>& opened_directories) {
  int parent = base_fd;
  for (size_t i = 0; i + 1 < components.size(); ++i) {
    int flags = directory_open_flags();
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#else
    fail_check("Secure external-data opening requires O_NOFOLLOW for tensor ", tensor_name, ".");
#endif
    const int directory = openat(parent, components[i].c_str(), flags);
    if (directory < 0) {
      fail_check_with_code(
          errno == ENOENT ? error_code::kExternalDataFileMissing : error_code::kExternalDataLocationInvalid,
          "Cannot open external data directory component for tensor ",
          tensor_name,
          ".");
    }
    ScopedFd directory_guard(directory);
    opened_directories.push_back(directory);
    directory_guard.release();
    parent = directory;
  }
  return parent;
}

static void validate_posix_file(const struct stat& file_stat, const std::string& tensor_name) {
  if (!S_ISREG(file_stat.st_mode)) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid, "Tensor ", tensor_name, " external data is not a regular file.");
  }
  if (file_stat.st_nlink > 1) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid, "Tensor ", tensor_name, " external data has multiple hard links.");
  }
}

#endif

ExternalDataContext::ExternalDataContext(const std::string& base_dir) : impl_(std::make_unique<Impl>()) {
  const auto normalized_base_dir = base_dir.empty() ? std::string(".") : base_dir;
#ifdef _WIN32
  HANDLE base_handle = CreateFileW(
      utf8_to_path(normalized_base_dir).native().c_str(),
      FILE_TRAVERSE | FILE_READ_ATTRIBUTES | SYNCHRONIZE,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
      nullptr,
      OPEN_EXISTING,
      FILE_FLAG_BACKUP_SEMANTICS,
      nullptr);
  if (base_handle == INVALID_HANDLE_VALUE) {
    fail_check("Cannot open external data base directory.");
  }
  ScopedHandle base_guard(base_handle);
  validate_windows_handle(base_handle, "base directory", true);
  impl_->base_handle = base_guard.release();
#else
  impl_->base_fd = ::open(normalized_base_dir.c_str(), directory_open_flags());
  if (impl_->base_fd < 0) {
    fail_check("Cannot open external data base directory.");
  }
  struct stat base_stat{};
  if (fstat(impl_->base_fd, &base_stat) != 0 || !S_ISDIR(base_stat.st_mode)) {
    close(impl_->base_fd);
    impl_->base_fd = -1;
    fail_check("External data base path is not a directory.");
  }
#endif
}

ExternalDataContext::~ExternalDataContext() {
#ifdef _WIN32
  if (impl_->base_handle != INVALID_HANDLE_VALUE) {
    CloseHandle(impl_->base_handle);
  }
#else
  if (impl_->base_fd >= 0) {
    close(impl_->base_fd);
  }
#endif
}

int64_t ExternalDataContext::open(const std::string& location, const std::string& tensor_name, bool read_only) const {
  ONNX_TRY {
    const auto components = external_data_path_components(location, tensor_name);
#ifdef _WIN32
    HANDLE handle = open_windows_leaf(
        impl_->base_handle,
        components,
        (read_only ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE)) | SYNCHRONIZE,
        read_only ? FILE_OPEN : FILE_OPEN_IF,
        tensor_name);
    ScopedHandle guard(handle);
    const auto info = validate_windows_handle(handle, tensor_name, false);
    if (info.nNumberOfLinks > 1) {
      fail_check_with_code(
          error_code::kExternalDataLocationInvalid, "Tensor ", tensor_name, " external data has multiple hard links.");
    }
    HANDLE released = guard.release();
    const int flags = read_only ? (_O_RDONLY | _O_BINARY) : (_O_RDWR | _O_BINARY);
    const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(released), flags);
    if (fd < 0) {
      CloseHandle(released);
      fail_check("Cannot convert external data handle to fd for tensor ", tensor_name, ".");
    }
    return static_cast<int64_t>(fd);
#else
    std::vector<int> directories;
    ONNX_NAMESPACE::ScopeExit close_directories([&]() noexcept {
      for (int fd : directories) {
        close(fd);
      }
    });
    const int parent = open_posix_parent(impl_->base_fd, components, tensor_name, directories);
    // A FIFO must reach regular-file validation without waiting for a writer.
    int flags = read_only ? (O_RDONLY | O_NONBLOCK) : (O_CREAT | O_RDWR);
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#else
    fail_check("Secure external-data opening requires O_NOFOLLOW for tensor ", tensor_name, ".");
#endif
    const int fd = read_only ? openat(parent, components.back().c_str(), flags)
                             : openat(parent, components.back().c_str(), flags, 0600);
    if (fd < 0) {
      fail_check_with_code(
          read_only && errno == ENOENT ? error_code::kExternalDataFileMissing
                                       : error_code::kExternalDataLocationInvalid,
          "Cannot open external data for tensor ",
          tensor_name,
          ".");
    }
    ScopedFd guard(fd);
    struct stat fd_stat{};
    if (fstat(fd, &fd_stat) != 0) {
      fail_check("Tensor ", tensor_name, " external data: fstat failed.");
    }
    validate_posix_file(fd_stat, tensor_name);
    return guard.release();
#endif
  }
  ONNX_CATCH(const ValidationError& error) {
    ONNX_HANDLE_EXCEPTION([&]() {
      // Write-side errors historically did not expose checker error codes.
      // Keep that API contract while read-side failures retain stable codes.
      if (!read_only && !error.code().empty()) {
        fail_check(error.what());
      }
      throw;
    });
  }
  return -1; // Unreachable: failures throw or abort, and successful opens return.
}

int64_t ExternalDataContext::open_model_file(const std::string& location, bool read_only) const {
  return open_model_file_impl(location, read_only ? ModelFileOpenMode::kRead : ModelFileOpenMode::kWriteOrCreate);
}

int64_t ExternalDataContext::open_existing_model_file_for_write(const std::string& location) const {
  return open_model_file_impl(location, ModelFileOpenMode::kWriteExisting);
}

int64_t ExternalDataContext::create_model_file(const std::string& location) const {
  return open_model_file_impl(location, ModelFileOpenMode::kCreateNew);
}

int64_t ExternalDataContext::open_model_file_impl(const std::string& location, ModelFileOpenMode mode) const {
  const auto components = model_file_path_components(location);
  const bool read_only = mode == ModelFileOpenMode::kRead;
#ifdef _WIN32
  ULONG disposition = FILE_OPEN_IF;
  if (mode == ModelFileOpenMode::kRead || mode == ModelFileOpenMode::kWriteExisting) {
    disposition = FILE_OPEN;
  } else if (mode == ModelFileOpenMode::kCreateNew) {
    disposition = FILE_CREATE;
  }
  HANDLE handle = open_windows_leaf(
      impl_->base_handle,
      components,
      (read_only ? GENERIC_READ : GENERIC_WRITE) | SYNCHRONIZE,
      disposition,
      "model",
      mode == ModelFileOpenMode::kCreateNew);
  ScopedHandle guard(handle);
  validate_windows_handle(handle, "model", false);
  HANDLE released = guard.release();
  const int flags = read_only ? (_O_RDONLY | _O_BINARY) : (_O_WRONLY | _O_BINARY);
  const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(released), flags);
  if (fd < 0) {
    CloseHandle(released);
    fail_check("Cannot convert model file handle to fd.");
  }
  return static_cast<int64_t>(fd);
#else
  std::vector<int> directories;
  ONNX_NAMESPACE::ScopeExit close_directories([&]() noexcept {
    for (int fd : directories) {
      close(fd);
    }
  });
  const int parent = open_posix_parent(impl_->base_fd, components, "model", directories);
  // Neither reading nor writing a FIFO should block before type validation.
  int flags = (read_only ? O_RDONLY : O_WRONLY) | O_NONBLOCK;
  if (mode == ModelFileOpenMode::kWriteOrCreate) {
    flags |= O_CREAT;
  } else if (mode == ModelFileOpenMode::kCreateNew) {
    flags |= O_CREAT | O_EXCL;
  }
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
  const int fd = (flags & O_CREAT) != 0 ? openat(parent, components.back().c_str(), flags, 0666)
                                        : openat(parent, components.back().c_str(), flags);
  if (fd < 0) {
    fail_check_with_code(
        errno == ENOENT ? error_code::kExternalDataFileMissing : error_code::kExternalDataLocationInvalid,
        "Cannot open model file.");
  }
  ScopedFd guard(fd);
  struct stat fd_stat{};
  if (fstat(fd, &fd_stat) != 0 || !S_ISREG(fd_stat.st_mode)) {
    fail_check("Model path is not a regular file.");
  }
  return guard.release();
#endif
}

std::uintmax_t ExternalDataContext::file_size(const std::string& location, const std::string& tensor_name) const {
  const auto components = external_data_path_components(location, tensor_name);
#ifdef _WIN32
  ScopedHandle handle(
      open_windows_leaf(impl_->base_handle, components, FILE_READ_ATTRIBUTES | SYNCHRONIZE, FILE_OPEN, tensor_name));
  const auto info = validate_windows_handle(handle.get(), tensor_name, false);
  if (info.nNumberOfLinks > 1) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid, "Tensor ", tensor_name, " external data has multiple hard links.");
  }
  ULARGE_INTEGER size{};
  size.HighPart = info.nFileSizeHigh;
  size.LowPart = info.nFileSizeLow;
  return static_cast<std::uintmax_t>(size.QuadPart);
#else
  std::vector<int> directories;
  ONNX_NAMESPACE::ScopeExit close_directories([&]() noexcept {
    for (int fd : directories) {
      close(fd);
    }
  });
  const int parent = open_posix_parent(impl_->base_fd, components, tensor_name, directories);
  struct stat file_stat{};
  if (fstatat(parent, components.back().c_str(), &file_stat, AT_SYMLINK_NOFOLLOW) != 0) {
    fail_check_with_code(
        errno == ENOENT ? error_code::kExternalDataFileMissing : error_code::kExternalDataLocationInvalid,
        "Cannot inspect external data for tensor ",
        tensor_name,
        ".");
  }
  validate_posix_file(file_stat, tensor_name);
  if (file_stat.st_size < 0) {
    fail_check("Tensor ", tensor_name, " external data has a negative file size.");
  }
  return static_cast<std::uintmax_t>(file_stat.st_size);
#endif
}

void CheckerContext::set_model_dir(const std::string& model_dir) {
  model_dir_ = model_dir;
  external_data_context_ = std::make_shared<ExternalDataContext>(model_dir);
  // Only a model checked from a file path has a model directory. An in-memory
  // model keeps the default (no directory), so its external data locations are
  // never resolved against the working directory.
  model_dir_from_path_ = true;
}

int64_t open_external_data(
    const std::string& base_dir,
    const std::string& location,
    const std::string& tensor_name,
    bool read_only) {
  return ExternalDataContext(base_dir).open(location, tensor_name, read_only);
}

static const std::unordered_set<std::string> experimental_ops = {
    "ATen",
    "Affine",
    "ConstantFill",
    "Crop",
    "DynamicSlice",
    "GRUUnit",
    "GivenTensorFill",
    "ImageScaler",
    "ParametricSoftplus",
    "Scale",
    "ScaledTanh"};

bool check_is_experimental_op(const NodeProto& node) {
  return (node.domain() == ONNX_DOMAIN || node.domain() == "ai.onnx") && experimental_ops.count(node.op_type());
}

#undef enforce_has_field
#undef enforce_non_empty_field

} // namespace ONNX_NAMESPACE::checker
