#ifndef ONNXIFI_H
#define ONNXIFI_H 1

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(_M_IX86)
/* Windows x86 */
#define ONNXIFI_ABI __stdcall
#elif defined(__i386__)
/* Linux x86 */
#define ONNXIFI_ABI __attribute__((__cdecl__))
#else
#define ONNXIFI_ABI
#endif

#ifndef ONNXIFI_PUBLIC
#if defined(__ELF__)
#define ONNXIFI_PUBLIC __attribute__((__visibility__("default")))
#elif defined(__MACH__)
#define ONNXIFI_PUBLIC __attribute__((__visibility__("default")))
#elif defined(_WIN32) && defined(__GNUC__)
#ifdef ONNXIFI_BUILD_LIBRARY
#define ONNXIFI_PUBLIC __attribute__((__dllexport__))
#else
#define ONNXIFI_PUBLIC __attribute__((__dllimport__))
#endif
#elif defined(_WIN32)
#ifdef ONNXIFI_BUILD_LIBRARY
#define ONNXIFI_PUBLIC __declspec(dllexport)
#else
#define ONNXIFI_PUBLIC __declspec(dllimport)
#endif
#else
#define ONNXIFI_PUBLIC
#endif
#endif

#ifndef ONNXIFI_CHECK_RESULT
  #if defined(__GNUC__) && (__GNUC__ >= 4)
    #define ONNXIFI_CHECK_RESULT __attribute__((__warn_unused_result__))
  #elif defined(_MSC_VER) && (_MSC_VER >= 1700)
    #define ONNXIFI_CHECK_RESULT _Check_return_
  #else
    #define ONNXIFI_CHECK_RESULT
  #endif
#endif

#include <stddef.h>

#if !defined(ONNXIFI_NO_STDINT_H)
#if defined(_MSC_VER) && (_MSC_VER < 1600)
typedef signed __int8 int8_t;
typedef unsigned __int8 uint8_t;
typedef signed __int16 int16_t;
typedef unsigned __int16 uint16_t;
typedef signed __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef signed __int64 int64_t;
typedef unsigned __int64 uint64_t;
#else
#include <stdint.h>
#endif
#endif /* !defined(ONNXIFI_NO_STDINT_H) */

/**
 * Opaque ONNXIFI backend ID.
 *
 * ONNXIFI backend is a combination of software layer and hardware device used
 * to run an ONNX graph. Backend ID uniquely identifies a backend for the life-
 * time of the process (i.e. no two hardware devices, software layers, or
 * combinations of both can have the same backend ID). Backend ID stays valid
 * even if the hardware device used by the backend disconnects from the system.
 */
typedef void* onnxBackendID;
/**
 * Opaque ONNXIFI backend handle.
 * ONNXIFI backend is a combination of software layer and hardware device used
 * to run an ONNX graph.
 */
typedef void* onnxBackend;
/** Opaque ONNXIFI graph handle. */
typedef void* onnxGraph;
/** Opaque ONNXIFI even handle. */
typedef void* onnxEvent;

/** Return code for ONNXIFI functions */
typedef int32_t onnxStatus;
/**
 * Type for enumeration values.
 *
 * The low 32 bits are reserved for standardized ONNXIFI values.
 * The high 32 bits are reserved for vendor-specific extensions. Applications
 * must check for specific vendor extensions before interpreting these bits.
 */
typedef uint64_t onnxEnum;
/**
 * Type for bit fields.
 *
 * The low 32 bits are reserved for standardized ONNXIFI values.
 * The high 32 bits are reserved for vendor-specific extensions. Applications
 * must check for specific vendor extensions before interpreting these bits.
 */
typedef uint64_t onnxBitfield;
/**
 * Type for pointers or handles for memory buffers.
 * This type is intended to work not only for CPU-addressable memory, but also
 * for device memory. uint64_t ensures the API can accomodate Vulkan buffers.
 */
typedef uint64_t onnxPointer;

#define ONNXIFI_STATUS_SUCCESS 0x0000
#define ONNXIFI_STATUS_FALLBACK 0x0001
#define ONNXIFI_STATUS_INVALID_ID 0x0101
#define ONNXIFI_STATUS_INVALID_SIZE 0x0102
#define ONNXIFI_STATUS_INVALID_POINTER 0x0103
#define ONNXIFI_STATUS_INVALID_PROTOBUF 0x0104
#define ONNXIFI_STATUS_INVALID_MODEL 0x0105
#define ONNXIFI_STATUS_INVALID_BACKEND 0x0106
#define ONNXIFI_STATUS_INVALID_GRAPH 0x0107
#define ONNXIFI_STATUS_INVALID_EVENT 0x0108
#define ONNXIFI_STATUS_INVALID_STATE 0x0109
#define ONNXIFI_STATUS_INVALID_NAME 0x010A
#define ONNXIFI_STATUS_INVALID_SHAPE 0x010B
#define ONNXIFI_STATUS_INVALID_DATATYPE 0x010C
#define ONNXIFI_STATUS_INVALID_MEMORY_TYPE 0x010D
#define ONNXIFI_STATUS_INVALID_MEMORY_LOCATION 0x010E
#define ONNXIFI_STATUS_INVALID_FENCE_TYPE 0x010F
#define ONNXIFI_STATUS_INVALID_PROPERTY 0x0110
#define ONNXIFI_STATUS_UNSUPPORTED_TAG 0x0201
#define ONNXIFI_STATUS_UNSUPPORTED_VERSION 0x0202
#define ONNXIFI_STATUS_UNSUPPORTED_OPERATOR 0x0203
#define ONNXIFI_STATUS_UNSUPPORTED_ATTRIBUTE 0x0204
#define ONNXIFI_STATUS_UNSUPPORTED_SHAPE 0x0205
#define ONNXIFI_STATUS_UNSUPPORTED_DATATYPE 0x0206
#define ONNXIFI_STATUS_UNSUPPORTED_MEMORY_TYPE 0x0207
#define ONNXIFI_STATUS_UNSUPPORTED_FENCE_TYPE 0x0208
#define ONNXIFI_STATUS_UNSUPPORTED_PROPERTY 0x0209
#define ONNXIFI_STATUS_UNIDENTIFIED_NAME 0x0301
#define ONNXIFI_STATUS_MISMATCHING_SHAPE 0x0302
#define ONNXIFI_STATUS_MISMATCHING_DATATYPE 0x0303
#define ONNXIFI_STATUS_NO_SYSTEM_MEMORY 0x0401
#define ONNXIFI_STATUS_NO_DEVICE_MEMORY 0x0402
#define ONNXIFI_STATUS_NO_SYSTEM_RESOURCES 0x0403
#define ONNXIFI_STATUS_NO_DEVICE_RESOURCES 0x0404
#define ONNXIFI_STATUS_BACKEND_UNAVAILABLE 0x0405
#define ONNXIFI_STATUS_INTERNAL_ERROR 0x0406

/**
 * State of an ONNXIFI event object.
 *
 * Possible values:
 *     ONNXIFI_EVENT_STATE_INVALID
 *     ONNXIFI_EVENT_STATE_NONSIGNALLED
 *     ONNXIFI_EVENT_STATE_SIGNALLED
 */
typedef int32_t onnxEventState;

/**
 * State for an invalid onnxEvent.
 */
#define ONNXIFI_EVENT_STATE_INVALID 0
/**
 * Non-signalled onnxEvent state.
 * onnxInitEvent creates events in non-signalled state.
 */
#define ONNXIFI_EVENT_STATE_NONSIGNALLED 0x16BD
/**
 * Signalled onnxEvent state.
 * onnxSignalEvent changes event state to signalled.
 */
#define ONNXIFI_EVENT_STATE_SIGNALLED 0x3395

/** Special-purpose accelerator for neural network */
#define ONNXIFI_DEVICE_TYPE_NPU 0x01
/** Digital signal processor */
#define ONNXIFI_DEVICE_TYPE_DSP 0x02
/** Graphics accelerator */
#define ONNXIFI_DEVICE_TYPE_GPU 0x04
/** General-purpose central processor */
#define ONNXIFI_DEVICE_TYPE_CPU 0x08
/** Field-programmable gate array */
#define ONNXIFI_DEVICE_TYPE_FPGA 0x10
/**
 * Heterogeneous backend whichinternally arbitrates or distributes work between
 * multiple device types.
 */
#define ONNXIFI_DEVICE_TYPE_HETEROGENEOUS 0x20

/**
 * The backend supports multi-threaded access to ONNXIFI backend, graph, and
 * event objects. E.g. onnxInitGraph can be called on a different thread than
 * onnxInitBackend.
 *
 * If this capability it not indicated, ONNXIFI backend, graph, and event
 * objects that relate to the backend must always be used on the same thread
 * where the backend object was initialized.
 */
#define ONNXIFI_CAPABILITY_THREAD_SAFE 0x01
/**
 * The backend supports ONNX graphs with symbolic variables in the outer
 * shape dimension (batch size), using TensorShapeProto.dim_param for
 * ModelProto.graph.input.type.shape or ModelProto.graph.output.type.shape.
 *
 * The exact numerical value of the  of all input and output tensors must be specified
 * in the onnxSetGraphIO call(s).
 */
#define ONNXIFI_CAPABILITY_SYMBOLIC_BATCH_SIZE 0x02
/**
 * The backend supports ONNX graphs with symbolic variables in the all
 * shape dimensions, using TensorShapeProto.dim_param for
 * ModelProto.graph.input.type.shape or ModelProto.graph.output.type.shape.
 *
 * Backends with this capability also MUST support
 * ONNXIFI_CAPABILITY_SYMBOLIC_BATCH_SIZE capability.
 *
 * The exact numerical shape of all input and output tensors must be specified
 * in the onnxSetGraphIO call(s).
 */
#define ONNXIFI_CAPABILITY_SYMBOLIC_SIZE_TENSORS 0x04
/**
 * The backend supports ONNX graphs with data-dependent outer shape dimension
 * (batch size) of graph outputs. The ONNX graph would specify unknown outer
 * shape dimension (batch size) using symbolic variables, so this capability
 * requires ONNXIFI_CAPABILITY_SYMBOLIC_BATCH_SIZE support.
 *
 * For outputs with data-dependent outer shape dimension (batch size) the value
 * specified in onnxSetGraphIO call is interpreted as the upper limit. The exact
 * numerical batch size of the output can be retrieved by attaching a Shape
 * operator to the tensor with data-dependent shape and reading its output
 * through ONNXIFI.
 */
#define ONNXIFI_CAPABILITY_VARIABLE_BATCH_SIZE 0x08
/**
 * The backend supports ONNX graphs with data-dependent output shapes.
 * The ONNX graph would specify unknown output shapes using symbolic variables,
 * so this capability requires ONNXIFI_CAPABILITY_SYMBOLIC_SIZE_TENSORS support.
 *
 * Backends with this capability also MUST support
 * ONNXIFI_CAPABILITY_VARIABLE_BATCH_SIZE capability.
 *
 * For outputs with data-dependent shapes the shape specified in onnxSetGraphIO
 * call is interpreted as the upper limit. The exact numerical shape of the
 * output can be retrieved by attaching a Shape operator to the tensor with
 * data-dependent shape and reading its output through ONNXIFI.
 */
#define ONNXIFI_CAPABILITY_VARIABLE_SIZE_OUTPUTS 0x10
/**
 * The backend uses a hot-pluggable device, and can be disconnected at any time.
 *
 * If the underlying device disconnects from the system, subsequent operations
 * with the backend, or objects created on the backend, will fail with
 * ONNXIFI_STATUS_BACKEND_UNAVAILABLE status code.
 */
#define ONNXIFI_CAPABILITY_HOT_PLUGGABLE 0x20

/**
 * Type of the backend information.
 *
 * Possible values:
 *     ONNXIFI_BACKEND_ONNXIFI_VERSION
 *     ONNXIFI_BACKEND_NAME
 *     ONNXIFI_BACKEND_VENDOR
 *     ONNXIFI_BACKEND_VERSION
 *     ONNXIFI_BACKEND_EXTENSIONS
 *     ONNXIFI_BACKEND_DEVICE
 *     ONNXIFI_BACKEND_DEVICE_TYPE
 *     ONNXIFI_BACKEND_ONNX_IR_VERSION
 *     ONNXIFI_BACKEND_OPSET_VERSION
 *     ONNXIFI_BACKEND_CAPABILITIES
 *     ONNXIFI_BACKEND_INIT_PROPERTIES
 *     ONNXIFI_BACKEND_MEMORY_TYPES
 *     ONNXIFI_BACKEND_GRAPH_INIT_PROPERTIES
 *     ONNXIFI_BACKEND_SYNCHRONIZATION_TYPES
 *     ONNXIFI_BACKEND_MEMORY_SIZE
 *     ONNXIFI_BACKEND_MAX_GRAPH_SIZE
 *     ONNXIFI_BACKEND_MAX_GRAPH_COUNT
 *     ONNXIFI_BACKEND_MACS_FP32
 *     ONNXIFI_BACKEND_MACS_FP16
 *     ONNXIFI_BACKEND_MEMORY_BANDWIDTH
 *     ONNXIFI_BACKEND_CPU_MEMORY_READ_BANDWIDTH
 *     ONNXIFI_BACKEND_CPU_MEMORY_WRITE_BANDWIDTH
 *     ONNXIFI_BACKEND_PCI_BUS_ID
 *     ONNXIFI_BACKEND_PCI_DEVICE_ID
 *     ONNXIFI_BACKEND_PCI_DOMAIN_ID
 *     ONNXIFI_BACKEND_DIRECTX_ID
 *     ONNXIFI_BACKEND_CUDA_INDEX
 *     ONNXIFI_BACKEND_OPENCL_PLATFORM_ID
 *     ONNXIFI_BACKEND_OPENCL_DEVICE_ID
 */
typedef int32_t onnxBackendInfo;

/**
 * Major and minor version of ONNXIFI specification implemented by the backend.
 *
 * Since ONNXIFI 1.0, backends MUST support this information query.
 *
 * Value type: uint64_t.
 *      The high 32 bits specify the major version.
 *      The low 32 bits specify the minor version.
 *
 * Possible values:
 *      UINT64_C(0x0000000100000000) for ONNXIFI 1.0
 */
#define ONNXIFI_BACKEND_ONNXIFI_VERSION 0

/**
 * Marketing name of the backend (excluding the vendor name).
 *
 * Since ONNXIFI 1.0, backends MUST support this information query.
 *
 * This string MUST be in UTF-8 encoding and NOT locale-sensitive.
 *
 * Value type: char[], e.g.:
 *    "Caffe2"
 *    "Glow"
 */
#define ONNXIFI_BACKEND_NAME 1

/**
 * Name of the backend vendor.
 *
 * Since ONNXIFI 1.0, backends MUST support this information query.
 *
 * This string MUST be in UTF-8 encoding and NOT locale-sensitive.
 *
 * Value type: char[], e.g.:
 *    "Facebook"
 *    "Marat Dukhan"
 */
#define ONNXIFI_BACKEND_VENDOR 2

/**
 * Version of the backend software. Exact format is vendor-specific, but MUST be
 * unique for the software release.
 *
 * Since ONNXIFI 1.0, backends MUST support this information query.
 *
 * This string MUST be in US-ASCII encoding and NOT locale-sensitive.
 *
 * Value type: char[], e.g.:
 *    "1.2.3"
 *    "1.2.3.0"
 *    "1.2.3-db3a4439d233276e25681fb4735b7f8e674dda65"
 */
#define ONNXIFI_BACKEND_VERSION 3

/**
 * Space-separated list of vendor- or device-specific extensions supported on
 * this backend.
 *
 * Since ONNXIFI 1.0, backends MUST support this information query.
 *
 * This string MUST be in US-ASCII encoding and NOT locale-sensitive.
 *
 * Value type: char[], e.g.:
 *    ""
 *    "onnx_clone_graph"
 *    "onnx_clone_graph fb_maskrcnn"
 */
#define ONNXIFI_BACKEND_EXTENSIONS 4

/**
 * Descriptive name of the device (i.e. CPU, GPU, DSP, or NPU model).
 *
 * Since ONNXIFI 1.0, backends MUST support this information query.
 *
 * This string MUST be in UTF-8 encoding and NOT locale-sensitive.
 *
 * Value type: char[], e.g.:
 *    "nnDuino 123"
 */
#define ONNXIFI_BACKEND_DEVICE 5

/**
 * Type of the device.
 *
 * Since ONNXIFI 1.0, backends MUST support this information query.
 *
 * Value type: onnxEnum.
 * Possible values:
 *      ONNXIFI_DEVICE_TYPE_NPU
 *      ONNXIFI_DEVICE_TYPE_DSP
 *      ONNXIFI_DEVICE_TYPE_GPU
 *      ONNXIFI_DEVICE_TYPE_CPU
 *      ONNXIFI_DEVICE_TYPE_FPGA
 *      ONNXIFI_DEVICE_TYPE_HETEROGENEOUS
 */
#define ONNXIFI_BACKEND_DEVICE_TYPE 6

/**
 * List of supported ONNX IR versions.
 *
 * Since ONNXIFI 1.0, backends MUST support this information query.
 *
 * Value type: char[], e.g.:
 *    "3" (IR version in ONNX 1.0)
 *
 * Possible values: space-separated list of supported ONNX IR versions,
 *     represented as decimal integers. ONNX IR versions must match values
 *     in ONNX Version enum.
 */
#define ONNXIFI_BACKEND_ONNX_IR_VERSION 7

/**
 * List of supported operator set domains and maximum supported operator set
 * version for each domain.
 *
 * Since ONNXIFI 1.0, backends MUST support this information query.
 *
 * Value type: char[], e.g.:
 *    "ai.onnx:1" (only operators in version 1 of default ONNX operator set)
 *    "ai.onnx:7" (operators up to version 7 of default ONNX operator set)
 *    "org.pytorch:2 ai.onnx:6 ai.facebook:1"
 *
 * Possible values: space-separated list of domain:max_version pairs where
 *     domain corresponds to OperatorSetIdProto.domain and max_version
 *     corresponds to the maximum value of OperatorSetIdProto.version supported
 *     by the backend for this domain. The backend MUST support all previous
 *     operator set versions as well.
 */
#define ONNXIFI_BACKEND_OPSET_VERSION 8

/**
 * Optional features supported by the backend.
 *
 * Since ONNXIFI 1.0, backends MUST support this information query.
 *
 * Value type: onnxBitfield.
 * Possible values: any combination of the following flags:
 *      ONNXIFI_CAPABILITY_THREAD_SAFE
 *      ONNXIFI_CAPABILITY_SYMBOLIC_BATCH_SIZE
 *      ONNXIFI_CAPABILITY_SYMBOLIC_SIZE_TENSORS
 *      ONNXIFI_CAPABILITY_VARIABLE_BATCH_SIZE
 *      ONNXIFI_CAPABILITY_VARIABLE_SIZE_OUTPUTS
 *      ONNXIFI_CAPABILITY_HOT_PLUGGABLE
 *      or any vendor-specific flags in the high 32 bits of the bit field.
 */
#define ONNXIFI_BACKEND_CAPABILITIES 10

/**
 * Auxiliary initialization properties supported by the backend.
 *
 * Since ONNXIFI 1.0, backends MUST support this information query.
 *
 * Value type: onnxBitfield.
 * Possible values: any combination of vendor-specific flags in high 32 bits of
 * the bit field.
 */
#define ONNXIFI_BACKEND_INIT_PROPERTIES 11

/**
 * Memory types supported for graph inputs and outputs.
 *
 * Since ONNXIFI 1.0, backends MUST support this information query.
 *
 * Value type: onnxBitfield.
 * Possible values are any combination of the following flags:
 *     ONNXIFI_MEMORY_TYPE_CPU (always supported)
 *     ONNXIFI_MEMORY_TYPE_CUDA_BUFFER
 *     ONNXIFI_MEMORY_TYPE_OPENCL_BUFFER
 *     ONNXIFI_MEMORY_TYPE_OPENGLES_TEXTURE_2D
 *     ONNXIFI_MEMORY_TYPE_D3D_RESOURCE
 *     or any vendor-specific flags in the high 32 bits of the bit field.
 */
#define ONNXIFI_BACKEND_MEMORY_TYPES 12

/**
 * Auxiliary initialization properties supported by graphs on the backend.
 *
 * Since ONNXIFI 1.0, backends MUST support this information query.
 *
 * Value type: onnxBitfield.
 * Possible values: any combination of vendor-specific flags in high 32 bits of
 * the bit field.
 */
#define ONNXIFI_BACKEND_GRAPH_INIT_PROPERTIES 13

/**
 * Memory synchronization primitives supported for graph inputs and outputs.
 *
 * Since ONNXIFI 1.0, backends MUST support this information query.
 *
 * Possible values are any combination of the following flags:
 *     ONNXIFI_SYNCHRONIZATION_EVENT    (onnxEvent, always supported)
 *     ONNXIFI_SYNCHRONIZATION_IMPLICIT
 *     or any vendor-specific flags in the high 32 bits of the bit field.
 */
#define ONNXIFI_BACKEND_SYNCHRONIZATION_TYPES 14

/**
 * Maximum amount of memory, in bytes, available to the use by the backend.
 *
 * Since ONNXIFI 1.0, backends MUST support this information query.
 *
 * Value type: uint64_t.
 */
#define ONNXIFI_BACKEND_MEMORY_SIZE 20

/**
 * Maximum size of network parameters, in bytes.
 *
 * Since ONNXIFI 1.0, backends MUST support this information query.
 *
 * Value type: uint64_t.
 */
#define ONNXIFI_BACKEND_MAX_GRAPH_SIZE 21

/**
 * Maximum number of independent network graphs supported by the backend.
 *
 * Since ONNXIFI 1.0, backends MUST support this information query.
 *
 * Value type: uint64_t.
 */
#define ONNXIFI_BACKEND_MAX_GRAPH_COUNT 22

/**
 * Number of FP32 multiply-accumulate operations per second delivered by the
 * backend.
 *
 * Since ONNXIFI 1.0, backends are recommended, but not required to support this
 * information query.
 *
 * Value type: uint64_t.
 * If the backend does not support FP32 computation, the value MUST be 0.
 */
#define ONNXIFI_BACKEND_MACS_FP32 30

/**
 * Number of FP16 multiply-accumulate operations per second delivered by the
 * backend.
 *
 * Since ONNXIFI 1.0, backends are recommended, but not required to support this
 * information query.
 *
 * Value type: uint64_t.
 * If the backend does not support FP16 computation, the value MUST be 0.
 */
#define ONNXIFI_BACKEND_MACS_FP16 31

/**
 * Bandwidth, in bytes per second, of the global memory specific to the backend
 * device.
 *
 * Since ONNXIFI 1.0, backends are recommended, but not required to support this
 * information query.
 *
 * Value type: uint64_t.
 */
#define ONNXIFI_BACKEND_MEMORY_BANDWIDTH 35

/**
 * Bandwidth, in bytes per second, of transferring data from cacheable
 * CPU-allocated memory to the backend device.
 *
 * Since ONNXIFI 1.0, backends are recommended, but not required to support this
 * information query.
 *
 * Value type: uint64_t.
 */
#define ONNXIFI_BACKEND_CPU_MEMORY_READ_BANDWIDTH 36

/**
 * Bandwidth, in bytes per second, of transferring data to cacheable
 * CPU-allocated memory from the backend device.
 *
 * Since ONNXIFI 1.0, backends are recommended, but not required to support this
 * information query.
 *
 * Value type: uint64_t.
 */
#define ONNXIFI_BACKEND_CPU_MEMORY_WRITE_BANDWIDTH 37

/**
 * PCI bus ID of the backend device.
 *
 * Since ONNXIFI 1.0, backends are recommended, but not required to support this
 * information query.
 *
 * Value type: uint64_t.
 */
#define ONNXIFI_BACKEND_PCI_BUS_ID 40

/**
 * PCI device ID of the backend device.
 *
 * Since ONNXIFI 1.0, backends are recommended, but not required to support this
 * information query.
 *
 * Value type: uint64_t.
 */
#define ONNXIFI_BACKEND_PCI_DEVICE_ID 41

/**
 * PCI domain/function ID of the backend device.
 *
 * Since ONNXIFI 1.0, backends are recommended, but not required to support this
 * information query.
 *
 * Value type: uint64_t.
 */
#define ONNXIFI_BACKEND_PCI_DOMAIN_ID 42

/**
 * DirectX ID of the backend device.
 *
 * This is the value that would be returned by ID3D12Device::GetAdapterLuid()
 * for the hardware device used by the backend.
 *
 * Since ONNXIFI 1.0, DXGI-based backends are recommended, but not required to
 * support this information query.
 *
 * Value type: LUID (8 bytes).
 */
#define ONNXIFI_BACKEND_DIRECTX_ID 43

/**
 * CUDA index of the backend device.
 *
 * Since ONNXIFI 1.0, CUDA-based backends are recommended, but not required to
 * support this information query.
 *
 * Value type: uint64_t.
 */
#define ONNXIFI_BACKEND_CUDA_INDEX 44

/**
 * OpenCL platform ID for the backend device.
 * This platform ID is guaranteed to remain valid for the lifetime of ONNXIFI
 * objects related to the same ONNXIFI backend (backend ID, backend, graph,
 * event).
 *
 * Since ONNXIFI 1.0, OpenCL-based backends are recommended, but not required to
 * support this information query.
 *
 * Value type: cl_platform_id.
 */
#define ONNXIFI_BACKEND_OPENCL_PLATFORM_ID 45

/**
 * OpenCL device ID for the backend device.
 * This device ID is guaranteed to remain valid for the lifetime of ONNXIFI
 * objects related to the same ONNXIFI backend (backend ID, backend, graph,
 * event).
 *
 * Since ONNXIFI 1.0, OpenCL-based backends are recommended, but not required to
 * support this information query.
 *
 * Value type: cl_device_id.
 */
#define ONNXIFI_BACKEND_OPENCL_DEVICE_ID 46

/* Note: the data type values match ONNX TensorProto.DataType enum */
#define ONNXIFI_DATATYPE_UNDEFINED 0
#define ONNXIFI_DATATYPE_FLOAT16 10
#define ONNXIFI_DATATYPE_FLOAT32 1
#define ONNXIFI_DATATYPE_FLOAT64 11
#define ONNXIFI_DATATYPE_INT8 3
#define ONNXIFI_DATATYPE_INT16 5
#define ONNXIFI_DATATYPE_INT32 6
#define ONNXIFI_DATATYPE_INT64 7
#define ONNXIFI_DATATYPE_UINT8 2
#define ONNXIFI_DATATYPE_UINT16 4
#define ONNXIFI_DATATYPE_UINT32 12
#define ONNXIFI_DATATYPE_UINT64 13
#define ONNXIFI_DATATYPE_COMPLEX64 14
#define ONNXIFI_DATATYPE_COMPLEX128 15
#define ONNXIFI_DATATYPE_BFLOAT16 16

/** Cacheable CPU memory */
#define ONNXIFI_MEMORY_TYPE_CPU 0
/** CUDA memory buffer (allocated via cudaMalloc/cuMalloc).  */
#define ONNXIFI_MEMORY_TYPE_CUDA_BUFFER 1
/** OpenCL cl_mem object for a buffer or sub-buffer. */
#define ONNXIFI_MEMORY_TYPE_OPENCL_BUFFER 2
/** OpenGL ES 2.0+ 2D Texture. */
#define ONNXIFI_MEMORY_TYPE_OPENGLES_TEXTURE_2D 4
/** Direct3D resource. */
#define ONNXIFI_MEMORY_TYPE_D3D_RESOURCE 8

/**
 * Terminates the list of auxiliary backend initialization properties passed to
 * onnxInitBackend.
 */
#define ONNXIFI_BACKEND_PROPERTY_NONE 0
/**
 * Optimization target for graphs initialized on the backend.
 *
 * Possible values:
 *     ONNXIFI_OPTIMIZATION_HIGH_THROUGHPUT
 *     ONNXIFI_OPTIMIZATION_LOW_LATENCY
 *     ONNXIFI_OPTIMIZATION_LOW_POWER
 *     ONNXIFI_OPTIMIZATION_LOW_DELAY
 */
#define ONNXIFI_BACKEND_PROPERTY_OPTIMIZATION 1
/**
 * Logging verbosity level for the backend.
 *
 * If this property is not specified during initialization, the backend should
 * assume ONNXIFI_LOG_LEVEL_WARNING logging verbosity level.
 *
 * Possible values:
 *     ONNXIFI_LOG_LEVEL_ERROR
 *     ONNXIFI_LOG_LEVEL_WARNING
 *     ONNXIFI_LOG_LEVEL_INFO
 *     ONNXIFI_LOG_LEVEL_DEBUG
 */
#define ONNXIFI_BACKEND_PROPERTY_LOG_LEVEL 2
/**
 * CUDA stream to be used by the backend.
 * CUDA stream must be created on the CUDA device used by the ONNXIFI backend.
 * Users can query which CUDA device is used by the ONNXIFI backend by calling
 * onnxGetBackendInfo with ONNXIFI_BACKEND_CUDA_INDEX info type.
 *
 * If this property is not specified during initialization, the backend can
 * create a new CUDA stream for the device, or use a default CUDA stream.
 *
 * Possible values: cudaStream_t or CUstream object, cast to uint64_t.
 */
#define ONNXIFI_BACKEND_CUDA_STREAM 4
/**
 * OpenCL context to be used by the backend.
 * The context must be created with the OpenCL device ID and the OpenCL platform
 * ID used by the ONNXIFI backend. Users can query which OpenCL device ID and
 * OpenCL platform ID are used by the ONNXIFI backend by calling
 * onnxGetBackendInfo with ONNXIFI_BACKEND_OPENCL_PLATFORM_ID
 * and ONNXIFI_BACKEND_OPENCL_DEVICE_ID info types.
 *
 * If this property is not specified during initialization, the backend will
 * create a new OpenCL context for the device.
 *
 * Possible values: cl_context object, cast to uint64_t.
 */
#define ONNXIFI_BACKEND_OPENCL_CONTEXT 8

/**
 * Terminates the list of auxiliary graph initialization properties passed to
 * onnxInitGraph.
 */
#define ONNXIFI_GRAPH_PROPERTY_NONE 0

/**
 * Optimize graph representation and compilation for highest throughput.
 */
#define ONNXIFI_OPTIMIZATION_HIGH_THROUGHPUT 0

/**
 * Optimize graph representation and compilation for lowest latency.
 */
#define ONNXIFI_OPTIMIZATION_LOW_LATENCY 1

/**
 * Optimize graph representation and compilation for lowest power consumption.
 */
#define ONNXIFI_OPTIMIZATION_LOW_POWER 2

/**
 * Optimize graph representation and compilation for lowest delay until first
 * result.
 */
#define ONNXIFI_OPTIMIZATION_LOW_DELAY 3

/**
 * Log events which caused a failure in an ONNXIFI function call.
 */
#define ONNXIFI_LOG_LEVEL_ERROR 4

/**
 * Log events in ONNXIFI_LOG_LEVEL_ERROR and events which caused
 * a performance, accuracy, or quality of service degradation in a backend.
 * Enabling this logging level SHOULD NOT have a measurable effect on
 * performance.
 */
#define ONNXIFI_LOG_LEVEL_WARNING 3

/**
 * Log events in ONNXIFI_LOG_LEVEL_WARNING and high-level status information
 * about operation of a backend. Enabling this logging level MAY cause a small
 * degradation in performance.
 */
#define ONNXIFI_LOG_LEVEL_INFO 2

/**
 * Log events in ONNXIFI_LOG_LEVEL_INFO and detailed status information about
 * operations of a backend. Enabling this logging level MAY cause a serious
 * degradation in performance.
 */
#define ONNXIFI_LOG_LEVEL_DEBUG 1

/**
 * Tag for version 1 of tensor descriptor structure (onnxTensorDescriptorV1).
 *
 * The tag is unique for this version. If ONNXIFI introduce a new version of
 * the tensor descriptor structure in the future, it will get a new tag value.
 */
#define ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1 0x43DFBF69

typedef struct onnxTensorDescriptorV1 {
  /**
   * 32-bit tag needed to distinguish different versions of a tensor descriptor
   * structure. In the onnxTensorDescriptorV1 structure, the tag MUST be set to
   * ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1. If ONNXIFI introduce a new version of the
   * tensor descriptor structure in the future, it WILL have 32-bit tag with a
   * different value as the first member of the structure.
   *
   * ONNXIFI implementations MUST validate tag before accessing any other
   * members of the structure.
   */
  int32_t tag;
  /**
   * Name of the blob corresponding to this tensor in the ONNX model. The name
   * must exactly match the ValueInfoProto.name of one of the
   * ModelProto.graph.input or ModelProto.graph.output
   */
  const char* name;
  /**
   * Data type of the elements in the tensor.
   *
   * Possible values:
   *     ONNXIFI_DATATYPE_FLOAT16
   *     ONNXIFI_DATATYPE_FLOAT32
   *     ONNXIFI_DATATYPE_FLOAT64
   *     ONNXIFI_DATATYPE_INT8
   *     ONNXIFI_DATATYPE_INT16
   *     ONNXIFI_DATATYPE_INT32
   *     ONNXIFI_DATATYPE_INT64
   *     ONNXIFI_DATATYPE_UINT8
   *     ONNXIFI_DATATYPE_UINT16
   *     ONNXIFI_DATATYPE_UINT32
   *     ONNXIFI_DATATYPE_UINT64
   *     ONNXIFI_DATATYPE_COMPLEX64
   *     ONNXIFI_DATATYPE_COMPLEX128
   */
  onnxEnum dataType;
  /**
   * Type of memory that stores the tensor.
   *
   * ONNXIFI_MEMORY_TYPE_CPU memory type is always supported by the backend, but
   * other memory types are optional. The use MUST call onnxGetBackendInfo with
   * ONNXIFI_BACKEND_MEMORY_TYPES to check if a particular memory type is
   * supported before using it.
   *
   * If the memory type is different than ONNXIFI_MEMORY_TYPE_CPU, it must be
   * allocated on the same device as the backend.
   *
   * Possible values:
   *     ONNXIFI_MEMORY_TYPE_CPU                 (always supported)
   *     ONNXIFI_MEMORY_TYPE_CUDA_BUFFER         (support is optional)
   *     ONNXIFI_MEMORY_TYPE_OPENCL_BUFFER       (support is optional)
   *     ONNXIFI_MEMORY_TYPE_OPENGLES_TEXTURE_2D (support is optional)
   *     ONNXIFI_MEMORY_TYPE_D3D_RESOURCE        (support is optional)
   */
  onnxEnum memoryType;
  /**
   * Number of dimensions in the tensor.
   * For a scalar, the number of dimensions is 0.
   */
  uint32_t dimensions;
  /**
   * Dimensions of the tensor.
   * For a scalar, this pointer can be NULL.
   */
  const uint64_t* shape;
  /**
   * Pointers to tensor data.
   *
   * Interpretation depends on memoryType:
   *   - ONNXIFI_MEMORY_TYPE_CPU: buffer is a valid pointer to CPU memory.
   *   - ONNXIFI_MEMORY_TYPE_CUDA_BUFFER: buffer is a valid pointer to CUDA
   *     device memory, allocated via cudaMalloc or cuMalloc. CUDA device memory
   *     must be allocated on the same device as the backend.
   *   - ONNXIFI_MEMORY_TYPE_OPENCL_BUFFER: buffer is a cl_mem handle for an
   *     OpenCL buffer or a sub-buffer. cl_mem object must be allocated on the
   *     same OpenCL context as the backend. The context must be specified via
   *     ONNXIFI_BACKEND_OPENCL_CONTEXT initialization property to
   *     onnxInitBackend.
   *   - ONNXIFI_MEMORY_TYPE_OPENGLES_TEXTURE_2D: buffer is a name of a 2D
   *     texture created by glGenTextures. The texture must be allocated on the
   *     same device as the backend.
   *   - ONNXIFI_MEMORY_TYPE_D3D_RESOURCE: TBD
   */
  onnxPointer buffer;
} onnxTensorDescriptorV1;

/**
 * Synchronization using ONNXIFI event object (onnxEvent).
 */
#define ONNXIFI_SYNCHRONIZATION_EVENT 0
/**
 * Implicit synchronization of inputs and outputs access with the caller.
 * The details are backend-specific, and may involve extra parameters passed
 * during backend initialization.
 *
 * Examples:
 *  - CUDA-based backends could implicitly synchronize with the caller through
 *    the use of the same CUDA stream.
 *  - OpenCL-based backends could implicitly synchronize with the caller through
 *    the use of the same in-order OpenCL command queue.
 */
#define ONNXIFI_SYNCHRONIZATION_IMPLICIT 2

/**
 * Tag for version 1 of memory fence structure (onnxMemoryFenceV1).
 *
 * The tag is unique for this version. If ONNXIFI introduce a new version of
 * the memory fence structure in the future, it will get a new tag value.
 */
#define ONNXIFI_TAG_MEMORY_FENCE_V1 0x23E08AAB

typedef struct onnxMemoryFenceV1 {
  /**
   * 32-bit tag needed to distinguish different versions of a memory fence
   * structure. In the onnxMemoryFenceV1 structure, the tag MUST be set to
   * ONNXIFI_TAG_MEMORY_FENCE_V1. If ONNXIFI introduce a new version of the
   * memory fence structure in the future, it WILL have 32-bit tag with a
   * different value as the first member of the structure.
   *
   * ONNXIFI implementations MUST validate tag before accessing any other
   * members of the structure.
   */
  int32_t tag;
  /**
   * Type of memory synchronization primitive.
   *
   * Possible values:
   *      ONNXIFI_SYNCHRONIZATION_EVENT    (onnxEvent, always supported)
   *      ONNXIFI_SYNCHRONIZATION_IMPLICIT
   */
  onnxEnum type;
  union {
    /**
     * Handle for a single-shot ONNXIFI event used as a synchronization
     * primitive. Event for the input fence must be created by the caller to
     * onnxRunGraph. Event for the output fence is created by implementation of
     * onnxRunGraph, and stored into the output memory fence structure before
     * onnxRunGraph returns.
     */
    onnxEvent event;
  };
} onnxMemoryFenceV1;

/* Function pointer declarations for dynamic loading */
typedef ONNXIFI_CHECK_RESULT onnxStatus
  (ONNXIFI_ABI* onnxGetBackendIDsFunction)(
    onnxBackendID* backendIDs,
    size_t* numBackends);
typedef onnxStatus
  (ONNXIFI_ABI* onnxReleaseBackendIDFunction)(
    onnxBackendID backendID);
typedef ONNXIFI_CHECK_RESULT onnxStatus
  (ONNXIFI_ABI* onnxGetBackendInfoFunction)(
    onnxBackendID backendID,
    onnxBackendInfo infoType,
    void* infoValue,
    size_t* infoValueSize);
typedef ONNXIFI_CHECK_RESULT onnxStatus
  (ONNXIFI_ABI* onnxGetBackendCompatibilityFunction)(
    onnxBackendID backendID,
    size_t onnxModelSize,
    const void* onnxModel);
typedef ONNXIFI_CHECK_RESULT onnxStatus
  (ONNXIFI_ABI* onnxInitBackendFunction)(
    onnxBackendID backendID,
    const uint64_t* auxPropertiesList,
    onnxBackend* backend);
typedef onnxStatus
  (ONNXIFI_ABI* onnxReleaseBackendFunction)(
    onnxBackend backend);
typedef ONNXIFI_CHECK_RESULT onnxStatus
  (ONNXIFI_ABI* onnxInitEventFunction)(
    onnxBackend backend,
    onnxEvent* event);
typedef ONNXIFI_CHECK_RESULT onnxStatus
  (ONNXIFI_ABI* onnxSignalEventFunction)(
    onnxEvent event);
typedef ONNXIFI_CHECK_RESULT onnxStatus
  (ONNXIFI_ABI* onnxGetEventStateFunction)(
    onnxEvent event,
    onnxEventState* state);
typedef ONNXIFI_CHECK_RESULT onnxStatus
  (ONNXIFI_ABI* onnxWaitEventFunction)(
    onnxEvent event);
typedef onnxStatus
  (ONNXIFI_ABI* onnxReleaseEventFunction)(
    onnxEvent event);
typedef ONNXIFI_CHECK_RESULT onnxStatus
  (ONNXIFI_ABI* onnxInitGraphFunction)(
    onnxBackend backend,
    const uint64_t* auxPropertiesList,
    size_t onnxModelSize,
    const void* onnxModel,
    uint32_t weightsCount,
    const onnxTensorDescriptorV1* weightDescriptors,
    onnxGraph* graph);
typedef ONNXIFI_CHECK_RESULT onnxStatus
  (ONNXIFI_ABI* onnxSetGraphIOFunction)(
    onnxGraph graph,
    uint32_t inputsCount,
    const onnxTensorDescriptorV1* inputDescriptors,
    uint32_t outputsCount,
    const onnxTensorDescriptorV1* outputDescriptors);
typedef ONNXIFI_CHECK_RESULT onnxStatus
  (ONNXIFI_ABI* onnxRunGraphFunction)(
    onnxGraph graph,
    const onnxMemoryFenceV1* inputFence,
    onnxMemoryFenceV1* outputFence);
typedef onnxStatus
  (ONNXIFI_ABI* onnxReleaseGraphFunction)(
    onnxGraph graph);

/**
 * Get stable IDs of available backends on the system.
 *
 * ONNXIFI backend is a combination of software layer and hardware device used
 * to run an ONNX graph. The same software layer may expose multiple backends
 * (e.g. one ONNXIFI backend for each GPU in the system, or one ONNXIFI backend
 * for GPU and another for CPU, both implemented in the same software). Backends
 * implemented in the same software, but targeting different devices (e.g.
 * "MyNN" for CPU and "MyNN" for GPU) have different backend IDs.
 *
 * Note that some (hot-pluggable) backends can be connected and disconnected at
 * any time, and thus subsequent calls to this function may return different
 * number or set of backend IDs. The returned IDs, however, stay valid even if
 * the hardware device used by the backend disconnects from the system.
 *
 * To avoid resource leak, the backend ID MUST be released through a call to
 * onnxReleaseBackendID when it is no longer needed.
 *
 * @param backendIDs[out] - pointer to the memory location where the backend IDs
 *                          will be returned. If the pointer is NULL, it is
 *                          ignored, and the function returns only the number
 *                          of backend IDs through numBackendIDs pointer.
 * @param numBackendIDs[in,out] - pointer to a variable specifying number of
 *                                available backends. On function entry, the
 *                                variable MUST contain the capacity, in number
 *                                of backend IDs, of the memory buffer specified
 *                                by backendIDs. For successful completion, this
 *                                capacity must be at least as large as the
 *                                number of available backends. If the function
 *                                completes with either ONNXIFI_STATUS_SUCCESS
 *                                or ONNXIFI_STATUS_FALLBACK status codes, the
 *                                number of backend IDs written into backendIDs
 *                                buffer is stored in the variable specified by
 *                                this pointer.
 *
 * @retval ONNXIFI_STATUS_SUCCESS The function call succeeded, and backend IDs
 *                                are stored in the location specified by
 *                                backendIDs, and the number of the backends
 *                                is stored in the location specified by
 *                                numBackends.
 * @retval ONNXIFI_STATUS_FALLBACK The function call completed, but the
 *                                 backend IDs were not stored in the
 *                                 location specified by backendIDs, either
 *                                 because it is NULL, or because the size of
 *                                 the memory buffer is insufficient to store
 *                                 all available backend IDs. The number of
 *                                 available backends is stored in the
 *                                 location specified by numBackends.
 * @retval ONNXIFI_STATUS_INVALID_POINTER The function call failed because
 *                                        numBackends is NULL.
 * @retval ONNXIFI_STATUS_NO_SYSTEM_MEMORY The function call failed because the
 *                                         system failed to allocate memory
 *                                         to store backend ID information.
 * @retval ONNXIFI_STATUS_INTERNAL_ERROR The function call failed because the
 *                                       implementation experienced an
 *                                       unrecovered internal error.
 */
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
  onnxGetBackendIDs(
    onnxBackendID* backendIDs,
    size_t* numBackends);

/**
 * Deinitialize ONNXIFI backend IDs and release associated resources.
 *
 * The user MUST deinitialize all objects created with this backend ID
 * (onnxBackend, onnxGraph, onnxEvent) before calling this function to
 * deinitialize the backend ID.
 *
 * @param backendID - backend ID returned by onnxGetBackendIDs.
 *
 * @retval ONNXIFI_STATUS_SUCCESS The function call succeeded and the resources
 *                                associated to the backend ID were released to
 *                                the operating system.
 * @retval ONNXIFI_STATUS_INVALID_ID The function call failed because backendID
 *                                   is not an ONNXIFI backend ID.
 * @retval ONNXIFI_STATUS_INTERNAL_ERROR The function call failed because the
 *                                       implementation experienced an
 *                                       unrecovered internal error.
 */
ONNXIFI_PUBLIC onnxStatus ONNXIFI_ABI
  onnxReleaseBackendID(
    onnxBackendID backendID);

/**
 * Query high-level information about the backend and its target device.
 *
 * ONNXIFI backend is a combination of software layer and hardware device used
 * to run an ONNX graph. The same software layer may expose multiple backends
 * (e.g. one ONNXIFI backend for each GPU in the system, or one ONNXIFI backend
 * for GPU and another for CPU, both implemented in the same software).
 *
 * The content, data type, and availability of information provided by this
 * function depends on infoType value as specified below:
 *
 *         infoType value                           data type      support
 *     ONNXIFI_BACKEND_ONNXIFI_VERSION               uint64_t     required
 *     ONNXIFI_BACKEND_NAME                           char[]      required
 *     ONNXIFI_BACKEND_VENDOR                         char[]      required
 *     ONNXIFI_BACKEND_VERSION                        char[]      required
 *     ONNXIFI_BACKEND_EXTENSIONS                     char[]      required
 *     ONNXIFI_BACKEND_DEVICE                         char[]      required
 *     ONNXIFI_BACKEND_DEVICE_TYPE                   onnxEnum     required
 *     ONNXIFI_BACKEND_ONNX_IR_VERSION                char[]      required
 *     ONNXIFI_BACKEND_OPSET_VERSION                  char[]      required
 *     ONNXIFI_BACKEND_CAPABILITIES                onnxBitfield   required
 *     ONNXIFI_BACKEND_INIT_PROPERTIES             onnxBitfield   required
 *     ONNXIFI_BACKEND_MEMORY_TYPES                onnxBitfield   required
 *     ONNXIFI_BACKEND_GRAPH_INIT_PROPERTIES       onnxBitfield   required
 *     ONNXIFI_BACKEND_SYNCHRONIZATION_TYPES       onnxBitfield   required
 *     ONNXIFI_BACKEND_MEMORY_SIZE                   uint64_t     required
 *     ONNXIFI_BACKEND_MAX_GRAPH_SIZE                uint64_t     required
 *     ONNXIFI_BACKEND_MAX_GRAPH_COUNT               uint64_t     required
 *     ONNXIFI_BACKEND_MACS_FP32                     uint64_t     optional
 *     ONNXIFI_BACKEND_MACS_FP16                     uint64_t     optional
 *     ONNXIFI_BACKEND_MEMORY_BANDWIDTH              uint64_t     optional
 *     ONNXIFI_BACKEND_CPU_MEMORY_READ_BANDWIDTH     uint64_t     optional
 *     ONNXIFI_BACKEND_CPU_MEMORY_WRITE_BANDWIDTH    uint64_t     optional
 *     ONNXIFI_BACKEND_PCI_BUS_ID                    uint64_t     optional
 *     ONNXIFI_BACKEND_PCI_DEVICE_ID                 uint64_t     optional
 *     ONNXIFI_BACKEND_PCI_DOMAIN_ID                 uint64_t     optional
 *     ONNXIFI_BACKEND_DIRECTX_ID                      LUID       optional
 *     ONNXIFI_BACKEND_CUDA_INDEX                    uint64_t     optional
 *     ONNXIFI_BACKEND_OPENCL_PLATFORM_ID         cl_platform_id  optional
 *     ONNXIFI_BACKEND_OPENCL_DEVICE_ID            cl_device_id   optional
 *
 * @param backendID - ID of the backend to query.
 * @param infoType - type of the backend information to query. Must be one of
 *                   the ONNXIFI_BACKEND_* constants. If this value is not
 *                   supported by the backend, the function will fail with
 *                   ONNXIFI_STATUS_UNSUPPORTED_ATTRIBUTE.
 * @param infoValue[out] - pointer to the memory location where the backend
 *                         information value will be returned. If the pointer is
 *                         NULL, is it ignored.
 * @param infoValueSize[in,out] - pointer to a variable specifying size, in
 *                                bytes, of the information value. On function
 *                                entry, the variable MUST contain the size of
 *                                the memory buffer specified by infoValue.
 *                                For successful completion, this size must be
 *                                at least as large as the queried value. If the
 *                                function completes with either
 *                                ONNXIFI_STATUS_SUCCESS or
 *                                ONNXIFI_STATUS_FALLBACK status codes, the
 *                                actual size of the value queried in the call
 *                                is stored in the variable specified by this
 *                                pointer.
 *
 * @retval ONNXIFI_STATUS_SUCCESS The function call succeeded, and requested
 *                                value is stored in the location specified by
 *                                infoValue, and the actual size of the
 *                                requested value is stored in the location
 *                                specified by infoValueSize.
 * @retval ONNXIFI_STATUS_FALLBACK The function call completed, but the
 *                                 requested value was not stored in the
 *                                 location specified by infoValue, either
 *                                 because it is NULL, or because the size of
 *                                 the memory buffer is insufficient for the
 *                                 value. The actual size of the requested value
 *                                 is stored in the location specified by
 *                                 infoValueSize.
 * @retval ONNXIFI_STATUS_INVALID_ID The function call failed because backendID
 *                                   is not an ONNXIFI backend ID.
 * @retval ONNXIFI_STATUS_INVALID_POINTER The function call failed because
 *                                        infoValueSize is NULL.
 * @retval ONNXIFI_STATUS_UNSUPPORTED_ATTRIBUTE The function call failed because
 *                                              the value of infoType is not
 *                                              supported by the backend.
 * @retval ONNXIFI_STATUS_BACKEND_UNAVAILABLE The function call failed because
 *                                            the backend was disconnected or
 *                                            uninstalled from the system.
 */
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
  onnxGetBackendInfo(
    onnxBackendID backendID,
    onnxBackendInfo infoType,
    void* infoValue,
    size_t* infoValueSize);

/**
 * Query if an ONNX model graph is compatible with the backend.
 *
 * Model graph is passed as a serialized ModelProto message, where types and
 * dimensions of all inputs (including static weights) and outputs are specified
 * through ModelProto.graph.input and ModelProto.graph.output messages. If the
 * backend supports ONNXIFI_CAPABILITY_SYMBOLIC_SIZE_TENSORS, some of the shape
 * dimensions can be symbolic. If the backend supports
 * ONNXIFI_CAPABILITY_SYMBOLIC_BATCH_SIZE, the outer shape dimension can be
 * symbolic. In these cases, the validation of symbolic dimension should be
 * deferred until graph inputs and outputs are specified in onnxSetGraphIO.
 *
 * Commonly, the serialized ModelProto message passed to this function would
 * not include the static weights (ModelProto.graph.initializer is empty), and
 * the backend implementation MUST NOT rely on the weights to determine if the
 * graph is supported.
 *
 * An important use-case is a ModelProto containing only a single NodeProto in
 * ModelProto.graph.node, which happens when a high-level framework checks
 * operators one-by-one to find a connected subgraph that can be offloaded to
 * the backend. Backend implementations SHOULD optimize performance for this
 * use-case.
 *
 * @param backend - ID of the backend to query.
 * @param onnxModelSize - size of the serialized ONNX ModelProto message,
 *                        in bytes.
 * @param[in] onnxModel - pointer to serialized ONNX ModelProto message
 *                        representing the model graph.
 *
 * @retval ONNXIFI_STATUS_SUCCESS The function call succeeded and the model
 *                                graph can efficiently run on the backend.
 * @retval ONNXIFI_STATUS_FALLBACK The function call succeeded and the model
 *                                 graph can run on the backend through some
 *                                 emulation layer with some efficiency loss. If
 *                                 a backend decomposes this operator into
 *                                 multiple sub-operators, it should return this
 *                                 code. E.g. if a backend does not natively
 *                                 support grouped or depthwise convolution, but
 *                                 can execute it as multiple unit-group
 *                                 convolution operators, it must returns this
 *                                 code.
 * @retval ONNXIFI_STATUS_INVALID_ID The function call failed because backendID
 *                                   is not an ONNXIFI backend ID.
 * @retval ONNXIFI_STATUS_INVALID_POINTER The function call failed because
 *                                        onnxModel is NULL.
 * @retval ONNXIFI_STATUS_INVALID_SIZE The function call failed because
 *                                     onnxModelSize is 0.
 * @retval ONNXIFI_STATUS_INVALID_PROTOBUF The function call failed because it
 *                                         couldn't parse the serialized
 *                                         protobuf as an ONNX ModelProto
 *                                         message.
 * @retval ONNXIFI_STATUS_INVALID_MODEL The function call failed because the
 *                                      parsed ModelProto message does not
 *                                      satisfy ONNX requirements and
 *                                      constraints.
 * @retval ONNXIFI_STATUS_UNSUPPORTED_VERSION The function call failed because
 *                                            the ONNX IR version or operator
 *                                            version is not supported by the
 *                                            backend.
 * @retval ONNXIFI_STATUS_UNSUPPORTED_OPERATOR The function call failed because
 *                                             one of the operators in the model
 *                                             graph is not supported by the
 *                                             backend.
 * @retval ONNXIFI_STATUS_UNSUPPORTED_ATTRIBUTE The function call failed because
 *                                              the backend does not support the
 *                                              particular AttributeProto
 *                                              values in one of the operators.
 * @retval ONNXIFI_STATUS_UNSUPPORTED_SHAPE The function call failed because the
 *                                          backend does not support the
 *                                          tensor shapes in an input or output
 *                                          of one of the operators. The
 *                                          problematic tensor shapes could be
 *                                          directly specified through
 *                                          ValueInfoProto in GraphProto.input,
 *                                          GraphProto.output, or
 *                                          GraphProto.value_info, through
 *                                          TensorProto in
 *                                          GraphProto.initializer, or inferred
 *                                          from the inputs by the backend.
 * @retval ONNXIFI_STATUS_UNSUPPORTED_DATATYPE The function call failed because
 *                                             the backend does not support the
 *                                             data types in an input or output
 *                                             of one of the operators. The
 *                                             problematic data types could be
 *                                             directly specified through
 *                                             ValueInfoProto in
 *                                             GraphProto.input,
 *                                             GraphProto.output, or
 *                                             GraphProto.value_info, through
 *                                             TensorProto in
 *                                             GraphProto.initializer, or
 *                                             inferred from the inputs by the
 *                                             backend.
 * @retval ONNXIFI_STATUS_MISMATCHING_SHAPE The function call failed because
 *                                          output or intermediate shapes
 *                                          specified in the ONNX model graph do
 *                                          not match the shapes inferred from
 *                                          input shapes.
 * @retval ONNXIFI_STATUS_MISMATCHING_DATATYPE The function call failed because
 *                                             output or intermediate data types
 *                                             specified in the ONNX model graph
 *                                             do not match the data types
 *                                             inferred from graph inputs.
 * @retval ONNXIFI_STATUS_NO_SYSTEM_MEMORY The function call failed because the
 *                                         backend could not allocate enough
 *                                         system memory to parse and analyze
 *                                         the model graph.
 * @retval ONNXIFI_STATUS_BACKEND_UNAVAILABLE The function call failed because
 *                                            the backend was disconnected or
 *                                            uninstalled from the system.
 * @retval ONNXIFI_STATUS_INTERNAL_ERROR The function call failed because the
 *                                       backend experienced an unrecovered
 *                                       internal error.
 */
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
  onnxGetBackendCompatibility(
    onnxBackendID backendID,
    size_t onnxModelSize,
    const void* onnxModel);

/**
 * Initialize an ONNXIFI backend.
 *
 * ONNXIFI backend is a combination of software layer and hardware device used
 * to run an ONNXIFI graph. The same software layer may expose multiple backends
 * (e.g. one ONNXIFI backend for each GPU in the system, or one ONNXIFI backend
 * for GPU and another for CPU, both implemented in the same software).
 *
 * @param backendID - ID of the backend to initialize.
 * @param[in] auxPropertiesList - optional list of backend initialization
 *                                properties, terminated by
 *                                ONNXIFI_BACKEND_PROPERTY_NONE entry. Can be
 *                                NULL or empty.
 * @param[out] backend - pointer to an opaque handle for the initialized ONNXIFI
 *                       backend. If the function fails, the handle is
 *                       initialized to NULL.
 *
 * @retval ONNXIFI_STATUS_SUCCESS The function call succeeded and the backend
 *                                was successfully initialized.
 * @retval ONNXIFI_STATUS_INVALID_ID The function call failed because backendID
 *                                   is not an ONNXIFI backend ID.
 * @retval ONNXIFI_STATUS_INVALID_POINTER The function call failed because
 *                                        backend pointer is NULL.
 * @retval ONNXIFI_STATUS_INVALID_PROPERTY The function call failed because one
 *                                         of the backend initialization
 *                                         property values is invalid.
 * @retval ONNXIFI_STATUS_UNSUPPORTED_PROPERTY The function call failed because
 *                                             backend does not recognize one
 *                                             of the initialization
 *                                             property IDs.
 * @retval ONNXIFI_STATUS_NO_SYSTEM_MEMORY The function call failed due to
 *                                         insufficient system memory to
 *                                         initialize backend.
 * @retval ONNXIFI_STATUS_NO_SYSTEM_RESOURCES The function call failed due to
 *                                            insufficient non-memory system
 *                                            resources (e.g. file handles) to
 *                                            initialize the backend.
 * @retval ONNXIFI_STATUS_NO_DEVICE_MEMORY The function call failed due to
 *                                         insufficient backend-specific memory
 *                                         to initialize the backend.
 * @retval ONNXIFI_STATUS_NO_DEVICE_RESOURCES The function call failed due to
 *                                            insufficient non-memory
 *                                            backend-specific resources (e.g.
 *                                            command queues) to initialize the
 *                                            backend.
 * @retval ONNXIFI_STATUS_BACKEND_UNAVAILABLE The function call failed because
 *                                            the backend was disconnected or
 *                                            uninstalled from the system.
 * @retval ONNXIFI_STATUS_INTERNAL_ERROR The function call failed because the
 *                                       backend experienced an unrecovered
 *                                       internal error.
 */
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
  onnxInitBackend(
    onnxBackendID backendID,
    const uint64_t* auxPropertiesList,
    onnxBackend* backend);

/**
 * Deinitialize an ONNXIFI backend and release associated resources.
 *
 * The user MUST deinitialize all objects created on this backend (onnxGraph,
 * onnxEvent) before calling this function to deinitialize the backend.
 *
 * @param backend - ONNXIFI backend handle created by onnxInitBackend.
 *
 * @retval ONNXIFI_STATUS_SUCCESS The function call succeeded and the backend
 *                                resources were released to the operating
 *                                system.
 * @retval ONNXIFI_STATUS_INVALID_BACKEND The function call failed because
 *                                        backend is not an ONNXIFI backend
 *                                        handle.
 * @retval ONNXIFI_STATUS_INTERNAL_ERROR The function call failed because the
 *                                       backend experienced an unrecovered
 *                                       internal error.
 */
ONNXIFI_PUBLIC onnxStatus ONNXIFI_ABI
  onnxReleaseBackend(
    onnxBackend backend);

/**
 * Initialize a single-shot ONNXIFI event.
 *
 * The newly created event is in non-signalled state.
 *
 * @param backend - backend handle created by onnxInitBackend. This backend
 *                  would be used to initialize the event.
 * @param[out] event - pointer to the opaque handle for the created ONNXIFI
 *                     event. If the function fails, the handle is initialized
 *                     to NULL.
 *
 * @retval ONNXIFI_STATUS_SUCCESS The function call succeeded and the event
 *                                was successfully initialized.
 * @retval ONNXIFI_STATUS_INVALID_BACKEND The function call failed because
 *                                        backend is not an ONNXIFI backend
 *                                        handle.
 * @retval ONNXIFI_STATUS_INVALID_POINTER The function call failed because
 *                                        event pointer is NULL.
 * @retval ONNXIFI_STATUS_NO_SYSTEM_MEMORY The function call failed due to
 *                                         insufficient system memory to
 *                                         initialize the event.
 * @retval ONNXIFI_STATUS_NO_SYSTEM_RESOURCES The function call failed due to
 *                                            insufficient non-memory system
 *                                            resources (e.g. file handles) to
 *                                            initialize the event.
 * @retval ONNXIFI_STATUS_NO_DEVICE_MEMORY The function call failed due to
 *                                         insufficient backend-specific memory
 *                                         to initialize the event.
 * @retval ONNXIFI_STATUS_NO_DEVICE_RESOURCES The function call failed due to
 *                                            insufficient non-memory
 *                                            backend-specific resources (e.g.
 *                                            command queues) to initialize the
 *                                            event.
 * @retval ONNXIFI_STATUS_BACKEND_UNAVAILABLE The function call failed because
 *                                            the backend was disconnected or
 *                                            uninstalled from the system.
 * @retval ONNXIFI_STATUS_INTERNAL_ERROR The function call failed because the
 *                                       backend experienced an unrecovered
 *                                       internal error.
 */
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
  onnxInitEvent(
    onnxBackend backend,
    onnxEvent* event);

/**
 * Change the state of an ONNXIFI event to signalled.
 *
 * @param event - event handle created by onnxInitEvent. While it is technically
 *                possible to use this function for output memory fence event
 *                created by onnxRunGraph, users SHOULD NOT do that.
 *
 * @retval ONNXIFI_STATUS_SUCCESS The function call succeeded and the event
 *                                was changed to signalled state.
 * @retval ONNXIFI_STATUS_INVALID_EVENT The function call failed because event
 *                                      is not an ONNXIFI event handle.
 * @retval ONNXIFI_STATUS_INVALID_STATE The function call failed because event
 *                                      is already in the signalled state.
 * @retval ONNXIFI_STATUS_BACKEND_UNAVAILABLE The function call failed because
 *                                            the backend was disconnected or
 *                                            uninstalled from the system.
 * @retval ONNXIFI_STATUS_INTERNAL_ERROR The function call failed because the
 *                                       implementation experienced an
 *                                       unrecovered internal error.
 */
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
  onnxSignalEvent(
    onnxEvent event);

/**
 * Query ONNXIFI event state without blocking.
 *
 * @param event - event handle created by onnxRunGraph. While it is technically
 *                possible to use this function to events created by
 *                onnxInitEvent, this is not the intended use-case.
 * @param[out] state - pointer to the variable that will store the state of the
 *                     event. If the function fails, the variable is initialized
 *                     to ONNXIFI_EVENT_STATE_INVALID.
 *
 * @retval ONNXIFI_STATUS_SUCCESS The function call succeeded and the state
 *                                variable was initialized to either
 *                                ONNXIFI_EVENT_STATE_SIGNALLED or
 *                                ONNXIFI_EVENT_STATE_NONSIGNALLED according
 *                                to the state of the event.
 * @retval ONNXIFI_STATUS_INVALID_EVENT The function call failed because event
 *                                      is not an ONNXIFI event handle.
 * @retval ONNXIFI_STATUS_INVALID_POINTER The function call failed because state
 *                                        pointer is NULL.
 * @retval ONNXIFI_STATUS_BACKEND_UNAVAILABLE The function call failed because
 *                                            the backend was disconnected or
 *                                            uninstalled from the system.
 * @retval ONNXIFI_STATUS_INTERNAL_ERROR The function call failed because the
 *                                       implementation experienced an
 *                                       unrecovered internal error.
 */
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
  onnxGetEventState(
    onnxEvent event,
    onnxEventState* state);

/**
 * Wait until an ONNXIFI event transitions to signalled state.
 *
 * @param event - event handle created by onnxRunGraph. While it is technically
 *                possible to use this function to events created by
 *                onnxInitEvent, this is not the intended use-case.
 *
 * @retval ONNXIFI_STATUS_SUCCESS The function call succeeded and the function
 *                                returned because event transitioned to
 *                                signalled state.
 * @retval ONNXIFI_STATUS_INVALID_EVENT The function call failed because event
 *                                      is not an ONNXIFI event handle.
 * @retval ONNXIFI_STATUS_BACKEND_UNAVAILABLE The function call failed because
 *                                            the backend was disconnected or
 *                                            uninstalled from the system.
 * @retval ONNXIFI_STATUS_INTERNAL_ERROR The function call failed because the
 *                                       implementation experienced an
 *                                       unrecovered internal error.
 */
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
  onnxWaitEvent(
    onnxEvent event);

/**
 * Deinitialize an ONNXIFI event and release associated resources.
 *
 * @param event - event handle created by either onnxInitEvent or onnxRunGraph.
 *
 * @retval ONNXIFI_STATUS_SUCCESS The function call succeeded and the event
 *                                resources were released to the operating
 *                                system.
 * @retval ONNXIFI_STATUS_INVALID_EVENT The function call failed because event
 *                                      is not an ONNXIFI event handle.
 * @retval ONNXIFI_STATUS_INTERNAL_ERROR The function call failed because the
 *                                       implementation experienced an
 *                                       unrecovered internal error.
 */
ONNXIFI_PUBLIC onnxStatus ONNXIFI_ABI
  onnxReleaseEvent(
    onnxEvent event);

/**
 * Parse an ONNXIFI graph and convert it for a particular backend.
 *
 * Model graph is passed as a serialized ModelProto message, where types and
 * dimensions of all inputs (including static weights) and outputs are specified
 * through ModelProto.graph.input and ModelProto.graph.output messages. If the
 * backend supports ONNXIFI_CAPABILITY_SYMBOLIC_SIZE_TENSORS, some of the shape
 * dimensions can be symbolic. If the backend supports
 * ONNXIFI_CAPABILITY_SYMBOLIC_BATCH_SIZE, the outer shape dimension can be
 * symbolic. In these cases, their validation should be deferred until a later
 * call to onnxSetGraphIO.
 *
 * Values of all static weights of the graph must be specified either in
 * ModelProto.graph.initializer, or through the weightDescriptors parameters,
 * but not through any combination of the two methods. If the caller creates the
 * graph on the fly, it SHOULD pass weights through weightDescriptors as it
 * involves less overhead.
 *
 * Blobs and operators in this graph are independent of the blobs and operators
 * of other graphs on the same backend.
 *
 * @param backend - backend handle created by onnxInitBackend. This backend
 *                  would be used to setup and run the model graph.
 * @param[in] auxPropertiesList - optional list of graph initialization
 *                                properties, terminated by
 *                                ONNXIFI_GRAPH_PROPERTY_NONE entry. Can be
 *                                NULL or empty.
 * @param onnxModelSize - size of the serialized ONNX ModelProto message,
 *                        in bytes.
 * @param[in] onnxModel - pointer to serialized ONNX ModelProto message
 *                        representing the model graph. The backend MUST not
 *                        assume that the serialized ModelProto message is
 *                        present at this address after the function returns.
 * @param weightsCount - number of weights specified in this function call
 *                       through tensor descriptors. Alternatively, the weights
 *                       can be specified in ModelProto.graph.initializer.
 *                       If weightsCount is non-zero, weightDescriptors must be
 *                       non-NULL.
 * @param[in] weightDescriptors - descriptors of static input tensors for the
 *                                graph. Elements of this array provide location
 *                                for blobs identified by ValueInfoProto.name
 *                                listed in ModelProto.graph.input of the ONNX
 *                                graph. If this parameter is non-NULL,
 *                                all static inputs must be specified through
 *                                the tensor descriptors, and the
 *                                ModelProto.graph.initilizer list must be
 *                                empty. The tensor descriptors
 *                                must use ONNXIFI_MEMORY_TYPE_CPU memory type,
 *                                and the backend must copy the values of the
 *                                tensors and all metadata, including shape,
 *                                into its own memory before the function
 *                                returns.
 * @param[out] graph - pointer to the opaque handle for the created ONNXIFI
 *                     graph. If the function fails, and this pointer is
 *                     non-NULL, the handle is initialized to NULL.
 *
 * @retval ONNXIFI_STATUS_SUCCESS The function call succeeded and the model
 *                                graph was successfully initialized on the
 *                                backend.
 * @retval ONNXIFI_STATUS_FALLBACK The function call succeeded and the model
 *                                 graph was initialized for the backend through
 *                                 an emulation layer with substantial
 *                                 efficiency loss. If a backend decomposes an
 *                                 operator into multiple sub-operators, it
 *                                 MUST return this code. E.g. if a backend
 *                                 does not natively support grouped or
 *                                 depthwise convolution, but can execute it as
 *                                 multiple unit-group convolution operators, it
 *                                 should return this code.
 * @retval ONNXIFI_STATUS_INVALID_BACKEND The function call failed because
 *                                        backend is not an ONNXIFI backend
 *                                        handle.
 * @retval ONNXIFI_STATUS_INVALID_PROPERTY The function call failed because one
 *                                         of the graph initialization property
 *                                         values is invalid.
 * @retval ONNXIFI_STATUS_INVALID_POINTER The function call failed because
 *                                        onnxModel or graph pointer is NULL, or
 *                                        weightDescriptors pointer is NULL
 *                                        while weightsCount is non-zero.
 * @retval ONNXIFI_STATUS_INVALID_SIZE The function call failed because
 *                                     onnxModelSize is 0.
 * @retval ONNXIFI_STATUS_INVALID_PROTOBUF The function call failed because it
 *                                         couldn't parse the serialized
 *                                         protobuf as an ONNX ModelProto
 *                                         message.
 * @retval ONNXIFI_STATUS_INVALID_MODEL The function call failed because the
 *                                      parsed ModelProto message does not
 *                                      satisfy ONNX requirements and
 *                                      constraints.
 * @retval ONNXIFI_STATUS_INVALID_SHAPE The function call failed because one of
 *                                      the shape dimensions in
 *                                      weightDescriptors is 0.
 * @retval ONNXIFI_STATUS_INVALID_DATATYPE The function call failed because
 *                                         one of the data types in
 *                                         weightDescriptors is unknown to the
 *                                         backend.
 * @retval ONNXIFI_STATUS_INVALID_MEMORY_TYPE The function call failed because
 *                                            one of the memory types in
 *                                            weightDescriptors is unknown to
 *                                            the backend.
 * @retval ONNXIFI_STATUS_INVALID_MEMORY_LOCATION The function call failed
 *                                                because one of the memory
 *                                                locations in weightDescriptors
 *                                                is invalid (NULL pointer).
 * @retval ONNXIFI_STATUS_UNSUPPORTED_PROPERTY The function call failed because
 *                                             backend does not recognize one
 *                                             of the graph initialization
 *                                             property IDs.
 * @retval ONNXIFI_STATUS_UNSUPPORTED_VERSION The function call failed because
 *                                            the ONNX IR version or operator
 *                                            version is not supported by the
 *                                            backend.
 * @retval ONNXIFI_STATUS_UNSUPPORTED_OPERATOR The function call failed because
 *                                             one of the operators in the model
 *                                             graph is not supported by the
 *                                             backend.
 * @retval ONNXIFI_STATUS_UNSUPPORTED_ATTRIBUTE The function call failed because
 *                                              the backend does not support the
 *                                              particular AttributeProto
 *                                              values in one of the operators.
 * @retval ONNXIFI_STATUS_UNSUPPORTED_SHAPE The function call failed because the
 *                                          backend does not support the
 *                                          tensor shapes in an input or
 *                                          output of one of the operators.
 *                                          The problematic tensor shapes could
 *                                          be directly specified through
 *                                          ValueInfoProto in GraphProto.input,
 *                                          GraphProto.output, or
 &                                          GraphProto.value_info, through
 *                                          TensorProto in
 *                                          GraphProto.initializer, through
 *                                          weightDescriptors argument,
 *                                          or inferred from the inputs by the
 *                                          backend.
 * @retval ONNXIFI_STATUS_UNSUPPORTED_DATATYPE The function call failed because
 *                                             the backend does not support the
 *                                             data types in an input or output
 *                                             of one of the operators. The
 *                                             problematic data types could be
 *                                             directly specified through
 *                                             ValueInfoProto in
 *                                             GraphProto.input,
 *                                             GraphProto.output, or
 *                                             GraphProto.value_info, through
 *                                             TensorProto in
 *                                             GraphProto.initializer, through
 *                                             weightDescriptors argument,
 *                                             or inferred from the inputs by
 *                                             the backend.
 * @retval ONNXIFI_STATUS_UNSUPPORTED_MEMORY_TYPE The function call failed
 *                                                because one of the memory
 *                                                types in weightDescriptors is
 *                                                different from
 *                                                ONNXIFI_MEMORY_TYPE_CPU.
 * @retval ONNXIFI_STATUS_MISMATCHING_SHAPE The function call failed because
 *                                          the shapes specified in weight
 *                                          descriptors do not match the shapes
 *                                          specified in the ONNX model graph,
 *                                          or output or intermediate shapes
 *                                          specified in the ONNX model graph do
 *                                          not match the shapes inferred from
 *                                          input shapes.
 * @retval ONNXIFI_STATUS_MISMATCHING_DATATYPE The function call failed because
 *                                             data types specified in weight
 *                                             descriptors do not match the data
 *                                             types specified in ONNX model
 *                                             graph, or output or intermediate
 *                                             data types specified in the ONNX
 *                                             model graph do not match the data
 *                                             types inferred from graph inputs.
 * @retval ONNXIFI_STATUS_NO_SYSTEM_MEMORY The function call failed because the
 *                                         backend could not allocate enough
 *                                         system memory to parse, analyze, and
 *                                         initialize the model graph.
 * @retval ONNXIFI_STATUS_NO_SYSTEM_RESOURCES The function call failed due to
 *                                            insufficient non-memory system
 *                                            resources (e.g. file handles) to
 *                                            initialize the graph.
 * @retval ONNXIFI_STATUS_NO_DEVICE_MEMORY The function call failed due to
 *                                         insufficient backend-specific memory
 *                                         to initialize the graph.
 * @retval ONNXIFI_STATUS_NO_DEVICE_RESOURCES The function call failed due to
 *                                            insufficient non-memory
 *                                            backend-specific resources (e.g.
 *                                            command queues) to initialize the
 *                                            graph.
 * @retval ONNXIFI_STATUS_BACKEND_UNAVAILABLE The function call failed because
 *                                            the backend was disconnected or
 *                                            uninstalled from the system.
 * @retval ONNXIFI_STATUS_INTERNAL_ERROR The function call failed because the
 *                                       implementation experienced an
 *                                       unrecovered internal error.
 */
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
  onnxInitGraph(
    onnxBackend backend,
    const uint64_t* auxPropertiesList,
    size_t onnxModelSize,
    const void* onnxModel,
    uint32_t weightsCount,
    const onnxTensorDescriptorV1* weightDescriptors,
    onnxGraph* graph);

/**
 * Set locations for inputs and outputs of an ONNXIFI graph.
 *
 * The caller MUST ensure that the memory buffers specified for input and output
 * tensors remain accessible until all in-flight graph executions which use
 * specified buffer locations complete AND
 * - Either a next call to onnxSetGraphIO specifies different buffer locations
 * - Or the graph is deinitialized via onnxReleaseGraph
 * The caller can invalidate other data in tensor descriptors, including shape,
 * once the function returns.
 *
 * Calls to onnxRunGraph WILL use input and output locations specified in the
 * preceeding onnxSetGraphIO on the same graph. Asynchronous graph executions
 * that were in-flight before onnxSetGraphIO call will continue to use buffer
 * locations that were current when these graph executions started. An ONNXIFI
 * implementation MAY block inside onnxSetGraphIO until all in-flight graph
 * executions that started before the call complete.
 *
 * If a call to onnxSetGraphIO fails, it invalidates input and output locations
 * for the graph, and a subsequent call to onnxRunGraph will fail with
 * ONNXIFI_STATUS_UNIDENTIFIED_NAME.
 *
 * @param graph - graph handle created by onnxInitGraph.
 * @param inputsCount - number of elements in the inputDescriptors array.
 * @param[in] inputDescriptors - descriptors of input tensors for the graph.
 *                               Elements of this array must provide a location
 *                               for each ValueInfoProto.name listed in
 *                               ModelProto.graph.input of the ONNX graph.
 *                               If inputsCount is non-zero, inputDescriptors
 *                               pointer must be non-NULL.
 * @param outputsCount - number of elements in the outputDescriptors array.
 *                       Must be greater than zero.
 * @param[in] outputDescriptors - descriptors of output tensors for the graph.
 *                                outputDescriptors pointer must be non-NULL.
 *                                Elements of this array must provide a location
 *                                for each ValueInfoProto.name listed in
 *                                ModelProto.graph.output of the ONNX graph.
 *
 * @retval ONNXIFI_STATUS_SUCCESS The function call succeeded and the all graph
 *                                inputs and outputs were matched to a memory
 *                                location.
 * @retval ONNXIFI_STATUS_INVALID_GRAPH The function call failed because
 *                                      graph is not an ONNXIFI graph handle.
 * @retval ONNXIFI_STATUS_INVALID_POINTER The function call failed because
 *                                        outputDescriptors pointer is NULL or
 *                                        inputDescriptors pointer is NULL while
 *                                        inputsCount is non-zero.
 * @retval ONNXIFI_STATUS_INVALID_NAME The function call failed because one of
 *                                     the names in tensor descriptors doesn't
 *                                     match blob name in ModelProto.graph.input
 *                                     or ModelProto.graph.output, or the same
 *                                     name appears in more than one tensor
 *                                     descriptor.
 * @retval ONNXIFI_STATUS_INVALID_SHAPE The function call failed because one of
 *                                      the shape dimensions is 0.
 * @retval ONNXIFI_STATUS_INVALID_DATATYPE The function call failed because
 *                                         one of the data types in
 *                                         inputDescriptors or outputDescriptors
 *                                         is unknown to the backend.
 * @retval ONNXIFI_STATUS_INVALID_MEMORY_TYPE The function call failed because
 *                                            one of the memory types in
 *                                            inputDescriptors or
 *                                            outputDescriptors is unknown to
 *                                            the backend.
 * @retval ONNXIFI_STATUS_INVALID_MEMORY_LOCATION The function call failed
 *                                                because one of the memory
 *                                                locations in inputDescriptors
 *                                                or outputDescriptors is not
 *                                                valid for the specified
 *                                                memory type (e.g. NULL pointer
 *                                                for ONNXIFI_MEMORY_TYPE_CPU).
 * @retval ONNXIFI_STATUS_UNSUPPORTED_TAG The function call failed because one
 *                                        of the tags in inputDescriptors or
 *                                        outputDescriptors is unknown to the
 *                                        backend (tag does not match
 *                                        ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1).
 * @retval ONNXIFI_STATUS_UNSUPPORTED_SHAPE The function call failed because the
 *                                          backend does not support the
 *                                          tensor shapes in an input or output
 *                                          of one of the operators. The
 *                                          problematic tensor shapes could be
 *                                          directly specified through
 *                                          inputDescriptors or
 *                                          outputDescriptors argument,
 *                                          or inferred from the inputs by the
 *                                          backend. This error code can be
 *                                          returned when the backend supports
 *                                          variable-size inputs and outputs,
 *                                          and the problematic tensor shape was
 *                                          provided in the ValueInfoProto as a
 *                                          symbolic variable.
 * @retval ONNXIFI_STATUS_UNSUPPORTED_MEMORY_TYPE The function call failed
 *                                                because the backend does not
 *                                                support one of the memory
 *                                                types in inputDescriptors or
 *                                                outputDescriptors.
 * @retval ONNXIFI_STATUS_UNIDENTIFIED_NAME The function call failed because one
 *                                          of the ValueInfoProto.name value in
 *                                          ModelProto.graph.input or
 *                                          ModelProto.graph.output doesn't have
 *                                          a match in the inputDescriptors or
 *                                          outputDescriptors.
 * @retval ONNXIFI_STATUS_MISMATCHING_SHAPE The function call failed because
 *                                          the shapes specified through
 *                                          inputDescriptors or
 *                                          outputDescriptors argument are
 *                                          inconsistent with the shapes
 *                                          specified in the ONNX model graph.
 * @retval ONNXIFI_STATUS_MISMATCHING_DATATYPE The function call failed because
 *                                             data types specified through
 *                                             inputDescriptors or
 *                                             outputDescriptors argument are
 *                                             inconsistent with the data types
 *                                             specified in the ONNX model
 *                                             graph.
 * @retval ONNXIFI_STATUS_NO_SYSTEM_MEMORY The function call failed because the
 *                                         backend could not allocate enough
 *                                         system memory to parse, analyze, and
 *                                         initialize the tensor locations.
 * @retval ONNXIFI_STATUS_NO_SYSTEM_RESOURCES The function call failed due to
 *                                            insufficient non-memory system
 *                                            resources (e.g. file handles) to
 *                                            initialize the tensor locations.
 * @retval ONNXIFI_STATUS_NO_DEVICE_MEMORY The function call failed due to
 *                                         insufficient backend-specific memory
 *                                         to initialize the tensor locations.
 * @retval ONNXIFI_STATUS_NO_DEVICE_RESOURCES The function call failed due to
 *                                            insufficient non-memory
 *                                            backend-specific resources (e.g.
 *                                            command queues) to initialize the
 *                                            tensor locations.
 * @retval ONNXIFI_STATUS_BACKEND_UNAVAILABLE The function call failed because
 *                                            the backend was disconnected or
 *                                            uninstalled from the system.
 * @retval ONNXIFI_STATUS_INTERNAL_ERROR The function call failed because the
 *                                       backend experienced an unrecovered
 *                                       internal error.
 */
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
  onnxSetGraphIO(
    onnxGraph graph,
    uint32_t inputsCount,
    const onnxTensorDescriptorV1* inputDescriptors,
    uint32_t outputsCount,
    const onnxTensorDescriptorV1* outputDescriptors);

/**
 * Asynchronously execute operations in an ONNXIFI graph using pre-specified
 * locations for inputs and outputs.
 *
 * This function operates asynchronously: it doesn't require that the locations
 * for graph inputs graph inputs hold valid values before the function is
 * called, and doesn't guarantee that the locations for graph outputs hold
 * valid values when the function returns. Instead, two synchronization
 * primitives are used to signal to the backend when inputs are ready to use,
 * and to signal to the caller when outputs are ready to use. The only
 * synchronization primitive that is always available is onnxEvent
 * (ONNXIFI_SYNCHRONIZATION_EVENT memory fence type). If a backend supports
 * additional types of synchronization primitives, it must indicate them in
 * ONNXIFI_BACKEND_SYNCHRONIZATION_TYPES information query.
 *
 * The caller must successfully specify locations of input and output tensors
 * for the graph through onnxSetGraphIO before calling this function.
 *
 * @param graph - graph handle created by onnxInitGraph.
 * @param[in] inputFence - synchronization primitive that signals when graph
 *                         inputs are ready to use by the backend. The
 *                         synchronization primitive always must be initialized
 *                         by the caller.
 * @param[out] outputFence - synchronization primitive that signals when graph
 *                           outputs are ready to use by the caller. The type
 *                           of the synchronization primitive always must be
 *                           initialized by the caller. The type of the
 *                           synchronization primitive determines whether it
 *                           is initialized by the user before the call or by
 *                           the backend as a result of this call. Single-shot
 *                           synchronizatiom objects are initialized as a result
 *                           of the call. Reusable synchronization objects are
 *                           generally initialized by the user prior to the
 *                           call.
 *
 * @retval ONNXIFI_STATUS_SUCCESS The function call succeeded and the all graph
 *                                inputs and outputs were matched to a memory
 *                                location.
 * @retval ONNXIFI_STATUS_INVALID_POINTER The function call failed because
 *                                        inputFence or outputFence pointer is
 *                                        NULL.
 * @retval ONNXIFI_STATUS_INVALID_GRAPH The function call failed because
 *                                      graph is not an ONNXIFI graph handle.
 * @retval ONNXIFI_STATUS_INVALID_FENCE_TYPE The function call failed because
 *                                           the type of synchronization
 *                                           primitive specified in inputFence
 *                                           or outputFence is unknown to the
 *                                           backend.
 * @retval ONNXIFI_STATUS_INVALID_EVENT The function call failed because
 *                                      the memory synchronization primitive
 *                                      specified in inputFence or outputFence
 *                                      is not valid (e.g. NULL onnxEvent).
 * @retval ONNXIFI_STATUS_UNSUPPORTED_TAG The function call failed because a tag
 *                                        in inputFence or outputFence is
 *                                        unknown to the backend (tag does not
 *                                        match ONNXIFI_TAG_MEMORY_FENCE_V1).
 * @retval ONNXIFI_STATUS_UNSUPPORTED_FENCE_TYPE The function call failed
 *                                               because the backend does not
 *                                               support the type of
 *                                               synchronization primitive
 *                                               specified in inputFence or
 *                                               outputFence.
 * @retval ONNXIFI_STATUS_UNIDENTIFIED_NAME The function call failed because
 *                                          some of the ValueInfoProto.name
 *                                          value in ModelProto.graph.input or
 *                                          ModelProto.graph.output were not
 *                                          specified in a call to
 *                                          onnxSetGraphIO.
 * @retval ONNXIFI_STATUS_NO_SYSTEM_MEMORY The function call failed because the
 *                                         backend could not allocate enough
 *                                         system memory to execute the model
 *                                         graph.
 * @retval ONNXIFI_STATUS_NO_SYSTEM_RESOURCES The function call failed due to
 *                                            insufficient non-memory system
 *                                            resources (e.g. file handles) to
 *                                            execute the model graph.
 * @retval ONNXIFI_STATUS_NO_DEVICE_MEMORY The function call failed due to
 *                                         insufficient backend-specific memory
 *                                         to execute the graph.
 * @retval ONNXIFI_STATUS_NO_DEVICE_RESOURCES The function call failed due to
 *                                            insufficient non-memory
 *                                            backend-specific resources (e.g.
 *                                            command queues) to execute the
 *                                            graph.
 * @retval ONNXIFI_STATUS_BACKEND_UNAVAILABLE The function call failed because
 *                                            the backend was disconnected or
 *                                            uninstalled from the system.
 * @retval ONNXIFI_STATUS_INTERNAL_ERROR The function call failed because the
 *                                       backend experienced an unrecovered
 *                                       internal error.
 */
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
  onnxRunGraph(
    onnxGraph graph,
    const onnxMemoryFenceV1* inputFence,
    onnxMemoryFenceV1* outputFence);

/**
 * Deinitialize an ONNXIFI graph and release associated resources.
 *
 * If there are in-flight asynchronous inference operations on this graph,
 * the function MUST block until all outstanding operations complete.
 *
 * @param graph - graph handle created by onnxInitGraph.
 *
 * @retval ONNXIFI_STATUS_SUCCESS The function call succeeded and the graph
 *                                resources were released to the operating
 *                                system.
 * @retval ONNXIFI_STATUS_INVALID_GRAPH The function call failed because graph
 *                                      is not an ONNXIFI graph handle.
 * @retval ONNXIFI_STATUS_INTERNAL_ERROR The function call failed because the
 *                                       graph backend experienced an
 *                                       unrecovered internal error.
 */
ONNXIFI_PUBLIC onnxStatus ONNXIFI_ABI
  onnxReleaseGraph(
    onnxGraph graph);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* !defined(ONNXIFI_H) */
