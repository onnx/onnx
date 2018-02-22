#ifndef ONNX_BACKEND_H
#define ONNX_BACKEND_H 1

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(_M_IX86)
/* Windows x86 */
#define ONNX_ABI __stdcall
#elif defined(__i386__)
/* Linux x86 */
#define ONNX_ABI __attribute__((__cdecl__))
#else
#define ONNX_ABI
#endif

#ifndef ONNX_PUBLIC
#if defined(__ELF__)
#define ONNX_PUBLIC __attribute__((__visibility__("default")))
#elif defined(__MACH__)
#define ONNX_PUBLIC __attribute__((__visibility__("default")))
#elif defined(_WIN32) && defined(__GNUC__)
#ifdef ONNX_BUILD_LIBRARY
#define ONNX_PUBLIC __attribute__((__dllexport__))
#else
#define ONNX_PUBLIC __attribute__((__dllimport__))
#endif
#elif defined(_WIN32)
#ifdef ONNX_BUILD_LIBRARY
#define ONNX_PUBLIC __declspec(dllexport)
#else
#define ONNX_PUBLIC __declspec(dllimport)
#endif
#else
#define ONNX_PUBLIC
#endif
#endif

#if defined(ONNX_BACKEND_SUFFIX)
#define ONNX_SYMBOL_CONCAT_(prefix, suffix) prefix##suffix
#define ONNX_SYMBOL_CONCAT(prefix, suffix) ONNX_SYMBOL_CONCAT_(prefix, suffix)
#define ONNX_SYMBOL_NAME(symbol_name) ONNX_SYMBOL_CONCAT(symbol_name, ONNX_BACKEND_SUFFIX)
#else
#define ONNX_SYMBOL_NAME(symbol_name) symbol_name
#endif

#include <stddef.h>

#if !defined(ONNX_NO_STDINT_H)
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
#endif /* !defined(ONNX_NO_STDINT_H) */

/**
 * Opaque ONNX backend handle.
 * ONNX backend is a combination of software layer and hardware device used to
 * run an ONNX graph.
 */
typedef void* onnxBackend;
/** Opaque ONNX graph handle. */
typedef void* onnxGraph;

/** Return code for ONNX functions */
typedef int32_t onnxStatus;
/** Type for enumeration values */
typedef uint32_t onnxEnum;
/** Type for bit fields */
typedef uint32_t onnxBitfield;
/**
 * Type for pointers or handles for memory buffers.
 * This type is intended to work not only for CPU-addressable memory, but also
 * for device memory. uint64_t ensures the API can accomodate Vulkan buffers.
 */
typedef uint64_t onnxPointer;

#define ONNX_STATUS_SUCCESS 0x0000
#define ONNX_STATUS_FALLBACK 0x0001
#define ONNX_STATUS_INVALID_INDEX 0x0010
#define ONNX_STATUS_INVALID_SIZE 0x0011
#define ONNX_STATUS_INVALID_POINTER 0x0012
#define ONNX_STATUS_INVALID_PROTOBUF 0x0013
#define ONNX_STATUS_INVALID_MODEL 0x0014
#define ONNX_STATUS_INVALID_BACKEND 0x0015
#define ONNX_STATUS_INVALID_GRAPH 0x0016
#define ONNX_STATUS_INVALID_NAME 0x0017
#define ONNX_STATUS_INVALID_SHAPE 0x0018
#define ONNX_STATUS_UNSUPPORTED_VERSION 0x0020
#define ONNX_STATUS_UNSUPPORTED_OPERATOR 0x0021
#define ONNX_STATUS_UNSUPPORTED_PARAMETER 0x0022
#define ONNX_STATUS_UNIDENTIFIED_NAME 0x0023
#define ONNX_STATUS_NO_SYSTEM_MEMORY 0x0030
#define ONNX_STATUS_NO_DEVICE_MEMORY 0x0031
#define ONNX_STATUS_NO_SYSTEM_RESOURCES 0x0032
#define ONNX_STATUS_NO_DEVICE_RESOURCES 0x0033
#define ONNX_STATUS_INTERNAL_ERROR 0x0034

/** Special-purpose accelerator for neural network */
#define ONNX_DEVICE_TYPE_NPU 0x01
/** Digital signal processor */
#define ONNX_DEVICE_TYPE_DSP 0x02
/** Graphics accelerator */
#define ONNX_DEVICE_TYPE_GPU 0x04
/** General-purpose central processor */
#define ONNX_DEVICE_TYPE_CPU 0x08
/** Field-programmable gate array */
#define ONNX_DEVICE_TYPE_FPGA 0x10
/**
 * High-level framework which internally arbitrates or distributes work between
 * multiple device types/
 */
#define ONNX_DEVICE_TYPE_FRAMEWORK 0x20

/**
 * The backend supports ONNX graphs with symbolic variables in shape names
 * (using TensorShapeProto.dim_param for ModelProto.graph.input.type.shape or
 * ModelProto.graph.output.type.shape).
 *
 * The exact numerical shape of all input and output tensors must be specified
 * in the onnxSetGraphIO call(s).
 */
#define ONNX_CAPABILITY_SYMBOLIC_SIZE_TENSORS 0x01
/**
 * The backend supports ONNX graphs with data-dependent output shapes.
 * The ONNX graph would specify unknown output shapes using symbolic variables,
 * so this capability requires ONNX_CAPABILITY_SYMBOLIC_SIZE_TENSORS support.
 *
 * For outputs with data-dependent shapes the shape specified in onnxSetGraphIO
 * call is interpreted as the upper limit.
 */
#define ONNX_CAPABILITY_VARIABLE_SIZE_OUTPUTS 0x02

#define ONNX_VENDOR_NAME_MAX 32
#define ONNX_VERSION_NAME_MAX 64
#define ONNX_BACKEND_NAME_MAX 64
#define ONNX_DEVICE_NAME_MAX 64

/** Denotes onnxBackendCoreInfo structure. */
#define ONNX_BACKEND_INFO_TYPE_CORE 0
/** Denotes onnxBackendPerformanceInfo structure. */
#define ONNX_BACKEND_INFO_TYPE_PERFORMANCE 1

typedef struct onnxBackendCoreInfo {
  /**
   * Marketing name of the backend (excluding the vendor name).
   *
   * Examples:
   *    "Caffe2"
   *    "ATen"
   */
  char name[ONNX_BACKEND_NAME_MAX];
  /**
   * Name of the backend vendor.
   *
   * Examples:
   *    "Facebook"
   *    "Marat Dukhan"
   */
  char vendor[ONNX_VENDOR_NAME_MAX];
  /**
   * Version of the backend software. Exact format is vendor-specific, but
   * must be unique for the software release.
   *
   * Examples:
   *    "1.2.3"
   *    "1.2.3.0"
   *    "1.2.3-db3a4439d233276e25681fb4735b7f8e674dda65"
   */
  char version[ONNX_VERSION_NAME_MAX];
  /**
   * Descriptive name of the device (i.e. CPU, GPU, DSP, or NPU model).
   */
  char device[ONNX_DEVICE_NAME_MAX];
  /**
   * Type of the device.
   *
   * Possible values:
   *      ONNX_DEVICE_TYPE_NPU
   *      ONNX_DEVICE_TYPE_DSP
   *      ONNX_DEVICE_TYPE_GPU
   *      ONNX_DEVICE_TYPE_CPU
   *      ONNX_DEVICE_TYPE_FPGA
   */
  onnxEnum deviceType;
  /**
   * Optional features supported by the backend.
   *
   * Possible values are any combination of the following flags:
   *      ONNX_CAPABILITY_SYMBOLIC_SIZE_TENSORS
   *      ONNX_CAPABILITY_VARIABLE_SIZE_OUTPUTS
   */
  onnxBitfield capabilities;
  /**
   * Auxiliary initialization properties supported by the backend.
   */
  onnxBitfield initProperties;
  /**
   * Memory types supported for graph inputs and outputs.
   *
   * Possible values are any combination of the following flags:
   *      ONNX_MEMORY_TYPE_CPU (always supported)
   */
  onnxBitfield memoryTypes;
  /**
   * Maximum amount of memory, in bytes, available to the use by the backend.
   */
  uint64_t memorySize;
  /**
   * Maximum size of network parameters, in bytes.
   */
  uint64_t maxGraphSize;
  /**
   * Maximum number of independent network graphs supported by the backend.
   */
  uint32_t maxGraphsCount;
} onnxBackendCoreInfo;

typedef struct onnxBackendPerformanceInfo {
  /**
   * Number of FP32 multiply-accumulate operations per second delivered by
   * the backend.
   *
   * If the backend does not support FP32 computation, it must return 0.
   */
  uint64_t macsFP32;
  /**
   * Number of FP16 multiply-accumulate operations per second delivered by
   * the backend.
   *
   * If the backend does not support FP16 computation, it must return 0.
   */
  uint64_t macsFP16;
  /**
   * Bandwidth, in bytes per second, of the global memory specific to the
   * backend device.
   */
  uint64_t deviceMemoryBandwidth;
  /**
   * Bandwidth, in bytes per second, of transferring data from cacheable
   * CPU-allocated memory to the backend device.
   */
  uint64_t cpuMemoryReadBandwidth;
  /**
   * Bandwidth, in bytes per second, of transferring data to cacheable
   * CPU-allocated memory from the backend device.
   */
  uint64_t cpuMemoryWriteBandwidth;
} onnxBackendPerformanceInfo;

/* Note: the data type values match ONNX TensorProto.DataType enum */
#define ONNX_DATATYPE_UNDEFINED 0
#define ONNX_DATATYPE_FLOAT16 10
#define ONNX_DATATYPE_FLOAT32 1
#define ONNX_DATATYPE_FLOAT64 11
#define ONNX_DATATYPE_INT8 3
#define ONNX_DATATYPE_INT16 5
#define ONNX_DATATYPE_INT32 6
#define ONNX_DATATYPE_INT64 7
#define ONNX_DATATYPE_UINT8 2
#define ONNX_DATATYPE_UINT16 4
#define ONNX_DATATYPE_UINT32 12
#define ONNX_DATATYPE_UINT64 13

/** Cacheable CPU memory */
#define ONNX_MEMORY_TYPE_CPU 0

#define ONNX_TENSOR_DIMS_MAX 8

/**
 * Terminates the list of auxiliary backend initialization properties passed to
 * onnxInitBackend.
 */
#define ONNX_BACKEND_PROPERTY_NONE 0

typedef struct onnxTensorDescriptor {
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
   *     ONNX_DATATYPE_FLOAT16
   *     ONNX_DATATYPE_FLOAT32
   *     ONNX_DATATYPE_INT8
   *     ONNX_DATATYPE_INT16
   *     ONNX_DATATYPE_INT32
   *     ONNX_DATATYPE_UINT8
   *     ONNX_DATATYPE_UINT16
   *     ONNX_DATATYPE_UINT32
   */
  onnxEnum dataType;
  /**
   * Type of memory that stores the tensor.
   *
   * Possible values:
   *     ONNX_MEMORY_TYPE_CPU
   */
  onnxEnum memoryType;
  /**
   * Number of dimensions in the tensor.
   * Must be between 0 (for a scalar) and ONNX_TENSOR_DIMS_MAX.
   */
  uint32_t dimensions;
  /**
   * Dimensions of the tensor.
   */
  uint64_t shape[ONNX_TENSOR_DIMS_MAX];
  /**
   * Pointers to tensor data.
   */
  onnxPointer buffer;
} onnxTensorDescriptor;

/* Function pointer declarations for dynamic loading */
typedef uint32_t(ONNX_ABI* onnxGetNumBackendsFunction)(void);
typedef onnxStatus(ONNX_ABI* onnxGetBackendInfoFunction)(
    uint32_t index,
    void* backendInfo,
    uint32_t backendInfoVersion);
typedef onnxStatus(ONNX_ABI* onnxGetBackendCompatibilityFunction)(
    uint32_t index,
    size_t onnxModelSize,
    const void* onnxModel);
typedef onnxStatus(ONNX_ABI* onnxInitBackendFunction)(
    uint32_t index,
    const uint64_t* auxPropertiesList,
    onnxBackend* backend);
typedef onnxStatus(ONNX_ABI* onnxReleaseBackendFunction)(onnxBackend backend);
typedef onnxStatus(ONNX_ABI* onnxInitGraphFunction)(
    onnxBackend backend,
    size_t onnxModelSize,
    const void* onnxModel,
    uint32_t weightsCount,
    const onnxTensorDescriptor* weightDescriptors,
    onnxGraph* graph);
typedef onnxStatus(ONNX_ABI* onnxSetGraphIOFunction)(
    onnxGraph graph,
    uint32_t inputsCount,
    const onnxTensorDescriptor* inputDescriptors,
    uint32_t outputsCount,
    const onnxTensorDescriptor* outputDescriptors);
typedef onnxStatus(ONNX_ABI* onnxRunGraphFunction)(onnxGraph graph);
typedef onnxStatus(ONNX_ABI* onnxReleaseGraphFunction)(onnxGraph graph);

/**
 * Query the number of available backends for ONNX graphs.
 *
 * ONNX backend is a combination of software layer and hardware device used to
 * run an ONNX graph. The same software layer may expose multiple backends (e.g.
 * one ONNX backend for each GPU in the system, or one ONNX backend for GPU and
 * another for CPU, both implemented in the same software). Backends implemented
 * in the same software, but targeting different devices (e.g. "MyNN" for CPU
 * and "MyNN" for GPU) are counted separately.
 */
ONNX_PUBLIC uint32_t ONNX_ABI ONNX_SYMBOL_NAME(onnxGetNumBackends)(void);

/**
 * Query high-level information about the backend and its target device.
 *
 * ONNX backend is a combination of software layer and hardware device used to
 * run an ONNX graph. The same software layer may expose multiple backends (e.g.
 * one ONNX backend for each GPU in the system, or one ONNX backend for GPU and
 * another for CPU, both implemented in the same software).
 *
 * @param index - index of the backend to query.
 * @param[out] backendInfo - pointer to backend information structure, either
 *                           onnxBackendCoreInfo (if backendInfoType is
 *                           ONNX_BACKEND_INFO_TYPE_CORE), or
 *                           onnxBackendPerformanceInfo (if backendInfoType is
 *                           ONNX_BACKEND_INFO_TYPE_PERFORMANCE).
 *                           In case of error, this structure is unchanged.
 * @param backendInfoType - type of the backend information structure to return.
 *                          In the current version it can be
 *                          ONNX_BACKEND_INFO_TYPE_CORE or
 *                          ONNX_BACKEND_INFO_TYPE_PERFORMANCE If this value is
 *                          not supported by the backend, the function will fail
 *                          with ONNX_STATUS_UNSUPPORTED_VERSION.
 *
 * @retval ONNX_STATUS_SUCCESS The function call succeeded and backendInfo is
 *                             populated with information about the backend.
 * @retval ONNX_STATUS_INVALID_INDEX The function call failed because backend
 *                                   index is out of bounds (is greater or equal
 *                                   to the value returned by
 *                                   onnxGetNumBackends)
 * @retval ONNX_STATUS_INVALID_POINTER The function call failed because
 *                                     backendInfo is NULL.
 * @retval ONNX_STATUS_UNSUPPORTED_VERSION The function call failed because
 *                                         the value of backendInfoType is
 *                                         not supported by the backend.
 */
ONNX_PUBLIC onnxStatus ONNX_ABI ONNX_SYMBOL_NAME(onnxGetBackendInfo)(
    uint32_t index,
    void* backendInfo,
    uint32_t backendInfoType);

/**
 * Query if an ONNX model graph is compatible with the backend.
 *
 * Model graph is passed as a serialized ModelProto message, where types and
 * dimensions of all inputs (including static weights) and outputs are specified
 * through ModelProto.graph.input and ModelProto.graph.output messages. If the
 * backend supports ONNX_CAPABILITY_SYMBOLIC_SIZE_TENSORS, some of the shape
 * dimensions can be symbolic. In this case, their validation should be deferred
 * until graph inputs and outputs are specified in onnxSetGraphIO.
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
 * @param index - index of the backend to query.
 * @param onnxModelSize - size of the serialized ONNX ModelProto message,
 *                        in bytes.
 * @param[in] onnxModel - pointer to serialized ONNX ModelProto message
 *                        representing the model graph.
 *
 * @retval ONNX_STATUS_SUCCESS The function call succeeded and the model graph
 *                             can efficiently run on the backend.
 * @retval ONNX_STATUS_FALLBACK The function call succeeded and the model graph
 *                              can run on the backend through some emulation
 *                              layer with some efficiency loss. If a backend
 *                              decomposes this operator into multiple
 *                              sub-operators, it should return this code.
 *                              E.g. if a backend does not natively support
 *                              grouped or depthwise convolution, but can
 *                              execute it as multiple unit-group convolution
 *                              operators, it must returns this code.
 * @retval ONNX_STATUS_INVALID_INDEX The function call failed because backend
 *                                   index is out of bounds (is greater or equal
 *                                   to the value returned by
 *                                   onnxGetNumBackends)
 * @retval ONNX_STATUS_INVALID_POINTER The function call failed because
 *                                     onnxModel is NULL.
 * @retval ONNX_STATUS_INVALID_SIZE The function call failed because
 *                                  onnxModelSize is 0.
 * @retval ONNX_STATUS_INVALID_PROTOBUF The function call failed because it
 *                                      couldn't parse the serialized protobuf
 *                                      as an ONNX ModelProto message.
 * @retval ONNX_STATUS_INVALID_MODEL The function call failed because the parsed
 *                                   ModelProto message does not satisfy ONNX
 *                                   requirements and constraints.
 * @retval ONNX_STATUS_UNSUPPORTED_VERSION The function call failed because the
 *                                         ONNX IR version or operator version
 *                                         is not supported by the backend.
 * @retval ONNX_STATUS_UNSUPPORTED_OPERATOR The function call failed because
 *                                          one of the operators in the model
 *                                          graph is not supported by the
 *                                          backend.
 * @retval ONNX_STATUS_UNSUPPORTED_PARAMETER The function call failed because
 *                                           the backend does not support the
 *                                           particular parameter values in one
 *                                           of the operators.
 * @retval ONNX_STATUS_NO_SYSTEM_MEMORY The function call failed because the
 *                                      backend could not allocate enough
 *                                      system memory to parse and analyze
 *                                      the model graph.
 * @retval ONNX_STATUS_INTERNAL_ERROR The function call failed because the
 *                                    backend experienced an unrecovered
 *                                    internal error.
 */
ONNX_PUBLIC onnxStatus ONNX_ABI ONNX_SYMBOL_NAME(onnxGetBackendCompatibility)(
    uint32_t index,
    size_t onnxModelSize,
    const void* onnxModel);

/**
 * Initialize an ONNX backend.
 *
 * ONNX backend is a combination of software layer and hardware device used to
 * run an ONNX graph. The same software layer may expose multiple backends (e.g.
 * one ONNX backend for each GPU in the system, or one ONNX backend for GPU and
 * another for CPU, both implemented in the same software).
 *
 * @param index - index of the backend to initialize.
 * @param[in] auxPropertiesList - optional list of backend initialization
 *                                properties, terminated by
 *                                ONNX_BACKEND_PROPERTY_NONE entry. Can be NULL
 *                                or empty.
 * @param[out] backend - pointer to an opaque handle for the initialized ONNX
 *                       backend. If the function fails, the handle is
 *                       initialized to NULL.
 *
 * @retval ONNX_STATUS_SUCCESS The function call succeeded and the backend
 *                             was successfully initialized.
 * @retval ONNX_STATUS_INVALID_INDEX The function call failed because backend
 *                                   index is out of bounds (is greater or equal
 *                                   to the value returned by
 *                                   onnxGetNumBackends)
 * @retval ONNX_STATUS_INVALID_PARAMETER The function call failed because one of
 *                                       the initialization parameter values is
 *                                       invalid.
 * @retval ONNX_STATUS_UNSUPPORTED_PARAMETER The function call failed because
 *                                           backend does not recognize one of
 *                                           the initialization parameters.
 * @retval ONNX_STATUS_NO_SYSTEM_MEMORY The function call failed due to
 *                                      insufficient system memory to initialize
 *                                      the backend.
 * @retval ONNX_STATUS_NO_SYSTEM_RESOURCES The function call failed due to
 *                                         insufficient non-memory system
 *                                         resources (e.g. file handles) to
 *                                         initialize the backend.
 * @retval ONNX_STATUS_NO_DEVICE_MEMORY The function call failed due to
 *                                      insufficient backend-specific memory to
 *                                      initialize the backend.
 * @retval ONNX_STATUS_NO_DEVICE_RESOURCES The function call failed due to
 *                                         insufficient non-memory
 *                                         backend-specific resources (e.g.
 *                                         command queues) to initialize the
 *                                         backend.
 * @retval ONNX_STATUS_INTERNAL_ERROR The function call failed because the
 *                                    backend experienced an unrecovered
 *                                    internal error.
 */
ONNX_PUBLIC onnxStatus ONNX_ABI ONNX_SYMBOL_NAME(onnxInitBackend)(
    uint32_t index,
    const uint64_t* auxPropertiesList,
    onnxBackend* backend);

/**
 * Deinitialize an ONNX backend and release associated resources.
 *
 * @param backend - backend handle created by onnxInitBackend.
 *
 * @retval ONNX_STATUS_SUCCESS The function call succeeded and the backend
 *                             resources were released to the operating system.
 * @retval ONNX_STATUS_INVALID_BACKEND The function call failed because
 *                                     backend is not an ONNX backend handle.
 * @retval ONNX_STATUS_INTERNAL_ERROR The function call failed because the
 *                                    backend experienced an unrecovered
 *                                    internal error.
 */
ONNX_PUBLIC onnxStatus ONNX_ABI
    ONNX_SYMBOL_NAME(onnxReleaseBackend)(onnxBackend backend);

/**
 * Parse an ONNX graph and convert it for a particular backend.
 *
 * Model graph is passed as a serialized ModelProto message, where types and
 * dimensions of all inputs (including static weights) and outputs are specified
 * through ModelProto.graph.input and ModelProto.graph.output messages. If the
 * backend supports ONNX_CAPABILITY_SYMBOLIC_SIZE_TENSORS, some of the shape
 * dimensions can be symbolic. In this case, their validation should be deferred
 * until a later call to onnxSetGraphIO.
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
 * @param onnxModelSize - size of the serialized ONNX ModelProto message,
 *                        in bytes.
 * @param[in] onnxModel - pointer to serialized ONNX ModelProto message
 *                        representing the model graph.
 * @param weightsCount - number of weights specified in this function call
 *                       through tensor descriptors. Alternatively, the weights
 *                       can be specified in ModelProto.graph.initializer
 * @param[in] weightDescriptors - descriptors of input tensors for the graph.
 *                                Elements of this array provide location
 *                                for blobs identified by ValueInfoProto.name
 *                                listed in ModelProto.graph.input of the ONNX
 *                                graph. If this parameter is non-NULL,
 *                                all static weights must be specified through
 *                                the tensor descriptors, and the
 *                                ModelProto.graph.initilizer list must be
 *                                empty. The tensor descriptors for the weights
 *                                must use ONNX_MEMORY_TYPE_CPU memory type,
 *                                and the backend must copy the values of the
 *                                weights into its own memory before the
 *                                function returns.
 * @param[out] graph - pointer to the opaque handle for the created ONNX graph.
 *                     If the function fails, the handle is initialized to NULL.
 *
 * @retval ONNX_STATUS_SUCCESS The function call succeeded and the model graph
 *                             was successfully initialized on the backend.
 * @retval ONNX_STATUS_FALLBACK The function call succeeded and the model graph
 *                              was initialized for the backend through an
 *                              emulation layer with substantial efficiency
 *                              loss. If a backend decomposes an operator into
 *                              multiple sub-operators, it should return this
 *                              code. E.g. if a backend does not natively
 *                              support grouped or depthwise convolution, but
 *                              can execute it as multiple unit-group
 *                              convolution operators, it must return this code.
 * @retval ONNX_STATUS_INVALID_BACKEND The function call failed because backend
 *                                     is not an ONNX backend handle.
 * @retval ONNX_STATUS_INVALID_POINTER The function call failed because
 *                                     onnxModel is NULL.
 * @retval ONNX_STATUS_INVALID_SIZE The function call failed because
 *                                  onnxModelSize is 0.
 * @retval ONNX_STATUS_INVALID_PROTOBUF The function call failed because it
 *                                      couldn't parse the serialized protobuf
 *                                      as an ONNX ModelProto message.
 * @retval ONNX_STATUS_INVALID_MODEL The function call failed because the parsed
 *                                   ModelProto message does not satisfy ONNX
 *                                   requirements and constraints.
 * @retval ONNX_STATUS_UNSUPPORTED_VERSION The function call failed because the
 *                                         ONNX IR version or operator version
 *                                         is not supported by the backend.
 * @retval ONNX_STATUS_UNSUPPORTED_OPERATOR The function call failed because one
 *                                          of the operators in the model graph
 *                                          is not supported by the backend.
 * @retval ONNX_STATUS_UNSUPPORTED_PARAMETER The function call failed because
 *                                           the backend does not support the
 *                                           particular parameter values in one
 *                                           of the operators.
 * @retval ONNX_STATUS_NO_SYSTEM_MEMORY The function call failed because the
 *                                      backend could not allocate enough
 *                                      system memory to parse, analyze, and
 *                                      initialize the model graph.
 * @retval ONNX_STATUS_NO_SYSTEM_RESOURCES The function call failed due to
 *                                         insufficient non-memory system
 *                                         resources (e.g. file handles) to
 *                                         initialize the graph.
 * @retval ONNX_STATUS_NO_DEVICE_MEMORY The function call failed due to
 *                                      insufficient backend-specific memory to
 *                                      initialize the graph.
 * @retval ONNX_STATUS_NO_DEVICE_RESOURCES The function call failed due to
 *                                         insufficient non-memory
 *                                         backend-specific resources (e.g.
 *                                         command queues) to initialize the
 *                                         graph.
 */
ONNX_PUBLIC onnxStatus ONNX_ABI ONNX_SYMBOL_NAME(onnxInitGraph)(
    onnxBackend backend,
    size_t onnxModelSize,
    const void* onnxModel,
    uint32_t weightsCount,
    const onnxTensorDescriptor* weightDescriptors,
    onnxGraph* graph);

/**
 * Set locations for inputs and outputs of an ONNX graph.
 *
 * The caller MUST ensure that the memory buffers specified for input and output
 * tensors remain accessible for the life-time of the ONNX graph.
 *
 * @param graph - graph handle created by onnxInitGraph.
 * @param inputsCount - number of elements in the inputDescriptors array.
 * @param[in] inputDescriptors - descriptors of input tensors for the graph.
 *                               Elements of this array must provide a location
 *                               for each ValueInfoProto.name listed in
 *                               ModelProto.graph.input of the ONNX graph.
 * @param outputsCount - number of elements in the outputDescriptors array.
 * @param[in] outputDescriptors - descriptors of output tensors for the graph.
 *                                Elements of this array must provide a location
 *                                for each ValueInfoProto.name listed in
 *                                ModelProto.graph.output of the ONNX graph.
 *
 * @retval ONNX_STATUS_SUCCESS The function call succeeded and the all graph
 *                             inputs and outputs were matched to a memory
 *                             location.
 * @retval ONNX_STATUS_INVALID_GRAPH The function call failed because
 *                                   graph is not an ONNX graph handle.
 * @retval ONNX_STATUS_INVALID_POINTER The function call failed because
 *                                     inputDescriptors or outputDescriptors
 *                                     pointer is NULL.
 * @retval ONNX_STATUS_INVALID_NAME The function call failed because one of the
 *                                  names in tensor descriptors doesn't match
 *                                  blob name in ModelProto.graph.input or
 *                                  ModelProto.graph.output, or the same name
 *                                  appears in more than one tensor descriptor.
 * @retval ONNX_STATUS_INVALID_SHAPE The function call failed because one of the
 *                                   shape dimensions is 0.
 * @retval ONNX_STATUS_UNSUPPORTED_PARAMETER The function call failed because
 *                                           the backend does not support the
 *                                           particular data type, memory type,
 *                                           or shape specified in one of the
 *                                           operators.
 * @retval ONNX_STATUS_UNSUPPORTED_SHAPE The function call failed because the
 *                                       backend does can not support the
 *                                       shape of input or output tensors.
 *                                       This error code may be returned when
 *                                       the backend supports variable-size
 *                                       inputs and outputs, and the tensor
 *                                       shape was provided to the graph as a
 *                                       symbolic variable.
 * @retval ONNX_STATUS_UNIDENTIFIED_NAME The function call failed because one
 *                                       of the ValueInfoProto.name value in
 *                                       ModelProto.graph.input or
 *                                       ModelProto.graph.output doesn't have a
 *                                       match in the inputDescriptors or
 *                                       outputDescriptors.
 * @retval ONNX_STATUS_NO_SYSTEM_MEMORY The function call failed because the
 *                                      backend could not allocate enough
 *                                      system memory to parse, analyze, and
 *                                      initialize the tensor locations.
 * @retval ONNX_STATUS_NO_SYSTEM_RESOURCES The function call failed due to
 *                                         insufficient non-memory system
 *                                         resources (e.g. file handles) to
 *                                         initialize the tensor locations.
 * @retval ONNX_STATUS_NO_DEVICE_MEMORY The function call failed due to
 *                                      insufficient backend-specific memory to
 *                                      initialize the tensor locations.
 * @retval ONNX_STATUS_NO_DEVICE_RESOURCES The function call failed due to
 *                                         insufficient non-memory
 *                                         backend-specific resources (e.g.
 *                                         command queues) to initialize the
 *                                         tensor locations.
 * @retval ONNX_STATUS_INTERNAL_ERROR The function call failed because the
 *                                    backend experienced an unrecovered
 *                                    internal error.
 */
ONNX_PUBLIC onnxStatus ONNX_ABI ONNX_SYMBOL_NAME(onnxSetGraphIO)(
    onnxGraph graph,
    uint32_t inputsCount,
    const onnxTensorDescriptor* inputDescriptors,
    uint32_t outputsCount,
    const onnxTensorDescriptor* outputDescriptors);

/**
 * Execute operations in an ONNX graph using pre-specified locations for inputs
 * and outputs.
 *
 * This function operates synchronously: it expects that graph inputs have
 * valid values before the function is called, and will finish writing graph
 * outputs before the function returns.
 *
 * The caller must successfully specify locations of input and output tensors
 * for the graph through onnxSetGraphIO before calling this function.
 *
 * @param graph - graph handle created by onnxInitGraph.
 *
 * @retval ONNX_STATUS_SUCCESS The function call succeeded and the all graph
 *                             inputs and outputs were matched to a memory
 *                             location.
 * @retval ONNX_STATUS_INVALID_GRAPH The function call failed because
 *                                   graph is not an ONNX graph handle.
 * @retval ONNX_STATUS_UNIDENTIFIED_NAME The function call failed because some
 *                                       of the ValueInfoProto.name value in
 *                                       ModelProto.graph.input or
 *                                       ModelProto.graph.output were not
 *                                       specified in a call to onnxSetGraphIO.
 * @retval ONNX_STATUS_NO_SYSTEM_MEMORY The function call failed because the
 *                                      backend could not allocate enough
 *                                      system memory to execute the model
 *                                      graph.
 * @retval ONNX_STATUS_NO_SYSTEM_RESOURCES The function call failed due to
 *                                         insufficient non-memory system
 *                                         resources (e.g. file handles) to
 *                                         execute the model graph.
 * @retval ONNX_STATUS_NO_DEVICE_MEMORY The function call failed due to
 *                                      insufficient backend-specific memory to
 *                                      execute the graph.
 * @retval ONNX_STATUS_NO_DEVICE_RESOURCES The function call failed due to
 *                                         insufficient non-memory
 *                                         backend-specific resources (e.g.
 *                                         command queues) to execute the
 *                                         graph.
 * @retval ONNX_STATUS_INTERNAL_ERROR The function call failed because the
 *                                    backend experienced an unrecovered
 *                                    internal error.
 */
ONNX_PUBLIC onnxStatus ONNX_ABI ONNX_SYMBOL_NAME(onnxRunGraph)(onnxGraph graph);

/**
 * Deinitialize an ONNX graph and release associated resources.
 *
 * @param graph - graph handle created by onnxInitGraph.
 *
 * @retval ONNX_STATUS_SUCCESS The function call succeeded and the graph
 *                             resources were released to the operating system.
 * @retval ONNX_STATUS_INVALID_GRAPH The function call failed because graph is
 *                                   not an ONNX graph handle.
 * @retval ONNX_STATUS_INTERNAL_ERROR The function call failed because the
 *                                    graph backend experienced an unrecovered
 *                                    internal error.
 */
ONNX_PUBLIC onnxStatus ONNX_ABI
    ONNX_SYMBOL_NAME(onnxReleaseGraph)(onnxGraph graph);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* !defined(ONNX_BACKEND_H) */
