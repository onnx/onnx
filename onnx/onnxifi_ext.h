#ifndef ONNXIFI_EXT_H
#define ONNXIFI_EXT_H 1

#include "onnx/onnxifi.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Generic ONNXIFI extension function pointer.
 *
 * The caller should convert this generic function pointer to the function
 * pointer specific for an extension function type.
 */
typedef onnxStatus (ONNXIFI_ABI* onnxExtensionFunctionPointer)(void);

/**
 * Query function pointer for an ONNXIFI extension function.
 *
 * The returned function pointer is specific to the provided backend ID, and
 * MUST NOT be used with objects created for other backend IDs.
 *
 * This function is a part of onnx_extension_function extension. Backends which
 * implement this function MUST list "onnx_extension_function" in the result of
 * onnxGetBackendInfo with ONNXIFI_BACKEND_EXTENSIONS information type.
 *
 * @param backendID - ID of the backend to query for extension function.
 * @param[in] name - name of the extension function to query.
 * @param[out] function - pointer to a generic function pointer for an ONNXIFI
 *                        extension function. If the function fails, the
 *                        function pointer is initialized to NULL. The caller
 *                        MUST cast this function pointer to the type specific
 *                        for the extension function before use.
 *
 * @retval ONNXIFI_STATUS_SUCCESS The function call succeeded and the extension
 *                                function pointer is stored in the location
 *                                specified by function argument.
 * @retval ONNXIFI_STATUS_INVALID_ID The function call failed because backendID
 *                                   is not an ONNXIFI backend ID.
 * @retval ONNXIFI_STATUS_INVALID_POINTER The function call failed because
 *                                        name or function argument is NULL.
 * @retval ONNXIFI_STATUS_UNIDENTIFIED_NAME The function call failed because
 *                                          the backend does not implement
 *                                          the function identified by the name.
 * @retval ONNXIFI_STATUS_INTERNAL_ERROR The function call failed because the
 *                                       backend experienced an unrecovered
 *                                       internal error.
 * @retval ONNXIFI_STATUS_BACKEND_UNAVAILABLE The function call failed because
 *                                            the backend was disconnected or
 *                                            uninstalled from the system.
 */
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
  onnxGetExtensionFunctionAddress(
    onnxBackendID backendID,
    const char* name,
    onnxExtensionFunctionPointer* function);

/**
 * Create a copy of an ONNXIFI graph with separate input and output buffers.
 *
 * The newly created graph has the same operators, graph structure, and static
 * weights as the original graph, but separate input and output buffers. The
 * newly created graph will reside on the same backend as the original graph.
 * When the copy of the graph is no longer needed, it MUST be released by a
 * call to onnxReleaseGraph.
 *
 * Execution and life-time of the cloned graph is independent from the original.
 * Both graph can be executed simultanously. Calling onnxSetGraphIO on one of
 * the graphs MUST NOT affect the other. Calling onnxReleaseGraph on one of
 * the graphs MUST NOT affect the other. 
 *
 * This function provides a way to share underlying resources between two copies
 * of a graph. Backends which implement this function SHOULD share static
 * weights and other static graph-related resources (e.g. generated code)
 * between the original and the clone.
 * 
 * This function is a part of fb_clone_graph extension. Backends which implement
 * this function MUST list "fb_clone_graph" in the result of onnxGetBackendInfo
 * with ONNXIFI_BACKEND_EXTENSIONS information type. To get function pointer for
 * this function, call onnxGetExtensionFunctionAddress with the backend ID that
 * corresponds to the graph argument and "onnxCloneGraphFB" name, and cast the
 * returned generic extension function pointer to onnxCloneGraphFBFunction.
 *
 * @param graph - graph handle created by either onnxInitGraph or
 *                onnxCloneGraph. The operators, structure, and static weights
 *                of the graph will be shared with the cloned graph.
 * @param[out] graphCopy - pointer to the opaque handle for the cloned ONNXIFI
 *                         graph. If the function fails, and this pointer is
 *                         non-NULL, the handle is initialized to NULL.
 *
 * @retval ONNXIFI_STATUS_SUCCESS The function call succeeded and the graph
 *                                was successfully cloned.
 * @retval ONNXIFI_STATUS_INVALID_GRAPH The function call failed because the
 *                                      graph is not an ONNXIFI graph.
 * @retval ONNXIFI_STATUS_INVALID_POINTER The function call failed because
 *                                        graphCopy pointer is NULL.
 * @retval ONNXIFI_STATUS_NO_SYSTEM_MEMORY The function call failed because the
 *                                         backend could not allocate enough
 *                                         system memory to clone the graph.
 * @retval ONNXIFI_STATUS_NO_SYSTEM_RESOURCES The function call failed due to
 *                                            insufficient non-memory system
 *                                            resources (e.g. file handles) to
 *                                            clone the graph.
 * @retval ONNXIFI_STATUS_NO_DEVICE_MEMORY The function call failed due to
 *                                         insufficient backend-specific memory
 *                                         to clone the graph.
 * @retval ONNXIFI_STATUS_NO_DEVICE_RESOURCES The function call failed due to
 *                                            insufficient non-memory
 *                                            backend-specific resources (e.g.
 *                                            command queues) to clone the
 *                                            graph.
 * @retval ONNXIFI_STATUS_BACKEND_UNAVAILABLE The function call failed because
 *                                            the backend was disconnected or
 *                                            uninstalled from the system.
 * @retval ONNXIFI_STATUS_INTERNAL_ERROR The function call failed because the
 *                                       implementation experienced an
 *                                       unrecovered internal error.
 */
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
  onnxCloneGraphFB(
    onnxGraph graph,
    onnxGraph* graphCopy);

typedef ONNXIFI_CHECK_RESULT onnxStatus
  (ONNXIFI_ABI* onnxCloneGraphFBFunction)(
    onnxGraph graph,
    onnxGraph* graphCopy);

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
 * This function is a part of ext_alternative_model_graph extension.
 * Backends which implement this function MUST list
 * "ext_alternative_model_graph" in the result of onnxGetBackendInfo
 * with ONNXIFI_BACKEND_EXTENSIONS information type. To get function pointer for
 * this function, call onnxGetExtensionFunctionAddress with the backend ID that
 * corresponds to the graph argument and "onnxGetBackendCompatibilityEXT" name,
 * and cast the returned generic extension function pointer to
 * onnxGetBackendCompatibilityEXTFunction.
 *
 * @param backend - ID of the backend to query.
 * @param[in] auxPropertiesList - optional list of backend-specific
 *                                properties, terminated by
 *                                ONNXIFI_BACKEND_PROPERTY_NONE entry. Can be
 *                                NULL or empty.
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
 * @retval ONNXIFI_STATUS_INVALID_PROPERTY The function call failed because one
 *                                         of the backend initialization
 *                                         property values is invalid.
 * @retval ONNXIFI_STATUS_UNSUPPORTED_PROPERTY The function call failed because
 *                                             backend does not recognize one
 *                                             of the initialization
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
  onnxGetBackendCompatibilityEXT(
    onnxBackendID backendID,
    const uint64_t* auxPropertiesList,
    size_t onnxModelSize,
    const void* onnxModel);

typedef ONNXIFI_CHECK_RESULT onnxStatus
  (ONNXIFI_ABI* onnxGetBackendCompatibilityEXTFunction)(
    onnxBackendID backendID,
    const uint64_t* auxPropertiesList,
    size_t onnxModelSize,
    const void* onnxModel);

/**
 * Graph format for the model passed in onnxInitGraph or
 * onnxGetBackendCompatibilityEXT. This enumeration value can be used as a key
 * for auxPropertiesList parameters to specify an alternative (non-ONNX) model
 * format.
 *
 * This enumeration value is a part of ext_alternative_model_graph extension.
 * Backends which implement this function MUST list
 * "ext_alternative_model_graph" in the result of onnxGetBackendInfo
 * with ONNXIFI_BACKEND_EXTENSIONS information type. Users MUST check that this
 * extension is reported by the backend before passing this value to the
 * onnxInitGraph or onnxGetBackendCompatibilityEXT functions.
 *
 * Possible values:
 *     ONNXIFI_GRAPH_FORMAT_ONNX
 *     ONNXIFI_GRAPH_FORMAT_CAFFE2 (only with fb_caffe2_model extension)
 */
#define ONNXIFI_GRAPH_PROPERTY_MODEL_FORMAT 0x100000000ull

/**
 * ONNX file format is used for the model graph passed in onnxInitGraph or
 * onnxGetBackendCompatibilityEXT.
 *
 * This enumeration value is a part of ext_alternative_model_graph extension.
 * Backends which implement this function MUST list
 * "ext_alternative_model_graph" in the result of onnxGetBackendInfo
 * with ONNXIFI_BACKEND_EXTENSIONS information type. Users MUST check that this
 * extension is reported by the backend before passing this value to the
 * onnxInitGraph or onnxGetBackendCompatibilityEXT functions.
 */
#define ONNXIFI_GRAPH_FORMAT_ONNX 0x100000000ull

/**
 * Caffe2 file format is used for the model graph passed in onnxInitGraph or
 * onnxGetBackendCompatibilityEXT.
 *
 * This enumeration value is a part of fb_caffe2_model extension.
 * Backends which implement this function MUST list "fb_caffe2_model" in the
 * result of onnxGetBackendInfo with ONNXIFI_BACKEND_EXTENSIONS information
 * type. Users MUST check that this extension is reported by the backend before
 * passing this value to the onnxInitGraph or onnxGetBackendCompatibilityEXT
 * functions.
 */
#define ONNXIFI_GRAPH_FORMAT_CAFFE2 0x200000000ull

/**
 * Tag for version 1 of Caffe2 quantized tensor descriptor structure
 * (onnxCaffe2QTensorDescriptorV1).
 *
 * The tag is unique for this tensor descriptor structure. If ONNXIFI introduce
 * a new version of the tensor descriptor structure in the future, it will get
 * a new tag value.
 *
 * This tag is a part of fb_caffe2_model extension.
 * Backends which support this tag MUST list "fb_caffe2_model" in the
 * result of onnxGetBackendInfo with ONNXIFI_BACKEND_EXTENSIONS information
 * type. Users MUST check that this extension is reported by the backend before
 * using this tag and onnxCaffe2QTensorDescriptorV1 structure.
 */
#define ONNXIFI_TAG_CAFFE2_QTENSOR_DESCRIPTOR_V1 0x3BC15AFE

/*
 * This structure is a part of fb_caffe2_model extension.
 * Backends which support this structure MUST list "fb_caffe2_model" in the
 * result of onnxGetBackendInfo with ONNXIFI_BACKEND_EXTENSIONS information
 * type. Users MUST check that this extension is reported by the backend before
 * passing this structure to onnxInitGraph or onnxSetGraphIO functions.
 */
typedef struct onnxCaffe2QTensorDescriptorV1 {
  /**
   * 32-bit tag needed to distinguish different versions of a tensor descriptor
   * structure. In the onnxTensorDescriptorV1 structure, the tag MUST be set to
   * ONNXIFI_TAG_CAFFE2_QTENSOR_DESCRIPTOR_V1. If ONNXIFI introduce a new
   * version of the tensor descriptor structure in the future, it WILL have
   * 32-bit tag with a different value as the first member of the structure.
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
   * Base data type of the quantized elements in the tensor.
   *
   * Possible values:
   *     ONNXIFI_DATATYPE_UINT8
   *     ONNXIFI_DATATYPE_INT32
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
   * "Zero point" quantization parameter for the elements of the tensor.
   *
   * Possible values depend on dataType:
   * - For tensors of ONNXIFI_DATATYPE_UINT8 element type, this value must be in
   *   [0, 255] range. 
   * - For tensors of ONNXIFI_DATATYPE_INT32 element type, this value must be 0.
   */
  int32_t zeroPoint;
  /**
   * "Scale" quantization parameter for the elements of the tensor.
   *
   * Scale must be a positive and normal floating-point number.
   */
  float scale;
  /**
   * Pointers to tensor data.
   *
   * Interpretation depends on memoryType:
   *   - ONNXIFI_MEMORY_TYPE_CPU: buffer is a valid pointer to CPU memory.
   *   - ONNXIFI_MEMORY_TYPE_CUDA_BUFFER: buffer is a valid pointer to CUDA
   *     device memory, allocated via cudaMalloc or cuMalloc. CUDA device memory
   *     must be allocated on the same device as the backend.
   */
  onnxPointer buffer;
} onnxCaffe2QTensorDescriptorV1;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* !defined(ONNXIFI_EXT_H) */
