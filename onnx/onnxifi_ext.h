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

/* Function pointer declarations for dynamic loading */
typedef ONNXIFI_CHECK_RESULT onnxStatus
  (ONNXIFI_ABI* onnxGetExtensionFunctionAddressFunction)(
    onnxBackendID backendID,
    const char* name,
    onnxExtensionFunctionPointer* function);

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

/* Extension function pointer declarations for dynamic loading */
typedef ONNXIFI_CHECK_RESULT onnxStatus
  (ONNXIFI_ABI* onnxSetIOAndRunGraphFunction)(
    onnxGraph graph,
    uint32_t inputsCount,
    const onnxTensorDescriptorV1* inputDescriptors,
    uint32_t outputsCount,
    const onnxTensorDescriptorV1* outputDescriptors,
    onnxMemoryFenceV1* outputFence);

/**
 * A combination of onnxSetIO and onnxRunGraph, functionally equals to first run
 * onnxSetIO(graph, inputsCount, inputDescriptors, outputsCount,
 * outputDescriptors), then run onnxRunGraph(graph, inputFence, outputFence)
 * with an internal inputFence.
 *
 * As two separate functions, it is difficult to do atomic evaluation.
 * Therefore, we would like to unify this process and make it evaluable.
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

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI onnxSetIOAndRunGraph(
    onnxGraph graph,
    uint32_t inputsCount,
    const onnxTensorDescriptorV1* inputDescriptors,
    uint32_t outputsCount,
    const onnxTensorDescriptorV1* outputDescriptors,
    onnxMemoryFenceV1* outputFence);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* !defined(ONNXIFI_EXT_H) */
