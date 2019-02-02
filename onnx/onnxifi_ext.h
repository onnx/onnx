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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
