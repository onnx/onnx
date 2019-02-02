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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
