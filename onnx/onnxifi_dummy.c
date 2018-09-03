/*
 * Dummy implementation of ONNX backend interface for manual test.
 * Prints the name of the called function and backend name on each call.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "onnx/onnxifi.h"

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxGetBackendIDs(onnxBackendID* backendIDs, size_t* numBackends) {
  if (backendIDs == NULL || numBackends == NULL) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  *numBackends = 1;
  *backendIDs = 0;
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxReleaseBackendID(onnxBackendID backendID) {
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI onnxGetBackendInfo(
    onnxBackendID backendID,
    onnxBackendInfo infoType,
    void* infoValue,
    size_t* infoValueSize) {
  if (infoValueSize == NULL) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  if (infoValue != NULL) {
    *infoValueSize = *infoValueSize > 10 ? 10 : *infoValueSize;
    memset(infoValue, 0, *infoValueSize);
  } else {
    *infoValueSize = sizeof(uint64_t);
  }
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxGetBackendCompatibility(
    onnxBackendID backendID,
    size_t onnxModelSize,
    const void* onnxModel) {
  if (onnxModel == NULL && onnxModelSize != 0){
	return ONNXIFI_STATUS_INVALID_POINTER;
  }
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI onnxInitBackend(
    onnxBackendID backendID,
    const uint64_t* auxPropertiesList,
    onnxBackend* backend) {
  if (backend == NULL) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  *backend = NULL;
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxReleaseBackend(onnxBackend backend) {
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxInitEvent(onnxBackend backend, onnxEvent* event) {
  if (event == NULL) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  *event = NULL;
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxSignalEvent(onnxEvent event) {
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxWaitEvent(onnxEvent event) {
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxReleaseEvent(onnxEvent event) {
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI onnxInitGraph(
    onnxBackend backend,
    const uint64_t* auxPropertiesList,
    size_t onnxModelSize,
    const void* onnxModel,
    uint32_t weightCount,
    const onnxTensorDescriptorV1* weightDescriptors,
    onnxGraph* graph) {
  if (graph == NULL) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  *graph = NULL;
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI onnxSetGraphIO(
    onnxGraph graph,
    uint32_t inputsCount,
    const onnxTensorDescriptorV1* inputDescriptors,
    uint32_t outputsCount,
    const onnxTensorDescriptorV1* outputDescriptors) {
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI onnxRunGraph(
    onnxGraph graph,
    const onnxMemoryFenceV1* inputFence,
    onnxMemoryFenceV1* outputFence) {
  if (outputFence == NULL) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  outputFence->type = ONNXIFI_SYNCHRONIZATION_EVENT;
  outputFence->event = NULL;
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxGetEventState(onnxEvent event, onnxEventState* state) {
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxReleaseGraph(onnxGraph graph) {
  return ONNXIFI_STATUS_SUCCESS;
}
