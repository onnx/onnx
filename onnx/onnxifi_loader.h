#ifndef ONNXIFI_LOADER_H
#define ONNXIFI_LOADER_H 1

#include "onnx/onnxifi.h"

#define ONNXIFI_LOADER_FUNCTION_COUNT 15

#define ONNXIFI_LOADER_FLAG_VERSION_MASK 0xFF
#define ONNXIFI_LOADER_FLAG_VERSION_1_0 0x01

#ifndef ONNXIFI_HIDDEN
#if defined(__ELF__)
#define ONNXIFI_HIDDEN __attribute__((__visibility__("hidden")))
#elif defined(__MACH__)
#define ONNXIFI_HIDDEN __attribute__((__visibility__("hidden")))
#else
#define ONNXIFI_HIDDEN
#endif
#endif /* !defined(ONNXIFI_HIDDEN) */

struct onnxifi_library {
  /*
   * Opaque handle for the loaded ONNXIFI library.
   *
   * Note: this is the value returned from LoadLibraryW (on Windows), or
   * dlopen (on other operating systems and environments).
   */
  void* handle;
  /*
   * Options used for dynamic loading of the ONNXIFI library and its API
   * functions. These are the options passed in flags parameter to onnxifi_load.
   */
  uint32_t flags;
  union {
    struct {
      onnxGetBackendIDsFunction onnxGetBackendIDs;
      onnxReleaseBackendIDFunction onnxReleaseBackendID;
      onnxGetBackendInfoFunction onnxGetBackendInfo;
      onnxGetBackendCompatibilityFunction onnxGetBackendCompatibility;
      onnxInitBackendFunction onnxInitBackend;
      onnxReleaseBackendFunction onnxReleaseBackend;
      onnxInitEventFunction onnxInitEvent;
      onnxSignalEventFunction onnxSignalEvent;
      onnxGetEventStateFunction onnxGetEventState;
      onnxWaitEventFunction onnxWaitEvent;
      onnxReleaseEventFunction onnxReleaseEvent;
      onnxInitGraphFunction onnxInitGraph;
      onnxSetGraphIOFunction onnxSetGraphIO;
      onnxRunGraphFunction onnxRunGraph;
      onnxReleaseGraphFunction onnxReleaseGraph;
    };
    void* functions[ONNXIFI_LOADER_FUNCTION_COUNT];
  };
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Dynamically load the ONNXIFI library.
 *
 * @param flags - options for dynamic loading of the ONNXIFI library.
 *                The ONNXIFI_LOADER_FLAG_VERSION_MASK part of the flags
 *                specifies which ONNX API function should be loaded. The only
 *                currently supported value is ONNXIFI_LOADER_FLAG_VERSION_1_0.
 * @param path - optional path to the ONNXIFI library to load.
 *               If this argument is null, the default path is used.
 * @param[out] library - the structure representing the dynamic library and
 *                       its API functions. On success, this structure will
 *                       be initialized with valid pointers to implentation.
 *                       On failure, this structure will be zero-initialized.
 *
 * @return Non-zero if the function succeeds, or zero if the function fails.
 */
ONNXIFI_HIDDEN int ONNXIFI_ABI onnxifi_load(
  uint32_t flags,
#ifdef _WIN32
  const wchar_t* path,
#else
  const char* path,
#endif
  struct onnxifi_library* library);

/**
 * Unload the dynamically loaded ONNXIFI library.
 *
 * @param[in,out] library - the structure representing the dynamic library and
 *                          its API functions. If this structure is
 *                          zero-initialized, the function does nothing.
 *                          The function zero-initialized the structure before
 *                          returning.
 */
ONNXIFI_HIDDEN void ONNXIFI_ABI onnxifi_unload(
  struct onnxifi_library* library);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* !defined(ONNXIFI_LOADER_H) */
