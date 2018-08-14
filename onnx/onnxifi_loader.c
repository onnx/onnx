#ifdef _WIN32
#include <windows.h>
#else
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dlfcn.h>
#endif

#include "onnx/onnxifi_loader.h"

/* ONNXIFI_LOADER_LOGGING macro enables/disables logging. Its OFF by default. */
#ifndef ONNXIFI_LOADER_LOGGING
#define ONNXIFI_LOADER_LOGGING 0
#endif

#if ONNXIFI_LOADER_LOGGING
#if defined(__ANDROID__)
#include <android/log.h>
/* Tag used for logging on Android */
#define ONNXIFI_LOADER_ANDROID_LOG_TAG "ONNXIFI-LOADER"
#else
#include <stdio.h>
#endif
#endif

#if defined(__APPLE__)
#define ONNXIFI_LIBRARY_NAME "libonnxifi.dylib"
#elif defined(_WIN32)
#define ONNXIFI_LIBRARY_NAME L"onnxifi.dll"
#else
#define ONNXIFI_LIBRARY_NAME "libonnxifi.so"
#endif

/* Order must match declaration order in onnxifi_library structure */
static const char onnxifi_function_names[] =
    "onnxGetBackendIDs\0"
    "onnxReleaseBackendID\0"
    "onnxGetBackendInfo\0"
    "onnxGetBackendCompatibility\0"
    "onnxInitBackend\0"
    "onnxReleaseBackend\0"
    "onnxInitEvent\0"
    "onnxSignalEvent\0"
    "onnxGetEventState\0"
    "onnxWaitEvent\0"
    "onnxReleaseEvent\0"
    "onnxInitGraph\0"
    "onnxSetGraphIO\0"
    "onnxRunGraph\0"
    "onnxReleaseGraph\0";

int ONNXIFI_ABI onnxifi_load(
  uint32_t flags,
#ifdef _WIN32
  const wchar_t* path,
#else
  const char* path,
#endif
  struct onnxifi_library* onnx)
{
  size_t i;
  const char* function_name;

  if (onnx == NULL) {
    return 0;
  }

#ifdef _WIN32
  ZeroMemory(onnx, sizeof(struct onnxifi_library));
#else
  memset(onnx, 0, sizeof(struct onnxifi_library));
#endif
  if (!(flags & ONNXIFI_LOADER_FLAG_VERSION_1_0)) {
    /* Unknown ONNXIFI version requested */
    return 0;
  }

  if (path == NULL) {
    path = ONNXIFI_LIBRARY_NAME;
  }

#ifdef _WIN32
  onnx->handle = (void*) LoadLibraryExW(path, NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
  /* Clear libdl error state */
  dlerror();
  onnx->handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
  if (onnx->handle == NULL) {
#if ONNXIFI_LOADER_LOGGING
#if defined(__ANDROID__)
    __android_log_print(
      ANDROID_LOG_ERROR,
      ONNXIFI_LOADER_ANDROID_LOG_TAG,
      "failed to load %s: %s",
      path, dlerror());
#elif defined(_WIN32)
    fprintf(
      stderr,
      "Error: failed to load %S: error %u\n",
      path, (unsigned long) GetLastError());
#else
    fprintf(stderr, "Error: failed to load %s: %s\n",
      path, dlerror());
#endif
#endif /* ONNXIFI_LOADER_LOGGING */

    goto failed;
  }

  function_name = onnxifi_function_names;
  for (i = 0; i < ONNXIFI_LOADER_FUNCTION_COUNT; i++) {
#ifdef _WIN32
    onnx->functions[i] = GetProcAddress((HMODULE) onnx->handle, function_name);
#else
    onnx->functions[i] = dlsym(onnx->handle, function_name);
#endif

    if (onnx->functions[i] == NULL) {
#if ONNXIFI_LOADER_LOGGING
#if defined(__ANDROID__)
      __android_log_print(
        ANDROID_LOG_ERROR,
        ONNXIFI_LOADER_ANDROID_LOG_TAG,
        "failed to find function %s in %s: %s",
        function_name,
        ONNXIFI_LIBRARY_NAME,
        dlerror());
#elif defined(_WIN32)
      fprintf(
        stderr,
        "Error: failed to find function %s in %s: error %u\n",
        function_name,
        ONNXIFI_LIBRARY_NAME,
        (unsigned long) GetLastError());
#else
      fprintf(
        stderr,
        "Error: failed to find function %s in %s: %s\n",
        function_name,
        ONNXIFI_LIBRARY_NAME,
        dlerror());
#endif
#endif /* ONNXIFI_LOADER_LOGGING */

      goto failed;
    }
#ifdef _WIN32
    function_name += lstrlenA(function_name);
#else
    function_name += strlen(function_name);
#endif
    /* Skip null-terminator */
    function_name += 1;
  }

  onnx->flags = flags & ONNXIFI_LOADER_FLAG_VERSION_MASK;
  return 1;

failed:
  onnxifi_unload(onnx);
  return 0;
}

void ONNXIFI_ABI onnxifi_unload(struct onnxifi_library* onnx) {
  if (onnx != NULL) {
    if (onnx->handle != NULL) {
#ifdef _WIN32
      if (FreeLibrary((HMODULE) onnx->handle) == FALSE) {
#if ONNXIFI_LOADER_LOGGING
        fprintf(
          stderr,
          "Error: failed to unload library %s: error %u\n",
          ONNXIFI_LIBRARY_NAME,
          (unsigned long) GetLastError());
#endif /* ONNXIFI_LOADER_LOGGING */
      }
#else /* !defined(_WIN32) */
      /* Clear libdl error state */
      dlerror();
      if (dlclose(onnx->handle) != 0) {
#if ONNXIFI_LOADER_LOGGING
#if defined(__ANDROID__)
        __android_log_print(
          ANDROID_LOG_ERROR,
          ONNXIFI_LOADER_ANDROID_LOG_TAG,
          "failed to unload %s: %s",
          ONNXIFI_LIBRARY_NAME,
          dlerror());
#else
        fprintf(
          stderr,
          "Error: failed to unload %s: %s\n",
          ONNXIFI_LIBRARY_NAME,
          dlerror());
#endif
#endif /* ONNXIFI_LOADER_LOGGING */
      }
#endif /* !defined(_WIN32) */
    }
#ifdef _WIN32
    ZeroMemory(onnx, sizeof(struct onnxifi_library));
#else
    memset(onnx, 0, sizeof(struct onnxifi_library));
#endif
  }
}
