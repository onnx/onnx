/*
 * ONNX wrapper api discovers vendor-specific implementations installed in
 * the system and exposes them under a single interface.
 *
 * 
 * 
 */

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#ifdef _WIN32
  #include <windows.h>
#else
  #include <string.h>
  #include <pthread.h>
  #include <dirent.h>
  #include <errno.h>
#endif

#include <onnx/onnxifi.h>
#include <onnx/onnxifi_loader.h>

#if defined(_WIN32)
#define ONNXIFI_FILENAME_WILDCARD L"\\onnxifi-*.dll"
#define ONNXIFI_FILENAME_WILDCARD_LENGTH 14
#elif defined(__APPLE__)
/* Minimum filename: "libonnxifi-?.dylib" */
#define ONNXIFI_FILENAME_MIN 18
#define ONNXIFI_FILENAME_PREFIX "libonnxifi-"
#define ONNXIFI_FILENAME_SUFFIX ".dylib"
#else
/* Minimum filename: "libonnxifi-?.so" */
#define ONNXIFI_FILENAME_MIN 15
#define ONNXIFI_FILENAME_PREFIX "libonnxifi-"
#define ONNXIFI_FILENAME_SUFFIX ".so"
#endif

#define ONNXIFI_BACKEND_ID_MAGIC UINT32_C(0x2EDD3764)
#define ONNXIFI_BACKEND_MAGIC    UINT32_C(0x4B9B2902)
#define ONNXIFI_GRAPH_MAGIC      UINT32_C(0xD9ACFACD)
#define ONNXIFI_EVENT_MAGIC      UINT32_C(0x18C1D735)

struct onnxifi_backend_id_wrapper {
  uint32_t magic;
  onnxBackendID backend_id;
  struct onnxifi_library* library;
};

struct onnxifi_backend_wrapper {
  uint32_t magic;
  onnxBackend backend;
  struct onnxifi_library* library;
};

struct onnxifi_graph_wrapper {
  uint32_t magic;
  onnxGraph graph;
  struct onnxifi_library* library;
};

struct onnxifi_event_wrapper {
  uint32_t magic;
  onnxEvent event;
  struct onnxifi_library* library;
};

static struct onnxifi_library* libraries = NULL;
static uint32_t num_libraries = 0;

#ifdef _WIN32
static INIT_ONCE init_guard = INIT_ONCE_STATIC_INIT;

static BOOL CALLBACK load_all_windows_backends(
  PINIT_ONCE init_once,
  PVOID parameter,
  PVOID* context)
{
  WCHAR* onnxifi_library_wildcard = NULL;
  HANDLE find_file_handle = INVALID_HANDLE_VALUE;
  WIN32_FIND_DATAW find_file_data;

  UINT system_directory_path_length = GetSystemDirectoryW(NULL, 0);
  if (system_directory_path_length == 0) {
    fprintf(stderr, "Error: failed to get system directory path: %u\n",
      (unsigned int) GetLastError());
    goto cleanup;
  }

  onnxifi_library_wildcard = malloc(sizeof(WCHAR) *
    (system_directory_path_length + ONNXIFI_FILENAME_WILDCARD_LENGTH + 1));
  if (onnxifi_library_wildcard == NULL) {
    fprintf(stderr,
      "Error: failed to allocate %Iu bytes for ONNXIFI path\n",
      sizeof(WCHAR) *
      (system_directory_path_length + ONNXIFI_FILENAME_WILDCARD_LENGTH + 1));
    goto cleanup;
  }

  if (GetSystemDirectoryW(
      onnxifi_library_wildcard, system_directory_path_length) == 0)
  {
    fprintf(stderr, "Error: failed to get system directory path: %u\n",
      (unsigned int) GetLastError());
    goto cleanup;
  }

  memcpy(onnxifi_library_wildcard + system_directory_path_length,
    ONNXIFI_FILENAME_WILDCARD,
    sizeof(WCHAR) * (ONNXIFI_FILENAME_WILDCARD_LENGTH + 1));

  find_file_handle = FindFirstFileW(onnxifi_library_wildcard, &find_file_data);
  if (find_file_handle == INVALID_HANDLE_VALUE) {
    const DWORD error = GetLastError();
    if (error != ERROR_FILE_NOT_FOUND) {
      fprintf(stderr,
        "Error: failed to list ONNXIFI libraries %S: error %u\n",
        onnxifi_library_wildcard, (unsigned int) error);
    }
    goto cleanup;
  }

  for (;;) {
    struct onnxifi_library library;
    if (!onnxifi_load(ONNXIFI_LOADER_FLAG_VERSION_1_0,
        find_file_data.cFileName, &library))
    {
      fprintf(stderr, "Error: failed to load library %S\n",
        find_file_data.cFileName);
      continue;
    }

    struct onnxifi_library* new_libraries =
      realloc(libraries, (num_libraries + 1) * sizeof(struct onnxifi_library));
    if (new_libraries == NULL) {
      fprintf(stderr, "Error: failed to allocate space for library %S\n",
        find_file_data.cFileName);
      onnxifi_unload(&library);
      continue;
    }

    /* All actions for the new library succeeded, commit changes */
    libraries = new_libraries;
    memcpy(&libraries[num_libraries], &library, sizeof(library));
    num_libraries++;

    if (FindNextFileW(find_file_handle, &find_file_data) != FALSE) {
      const DWORD error = GetLastError();
      if (error != ERROR_NO_MORE_FILES) {
        fprintf(stderr,
          "Error: failed to some of ONNXIFI libraries %S: error %u\n",
          onnxifi_library_wildcard, (unsigned int) error);
      }
      break;
    }
  }

cleanup:
  if (find_file_handle != INVALID_HANDLE_VALUE) {
    FindClose(find_file_handle);
  }
  free(onnxifi_library_wildcard);
  return TRUE;
}
#else
static pthread_once_t init_guard = PTHREAD_ONCE_INIT;

#ifndef ONNXIFI_SEARCH_DIR
#ifdef __APPLE__
#define ONNXIFI_SEARCH_DIR "/opt/onnx/lib/"
#else
#define ONNXIFI_SEARCH_DIR "/usr/lib/"
#endif
#endif

/* Finds filename in a null-terminated file path */
static inline const char* find_filename(const char* filepath) {
  const char* filename_separator = strrchr(filepath, '/');
  if (filename_separator == NULL) {
    return filepath;
  } else {
    return filename_separator + 1;
  }
}

static inline bool is_onnxifi_path(const char* filepath) {
  const char* filename = find_filename(filepath);

  const size_t filename_length = strlen(filename);
  if (filename_length < ONNXIFI_FILENAME_MIN) {
    /* Filename too short */
    return false;
  }
  const char* filename_end = filename + filename_length;

  /* Expected filename structure: <prefix><backend><suffix> */
  if (memcmp(filename,
      ONNXIFI_FILENAME_PREFIX,
      strlen(ONNXIFI_FILENAME_PREFIX)) != 0)
  {
    /* Prefix mismatch */
    return false;
  }

  const char* suffix = filename_end - strlen(ONNXIFI_FILENAME_SUFFIX);
  if (memcmp(suffix,
      ONNXIFI_FILENAME_SUFFIX,
      strlen(ONNXIFI_FILENAME_SUFFIX)) != 0)
  {
    /* Suffix mismatch */
    return false;
  }

  return true;
}

static void load_all_posix_backends(void) {
  DIR* directory = opendir(ONNXIFI_SEARCH_DIR);
  if (directory == NULL) {
    fprintf(stderr, "Error: failed to open directory %s: %s\n",
      ONNXIFI_SEARCH_DIR, strerror(errno));
    return;
  }

  for (;;) {
    /* Required to distinguish between error and end of directory in readdir */
    errno = 0;

    struct dirent* entry = readdir(directory);
    if (entry == NULL) {
      if (errno != 0) {
        fprintf(stderr, "Error: failed to read directory %s: %s\n",
          ONNXIFI_SEARCH_DIR, strerror(errno));
      }
      goto cleanup;
    }

    if (!is_onnxifi_path(entry->d_name)) {
      continue;
    }

    struct onnxifi_library library;
    if (!onnxifi_load(ONNXIFI_LOADER_FLAG_VERSION_1_0, entry->d_name, &library))
    {
      fprintf(stderr, "Error: failed to load library %s\n", entry->d_name);
      continue;
    }

    struct onnxifi_library* new_libraries =
      realloc(libraries, (num_libraries + 1) * sizeof(struct onnxifi_library));
    if (new_libraries == NULL) {
      fprintf(stderr, "Error: failed to allocate space for library %s\n",
        entry->d_name);
      onnxifi_unload(&library);
      continue;
    }

    /* All actions for the new library succeeded, commit changes */
    libraries = new_libraries;
    memcpy(&libraries[num_libraries], &library, sizeof(library));
    num_libraries++;
  }

cleanup:
  if (directory != NULL) {
    if (closedir(directory) != 0) {
      /* Error */
      fprintf(stderr, "Warning: failed to close directory %s: %s\n",
        ONNXIFI_SEARCH_DIR, strerror(errno));
    }
  }
}
#endif

/* Platform-independent wrapper for init_once */
static void load_all_libraries_once(void) {
#ifdef _WIN32
  InitOnceExecuteOnce(&init_guard, load_all_windows_backends, NULL, NULL);
#else
  pthread_once(&init_guard, load_all_posix_backends);
#endif
}

static onnxStatus wrap_backend_ids(
  struct onnxifi_library* library,
  size_t num_backends,
  onnxBackendID* backend_ids)
{
  size_t num_wrapped_backends = 0;
  for (; num_wrapped_backends < num_backends; num_wrapped_backends++) {
    struct onnxifi_backend_id_wrapper* backend_id_wrapper =
      (struct onnxifi_backend_id_wrapper*)
        malloc(sizeof(struct onnxifi_backend_id_wrapper));
    if (backend_id_wrapper == NULL) {
      goto cleanup;
    }

    backend_id_wrapper->magic      = ONNXIFI_BACKEND_ID_MAGIC;
    backend_id_wrapper->backend_id = backend_ids[num_wrapped_backends];
    backend_id_wrapper->library    = library;

    /* Replace backend ID with its wrapper */
    backend_ids[num_wrapped_backends] = (onnxBackendID) backend_id_wrapper;
  }
  return ONNXIFI_STATUS_SUCCESS;

cleanup:
  /* Unwrap all the backends */
  for (size_t i = 0; i < num_wrapped_backends; i++) {
    struct onnxifi_backend_id_wrapper* backend_id_wrapper =
      (struct onnxifi_backend_id_wrapper*) backend_ids[i];
    assert(backend_id_wrapper->magic == ONNXIFI_BACKEND_ID_MAGIC);

    /* Replace wrapper with the wrapped backend ID */
    backend_ids[i] = backend_id_wrapper->backend_id;

    /* Safety precaution to avoid use-after-free bugs */
    memset(backend_id_wrapper, 0, sizeof(struct onnxifi_backend_id_wrapper));
    free(backend_id_wrapper);
  }
  return ONNXIFI_STATUS_NO_SYSTEM_MEMORY;
}

ONNXIFI_PUBLIC onnxStatus ONNXIFI_ABI
  onnxGetBackendIDs(
    onnxBackendID* backendIDs,
    size_t* numBackends)
{
  load_all_libraries_once();
  onnxStatus status;

  /* Number of backend IDs requested to be stored by the caller */
  const size_t num_expected_ids = (backendIDs == NULL) ? 0 : *numBackends;
  /* Number of backend IDs in the system */
  size_t num_available_ids = 0;
  /* Number of backend IDs wrapped and ready to return */
  size_t num_wrapped_ids = 0;
  onnxBackendID* backend_ids = NULL;
  if (num_expected_ids != 0) {
    backend_ids = malloc(num_expected_ids * sizeof(onnxBackendID));
    if (backend_ids == NULL) {
      status = ONNXIFI_STATUS_NO_SYSTEM_MEMORY;
      goto error;
    }
  }

  /* Safety precaution to avoid dangling pointer bugs */
  memset(backend_ids, 0, num_expected_ids * sizeof(onnxBackendID));
  for (size_t l = 0; l < num_libraries; l++) {
    if (num_expected_ids > num_available_ids) {
      /* Query and wrap backend IDs from ONNXIFI library */
      const size_t max_library_ids = num_expected_ids - num_available_ids;
      size_t num_library_ids = max_library_ids;
      status = libraries[l].onnxGetBackendIDs(
        &backend_ids[num_available_ids],
        &num_library_ids);
      const size_t num_stored_ids =
        (num_library_ids < max_library_ids) ? num_library_ids : max_library_ids;
      switch (status) {
        case ONNXIFI_STATUS_SUCCESS:
        case ONNXIFI_STATUS_FALLBACK:
          status = wrap_backend_ids(
            &libraries[l],
            num_stored_ids,
            &backend_ids[num_available_ids]);
          if (status != ONNXIFI_STATUS_SUCCESS) {
            /* Release unwrapped backends for this library */
            for (size_t i = 0; i < num_stored_ids; i++) {
              (void) libraries[l].onnxReleaseBackendID(backend_ids[i]);

              /* Safety precaution to avoid use-after-free bugs */
              backend_ids[i] = NULL;
            }
            /* Release wrapped backends for other libraries */
            goto error;
          }

          num_wrapped_ids += num_stored_ids;
          num_available_ids += num_library_ids;
          break;
        default:
          /* Release wrapped backends for other libraries */
          goto error;
      }
    } else {
      /* Not enough space in user-provided buffer: only count the backend IDs */
      size_t num_library_ids = 0;
      status = libraries[l].onnxGetBackendIDs(NULL, &num_library_ids);
      if (status != ONNXIFI_STATUS_FALLBACK) {
        /* Release wrapped backends for other libraries */
        goto error;
      }

      num_available_ids += num_library_ids;
    }
  }

  /*
   * Successful return:
   * - Copy backend IDs to user-provided memory (if applicable)
   * - Store number of backend IDs to user-provided variable
   * - Return success or fallback codes
   */
  if (backendIDs == NULL) {
    *numBackends = num_available_ids;
    return ONNXIFI_STATUS_FALLBACK;
  } else {
    memcpy(backendIDs, backend_ids, num_wrapped_ids * sizeof(onnxBackendID));
    free(backend_ids);
    *numBackends = num_available_ids;
    return num_available_ids <= num_expected_ids ?
      ONNXIFI_STATUS_SUCCESS : ONNXIFI_STATUS_FALLBACK;
  }

error:
  /*
   * Error return:
   * - Release all wrapped backend IDs.
   * - Deallocate wrapper structures.
   */
  for (size_t i = 0; i < num_wrapped_ids; i++) {
    struct onnxifi_backend_id_wrapper* backend_id_wrapper =
      (struct onnxifi_backend_id_wrapper*) backend_ids[i];
    assert(backend_id_wrapper->magic == ONNXIFI_BACKEND_ID_MAGIC);

    (void) backend_id_wrapper->library->onnxReleaseBackendID(
      backend_id_wrapper->backend_id);

    /* Safety precaution to avoid use-after-free bugs */
    memset(backend_id_wrapper, 0, sizeof(struct onnxifi_backend_id_wrapper));
    free(backend_id_wrapper);
  }
  free(backend_ids);
  return status;
}

ONNXIFI_PUBLIC onnxStatus ONNXIFI_ABI
  onnxReleaseBackendID(
    onnxBackendID backendID)
{
  if (backendID == NULL) {
    return ONNXIFI_STATUS_INVALID_ID;
  }

  struct onnxifi_backend_id_wrapper* backend_id_wrapper =
    (struct onnxifi_backend_id_wrapper*) backendID;
  if (backend_id_wrapper->magic != ONNXIFI_BACKEND_ID_MAGIC) {
    return ONNXIFI_STATUS_INVALID_ID;
  }
  if (backend_id_wrapper->library == NULL) {
    return ONNXIFI_STATUS_INVALID_ID;
  }

  /* Call onnxReleaseBackendID with unwrapped backend ID */
  const onnxStatus status =
    backend_id_wrapper->library->onnxReleaseBackendID(
      backend_id_wrapper->backend_id);

  /*
   * Note: ReleaseBackendID either succeeded, or failed with internal error.
   * Either way, it is not safe to use the backend ID again.
   */
  memset(backend_id_wrapper, 0, sizeof(struct onnxifi_backend_id_wrapper));
  free(backend_id_wrapper);

  return status;
}

ONNXIFI_PUBLIC onnxStatus ONNXIFI_ABI
  onnxGetBackendInfo(
    onnxBackendID backendID,
    onnxBackendInfo infoType,
    void* infoValue,
    size_t* infoValueSize)
{
  if (backendID == NULL) {
    return ONNXIFI_STATUS_INVALID_ID;
  }

  const struct onnxifi_backend_id_wrapper* backend_id_wrapper =
    (const struct onnxifi_backend_id_wrapper*) backendID;
  if (backend_id_wrapper->magic != ONNXIFI_BACKEND_ID_MAGIC) {
    return ONNXIFI_STATUS_INVALID_ID;
  }
  if (backend_id_wrapper->library == NULL) {
    return ONNXIFI_STATUS_INVALID_ID;
  }

  /* Call onnxGetBackendInfo with unwrapped backend ID */
  return backend_id_wrapper->library->onnxGetBackendInfo(
    backend_id_wrapper->backend_id,
    infoType,
    infoValue,
    infoValueSize);
}

ONNXIFI_PUBLIC onnxStatus ONNXIFI_ABI
  onnxGetBackendCompatibility(
    onnxBackendID backendID,
    size_t onnxModelSize,
    const void* onnxModel)
{
  if (backendID == NULL) {
    return ONNXIFI_STATUS_INVALID_ID;
  }

  const struct onnxifi_backend_id_wrapper* backend_id_wrapper =
    (const struct onnxifi_backend_id_wrapper*) backendID;
  if (backend_id_wrapper->magic != ONNXIFI_BACKEND_ID_MAGIC) {
    return ONNXIFI_STATUS_INVALID_ID;
  }
  if (backend_id_wrapper->library == NULL) {
    return ONNXIFI_STATUS_INVALID_ID;
  }

  /* Call onnxGetBackendCompatibility with unwrapped backend ID */
  return backend_id_wrapper->library->onnxGetBackendCompatibility(
    backend_id_wrapper->backend_id,
    onnxModelSize,
    onnxModel);
}

ONNXIFI_PUBLIC onnxStatus ONNXIFI_ABI
  onnxInitBackend(
    onnxBackendID backendID,
    const uint64_t* auxPropertiesList,
    onnxBackend* backend)
{
  if (backend == NULL) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  *backend = NULL;

  if (backendID == NULL) {
    return ONNXIFI_STATUS_INVALID_ID;
  }

  const struct onnxifi_backend_id_wrapper* backend_id_wrapper =
    (const struct onnxifi_backend_id_wrapper*) backendID;
  if (backend_id_wrapper->magic != ONNXIFI_BACKEND_ID_MAGIC) {
    return ONNXIFI_STATUS_INVALID_ID;
  }
  if (backend_id_wrapper->library == NULL) {
    return ONNXIFI_STATUS_INVALID_ID;
  }

  struct onnxifi_backend_wrapper* backend_wrapper =
    (struct onnxifi_backend_wrapper*)
      malloc(sizeof(struct onnxifi_backend_wrapper));
  if (backend_wrapper == NULL) {
    return ONNXIFI_STATUS_NO_SYSTEM_MEMORY;
  }
  memset(backend_wrapper, 0, sizeof(struct onnxifi_backend_wrapper));

  /* Call onnxInitBackend with unwrapped backend ID */
  const onnxStatus status = backend_id_wrapper->library->onnxInitBackend(
    backend_id_wrapper->backend_id,
    auxPropertiesList,
    &backend_wrapper->backend);
  if (status == ONNXIFI_STATUS_SUCCESS) {
    /* Success, return wrapped graph */
    backend_wrapper->magic = ONNXIFI_BACKEND_MAGIC;
    backend_wrapper->library = backend_id_wrapper->library;
    *backend = (onnxBackend) backend_wrapper;
    return ONNXIFI_STATUS_SUCCESS;
  } else {
    /* Failure, release allocated memory */
    free(backend_wrapper);
    return status;
  }
}

ONNXIFI_PUBLIC onnxStatus ONNXIFI_ABI
  onnxReleaseBackend(
    onnxBackend backend)
{
  if (backend == NULL) {
    return ONNXIFI_STATUS_INVALID_BACKEND;
  }

  struct onnxifi_backend_wrapper* backend_wrapper =
    (struct onnxifi_backend_wrapper*) backend;
  if (backend_wrapper->magic != ONNXIFI_BACKEND_MAGIC) {
    return ONNXIFI_STATUS_INVALID_BACKEND;
  }
  if (backend_wrapper->library == NULL) {
    return ONNXIFI_STATUS_INVALID_BACKEND;
  }

  /* Call onnxReleaseBackend with unwrapped backend handle */
  const onnxStatus status = backend_wrapper->library->onnxReleaseBackend(
    backend_wrapper->backend);

  /*
   * Note: ReleaseBackend either succeeded, or failed with internal error.
   * Either way, it is not safe to use the backend handle again.
   */
  memset(backend_wrapper, 0, sizeof(struct onnxifi_backend_wrapper));
  free(backend_wrapper);

  return status;
}

ONNXIFI_PUBLIC onnxStatus ONNXIFI_ABI
  onnxInitEvent(
    onnxBackend backend,
    onnxEvent* event)
{
  if (event == NULL) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  *event = NULL;

  if (backend == NULL) {
    return ONNXIFI_STATUS_INVALID_BACKEND;
  }

  const struct onnxifi_backend_wrapper* backend_wrapper =
    (const struct onnxifi_backend_wrapper*) backend;
  if (backend_wrapper->magic != ONNXIFI_BACKEND_MAGIC) {
    return ONNXIFI_STATUS_INVALID_BACKEND;
  }
  if (backend_wrapper->library == NULL) {
    return ONNXIFI_STATUS_INVALID_BACKEND;
  }

  struct onnxifi_event_wrapper* event_wrapper =
    (struct onnxifi_event_wrapper*)
      malloc(sizeof(struct onnxifi_event_wrapper));
  if (event_wrapper == NULL) {
    return ONNXIFI_STATUS_NO_SYSTEM_MEMORY;
  }
  memset(event_wrapper, 0, sizeof(struct onnxifi_event_wrapper));

  /* Call onnxInitEvent with unwrapped backend handle */
  const onnxStatus status = backend_wrapper->library->onnxInitEvent(
    backend_wrapper->backend,
    &event_wrapper->event);
  if (status == ONNXIFI_STATUS_SUCCESS) {
    /* Success, return wrapped graph */
    event_wrapper->magic = ONNXIFI_EVENT_MAGIC;
    event_wrapper->library = backend_wrapper->library;
    *event = (onnxEvent) event_wrapper;
    return ONNXIFI_STATUS_SUCCESS;
  } else {
    /* Failure, release allocated memory */
    free(event_wrapper);
    return status;
  }
}

ONNXIFI_PUBLIC onnxStatus ONNXIFI_ABI
  onnxSignalEvent(
    onnxEvent event)
{
  if (event == NULL) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }

  const struct onnxifi_event_wrapper* event_wrapper =
    (const struct onnxifi_event_wrapper*) event;
  if (event_wrapper->magic != ONNXIFI_EVENT_MAGIC) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }
  if (event_wrapper->library == NULL) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }

  /* Call onnxSignalEvent with unwrapped backend handle */
  return event_wrapper->library->onnxSignalEvent(
    event_wrapper->event);
}

ONNXIFI_PUBLIC onnxStatus ONNXIFI_ABI
  onnxGetEventState(
    onnxEvent event,
    onnxEventState* state)
{
  if (state == NULL) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  *state = ONNXIFI_EVENT_STATE_INVALID;

  if (event == NULL) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }

  const struct onnxifi_event_wrapper* event_wrapper =
    (const struct onnxifi_event_wrapper*) event;
  if (event_wrapper->magic != ONNXIFI_EVENT_MAGIC) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }
  if (event_wrapper->library == NULL) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }

  /* Call onnxGetEventState with unwrapped backend handle */
  return event_wrapper->library->onnxGetEventState(
    event_wrapper->event, state);
}

ONNXIFI_PUBLIC onnxStatus ONNXIFI_ABI
  onnxWaitEvent(
    onnxEvent event)
{
  if (event == NULL) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }

  const struct onnxifi_event_wrapper* event_wrapper =
    (const struct onnxifi_event_wrapper*) event;
  if (event_wrapper->magic != ONNXIFI_EVENT_MAGIC) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }
  if (event_wrapper->library == NULL) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }

  /* Call onnxWaitEvent with unwrapped backend handle */
  return event_wrapper->library->onnxWaitEvent(
    event_wrapper->event);
}

ONNXIFI_PUBLIC onnxStatus ONNXIFI_ABI
  onnxReleaseEvent(
    onnxEvent event)
{
  if (event == NULL) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }

  struct onnxifi_event_wrapper* event_wrapper =
    (struct onnxifi_event_wrapper*) event;
  if (event_wrapper->magic != ONNXIFI_EVENT_MAGIC) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }
  if (event_wrapper->library == NULL) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }

  /* Call onnxReleaseEvent with unwrapped event handle */
  const onnxStatus status = event_wrapper->library->onnxReleaseEvent(
    event_wrapper->event);

  /*
   * Note: ReleaseEvent either succeeded, or failed with internal error.
   * Either way, it is not safe to use the event handle again.
   */
  memset(event_wrapper, 0, sizeof(struct onnxifi_event_wrapper));
  free(event_wrapper);

  return status;
}

ONNXIFI_PUBLIC onnxStatus ONNXIFI_ABI onnxInitGraph(
    onnxBackend backend,
    const uint64_t* auxPropertiesList,
    size_t onnxModelSize,
    const void* onnxModel,
    uint32_t weightsCount,
    const onnxTensorDescriptorV1* weightDescriptors,
    onnxGraph* graph)
{
  if (graph == NULL) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  *graph = NULL;

  if (backend == NULL) {
    return ONNXIFI_STATUS_INVALID_BACKEND;
  }

  struct onnxifi_backend_wrapper* backend_wrapper =
    (struct onnxifi_backend_wrapper*) backend;
  if (backend_wrapper->magic != ONNXIFI_BACKEND_MAGIC) {
    return ONNXIFI_STATUS_INVALID_BACKEND;
  }
  if (backend_wrapper->library == NULL) {
    return ONNXIFI_STATUS_INVALID_BACKEND;
  }

  struct onnxifi_graph_wrapper* graph_wrapper =
    (struct onnxifi_graph_wrapper*)
      malloc(sizeof(struct onnxifi_graph_wrapper));
  if (graph_wrapper == NULL) {
    return ONNXIFI_STATUS_NO_SYSTEM_MEMORY;
  }
  memset(graph_wrapper, 0, sizeof(struct onnxifi_graph_wrapper));

  /* Call onnxInitGraph with unwrapped backend handle */
  const onnxStatus status = backend_wrapper->library->onnxInitGraph(
    backend_wrapper->backend,
    auxPropertiesList,
    onnxModelSize,
    onnxModel,
    weightsCount,
    weightDescriptors,
    &graph_wrapper->graph);
  switch (status) {
    case ONNXIFI_STATUS_SUCCESS:
    case ONNXIFI_STATUS_FALLBACK:
      /* Success, return wrapped graph */
      graph_wrapper->magic = ONNXIFI_GRAPH_MAGIC;
      graph_wrapper->library = backend_wrapper->library;
      *graph = (onnxGraph) graph_wrapper;
      return status;
    default:
      /* Failure, release allocated memory */
      free(graph_wrapper);
      return status;
  }
}

ONNXIFI_PUBLIC onnxStatus ONNXIFI_ABI onnxSetGraphIO(
    onnxGraph graph,
    uint32_t inputsCount,
    const onnxTensorDescriptorV1* inputDescriptors,
    uint32_t outputsCount,
    const onnxTensorDescriptorV1* outputDescriptors)
{
  if (graph == NULL) {
    return ONNXIFI_STATUS_INVALID_GRAPH;
  }

  struct onnxifi_graph_wrapper* graph_wrapper =
    (struct onnxifi_graph_wrapper*) graph;
  if (graph_wrapper->magic != ONNXIFI_GRAPH_MAGIC) {
    return ONNXIFI_STATUS_INVALID_GRAPH;
  }
  if (graph_wrapper->library == NULL) {
    return ONNXIFI_STATUS_INVALID_GRAPH;
  }

  /* Call onnxSetGraphIO with unwrapped graph handle */
  return graph_wrapper->library->onnxSetGraphIO(
    graph_wrapper->graph,
    inputsCount,
    inputDescriptors,
    outputsCount,
    outputDescriptors);
}

ONNXIFI_PUBLIC onnxStatus ONNXIFI_ABI
  onnxRunGraph(
    onnxGraph graph,
    const onnxMemoryFenceV1* inputFence,
    onnxMemoryFenceV1* outputFence)
{
  if (graph == NULL) {
    return ONNXIFI_STATUS_INVALID_GRAPH;
  }
  if (inputFence == NULL) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  if (outputFence == NULL) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  switch (inputFence->tag) {
    case ONNXIFI_TAG_MEMORY_FENCE_V1:
      break;
    default:
      return ONNXIFI_STATUS_UNSUPPORTED_TAG;
  }
  switch (outputFence->tag) {
    case ONNXIFI_TAG_MEMORY_FENCE_V1:
      break;
    default:
      return ONNXIFI_STATUS_UNSUPPORTED_TAG;
  }

  struct onnxifi_graph_wrapper* graph_wrapper =
    (struct onnxifi_graph_wrapper*) graph;
  if (graph_wrapper->magic != ONNXIFI_GRAPH_MAGIC) {
    return ONNXIFI_STATUS_INVALID_GRAPH;
  }
  if (graph_wrapper->library == NULL) {
    return ONNXIFI_STATUS_INVALID_GRAPH;
  }

  const onnxMemoryFenceV1* input_fence_ptr = inputFence;
  onnxMemoryFenceV1 input_fence_wrapper;
  switch (inputFence->type) {
    case ONNXIFI_SYNCHRONIZATION_EVENT:
    {
      if (inputFence->event == NULL) {
        return ONNXIFI_STATUS_INVALID_EVENT;
      }
      const struct onnxifi_event_wrapper* event_wrapper =
        (const struct onnxifi_event_wrapper*) inputFence->event;
      if (event_wrapper->magic != ONNXIFI_EVENT_MAGIC) {
        return ONNXIFI_STATUS_INVALID_EVENT;
      }
      if (event_wrapper->library != graph_wrapper->library) {
        return ONNXIFI_STATUS_INVALID_EVENT;
      }

      /* Initialize wrapper for input fence */
      input_fence_wrapper.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
      input_fence_wrapper.type = ONNXIFI_SYNCHRONIZATION_EVENT;
      input_fence_wrapper.event = event_wrapper->event;
      input_fence_ptr = &input_fence_wrapper;
      break;
    }
    default:
      /* Pass inputFence as is */
      break;
  }

  onnxMemoryFenceV1* output_fence_ptr = outputFence;
  onnxMemoryFenceV1 output_fence_wrapper;
  struct onnxifi_event_wrapper* output_event = NULL;
  switch (outputFence->type) {
    case ONNXIFI_SYNCHRONIZATION_EVENT:
    {
      /* Initialize wrapper for output fence */
      output_fence_wrapper.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
      output_fence_wrapper.type = ONNXIFI_SYNCHRONIZATION_EVENT;
      /* event will be populated by onnxRunGraph */
      output_fence_wrapper.event = NULL;
      output_fence_ptr = &output_fence_wrapper;

      /*
       * Pre-allocate memory for output event wrapper.
       * This must be done before onnxRunGraph, so in case allocation fails,
       * the function call has no side-effects.
       */
      output_event = malloc(sizeof(struct onnxifi_event_wrapper));
      if (output_event == NULL) {
        return ONNXIFI_STATUS_NO_SYSTEM_MEMORY;
      }
      memset(output_event, 0, sizeof(struct onnxifi_event_wrapper));
    }
  }

  const onnxStatus status = graph_wrapper->library->onnxRunGraph(
    graph_wrapper->graph,
    input_fence_ptr,
    output_fence_ptr);

  if (status == ONNXIFI_STATUS_SUCCESS) {
    /* Wrap output event, if needed */
    if (output_event != NULL) {
      output_event->magic = ONNXIFI_EVENT_MAGIC;
      output_event->event = output_fence_wrapper.event;
      output_event->library = graph_wrapper->library;
      outputFence->event = (onnxEvent) output_event;
    }
  } else {
    /* Deallocate output event wrapper, if needed */
    free(output_event);
  }

  return status;
}

ONNXIFI_PUBLIC onnxStatus ONNXIFI_ABI onnxReleaseGraph(
    onnxGraph graph)
{
  if (graph == NULL) {
    return ONNXIFI_STATUS_INVALID_GRAPH;
  }

  struct onnxifi_graph_wrapper* graph_wrapper =
    (struct onnxifi_graph_wrapper*) graph;
  if (graph_wrapper->magic != ONNXIFI_GRAPH_MAGIC) {
    return ONNXIFI_STATUS_INVALID_GRAPH;
  }

  const onnxStatus status =
    graph_wrapper->library->onnxReleaseGraph(graph_wrapper->graph);

  /*
   * Note: ReleaseGraph either succeeded, or failed with internal error.
   * Either way, it is not safe to use the graph handle again.
   */
  memset(graph_wrapper, 0, sizeof(struct onnxifi_graph_wrapper));
  free(graph_wrapper);

  return status;
}
