#if defined(__APPLE__)
#define ONNXIFI_DUMMY_LIBRARY "libonnxifi_dummy.dylib"
#elif defined(_WIN32)
#define ONNXIFI_DUMMY_LIBRARY L"onnxifi_dummy.dll"
#else
#define ONNXIFI_DUMMY_LIBRARY "libonnxifi_dummy.so"
#endif
