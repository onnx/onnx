# ONNX Interface for Framework Integration (ONNXIFI) Proposal

We propose a cross-platform API for loading and executing ONNX graphs on optimized backends. High-level frameworks and applications can use this API to execute neural network and machine learning models. Hardware vendors can implement this API to expose specialized hardware accelerators and highly optimized software infrastructure to the users.

## Core Features

- Standardized interface for neural network inference on special-purpose accelerators, CPUs, GPUs, DSPs, and FPGAs
- Based on widely supported technologies
  - C API for function calls
  - ONNX format for passing model graphs
  - NCHW tensor layout for passing inputs and outputs
- Dynamic discovery of available backends for model execution
  - Multiple backends from different vendors can co-exist on the same system
- Dynamic discovery of supported ONNX Operators on each backend

### Optional Features:

- Graphs with variable-shape inputs and/or outputs
- Graphs with data-dependendent output shapes

## How to Use ONNX Interface for Framework Integration

0. (Optional) Use `onnxifi_library_load` to dynamically load the ONNX Interface for Framework Integration library.
1. Call `onnxGetNumBackends` to get the number of available backends. Note that it can be 0.
2. Call `onnxGetBackendInfo` to check additional information about any available backend.
3. Call `onnxGetBackendCompatibility` to check if your model, or parts of it, can run on the backend.
4. Call `onnxInitBackend` to initialize a backend, then call `onnxInitGraph` to offload one or more model graphs to the backend.
5. Call `onnxSetGraphIO` to set inputs and output for the graph, then call `onnxRunGraph` to execute the graph(s). If your model works with fixed-size inputs, one call to `onnxSetGraphIO` is sufficient for multiple `onnxRunGraph` calls. For models with variable-size inputs, you'd need to call `onnxSetGraphIO` before each `onnxRunGraph` call.
6. When done using the model, release the model graph(s) with `onnxReleaseGraph`, then release the backend with `onnxReleaseBackend`

## How to Implement ONNX Interface for Framework Integration

The minimum functionality an ONNXIFI implementation must provide is the following:

- Support ONNX 1.0 model format.
  - There is no minimum list of Operators a backed has to support.
- Support graph inputs / outputs in CPU memory.
- Support graph inputs / outputs with fixed shape, specified in GraphProto message.

### Discovery
Vendor-provided libraries should adhere to some rules to ensure discovery by ONNX-supported frameworks and applications:
- Use `libonnxifi-<backend>.so` filename on Linux/Android
- Use `libonnxifi-<backend>.dylib` filename on macOS
- Use `onnxifi-<backend>.dll` filename on Windows
- Append ONNX function names with `<BACKEND>` (a vendor may define `ONNXIFI_LIBRARY_SUFFIX=<BACKEND>` and when using `onnxifi.h` header).
`<backend>` is the vendor-specific name in lowercase, and `<BACKEND>` is the same name in uppercase. E.g. a vendor **Gamma** would provide `libonnxifi-gamma.so` for Linux systems, and this library would implement functions `onnxGetNumBackendsGAMMA`, `onnxGetBackendInfoGAMMA`, etc.

### Extensions

Hardware vendors are welcome to add their own extensions to ONNX backend interface. The backend interface offers several extension mechanisms:
- Experimental, exotic, or vendor-specific operators can be supported in a private domain using NodeProto.domain attribute.
- Vendor-provided ONNXIFI implementation can expose additional functions
