# ONNX Interface for Framework Integration (ONNXIFI)

ONNXIFI is a cross-platform API for loading and executing ONNX graphs on optimized backends. High-level frameworks and applications can use this API to execute neural network and machine learning models. Hardware vendors can implement this API to expose specialized hardware accelerators and highly optimized software infrastructure to the users.

## Core Features

- Standardized interface for neural network inference on special-purpose accelerators (NPUs), CPUs, GPUs, DSPs, and FPGAs
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

0. (Optional) Use `onnxifi_load` to dynamically load the ONNX Interface for Framework Integration library.
1. Call `onnxGetBackendIDs` to get stable identifiers of available backends. Note: it is possible there are no backends installed in the system.
2. Call `onnxGetBackendInfo` to check additional information about any available backend.
3. Call `onnxGetBackendCompatibility` to check which operations within your model can run on the backend.
4. Call `onnxInitBackend` to initialize a backend, then call `onnxInitGraph` to offload one or more model graphs to the backend.
5. Call `onnxSetGraphIO` to set locations and shapes for inputs and outputs of a graph.
6. Initialize an `inputFence` structure of type `onnxMemoryFenceV1`: set `tag` to `ONNXIFI_TAG_MEMORY_FENCE_V1`, `type` to `ONNXIFI_SYNCHRONIZATION_EVENT`, and call `onnxInitEvent` to initiaze the `event` member.
7. Initialize an `outputFence` structure of type `onnxMemoryFenceV1`: set `tag` to `ONNXIFI_TAG_MEMORY_FENCE_V1`, `type` to `ONNXIFI_SYNCHRONIZATION_EVENT`, and `event` to null.
8. Call `onnxRunGraph` with the initialized `inputFence` and `outputFence` structures to enable execution of the graph. The call to `onnxRunGraph` will populate `event` member of the `outputFence` with a newly created event object, asynchronously execute the graph once `inputFence`'s `event` is signalled, and then signal the `outputFence`'s `event`.
9. Call `onnxSignalEvent` with `event` member of `inputFence` to signal to the backend that the inputs are ready to be consumed. 
10. Call `onnxWaitEvent` (alternatively, repeatedly call `onnxGetEventState` in a loop until the event state is `ONNXIFI_EVENT_STATE_SIGNALLED`) with `event` member of `outputFence` to wait until graph outputs are ready to be consumed. Release events for inputs and outputs using `onnxReleaseEvent`.
11. If your model works with fixed-size inputs and outputs, and shape and location of inputs and outputs does not change, one call to `onnxSetGraphIO` is sufficient for multiple `onnxRunGraph` calls. The previous call to `onnxRunGraph`, however, must have finished before a user calls `onnxRunGraph` again, because concurrent execution with the same input and output locations is not allowed. For models with variable-size inputs or outputs, you'd need to call `onnxSetGraphIO` before each `onnxRunGraph` call.
12. When done using the model, release the model graph(s) with `onnxReleaseGraph`, then release the backend with `onnxReleaseBackend` and backend ID with `onnxReleaseBackendID`.

## How to Implement ONNX Interface for Framework Integration

The minimum functionality an ONNXIFI implementation must provide is the following:

- Support ONNX 1.0 model format.
  - There is no minimum list of Operators a backed has to support.
- Support graph inputs / outputs in CPU memory.
- Support graph inputs / outputs with fixed shape, specified in GraphProto message.

### Discovery
Vendor-provided libraries should adhere to some rules to ensure discovery by ONNX-supported frameworks and applications:

1. The libraries must be installed in the following directories:
  - GNU/Linux: user-installed system library directory (typically /usr/lib)
  - macOS: /opt/onnx/lib
  - Windows: system directory (typically C:\Windows\System32)

2. Filenames of vendor-specific libraries must follow the rule below:
  - On Windows, library filename must match wildcard `onnxifi-*.dll`
  - On macOS, library filename must match wildcard `libonnxifi-*.dylib`
  - On Linux and other OSes, library filename must match wildcard `libonnxifi-*.so`

### Extensions

Hardware vendors are welcome to add their own extensions to ONNX backend interface. The backend interface offers several extension mechanisms:
- Experimental, exotic, or vendor-specific operators can be supported in a private domain using NodeProto.domain attribute.
- Vendor-provided ONNXIFI implementation can expose additional functions
