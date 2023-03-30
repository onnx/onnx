<!--
Copyright (c) ONNX Project Contributors
-->

<!--- SPDX-License-Identifier: Apache-2.0 -->

# ONNX Interface for Framework Integration: API Proposal

## Background

Leading hardware and systems vendors offer highly optimized software to run neural network graphs. These software can deliver order-of-magnitude speedups compared to generic implementations, but their integration with deep learning frameworks and applications is complicated by large variety in vendor-specific interfaces, and subtle incompatibilities with the software stack of high-level applications.

So far, ONNX format targets the problem of offline conversion of neural network models between different high-level frameworks and vendor-specific libraries through offline translation. In this proposal, we suggest that ONNX ecosystem could be enriched to enable runtime discovery and selection of high-performance graph execution backends, and online (in runtime) conversion of ONNX graph to internal representations of these implementations.

## Ultimate Goal

We should strive for consensus on a library API to interface with optimized backends and offload parts of ONNX graphs to these high-performance hardware and software implementation. The API should enable wide interoperability between high-level deep learning frameworks, software implementations of optimized graph runtimes, and existing and upcoming neural network acceleration hardware.

The standardized API should reduce friction in deploying neural network models for all involved parties:
- Applications would be able to ship only one version of a neural network model (either in ONNX format, or in the format of their deep learning framework, and convert it on the fly to ONNX).
- Deep learning frameworks would be able to integrate with many hardware vendors by using only a single interface.
- Hardware vendors would be able to implement only one interface and get integration with many deep learning frameworks.

## Design Choices

- Interface must use only highly portable aspects of C ABI.
- Neural network graphs are passed as serialized ONNX ModelProto messages. To avoid serialization overhead, weights can be passed as raw memory blobs.
- Input and output tensors are allocated by the caller and use NCHW layout.
- Intermediate tensors are allocated by the vendor implementation, and can use any layout.
- Backends (software implementations and hardware accelerators) are discovered, selected, and initialized on-demand in run-time. Multiple backends can be used in the same application simultaneously.
- There is no minimal set of ONNX operators to implement. The implementer and the user (a deep learning framework) of the API decide which operators can and will be offloaded in runtime.
- The proposal includes the minimal functionality to let deep learning frameworks and vendor libraries work together. Several extension mechanisms can be used for more efficient vendor- or platform-specific functionality.

## Proposed Interface

We propose a small C-based API, which includes the following functionality:

* Discover (`onnxGetNumBackends`) and query information (`onnxGetBackendInfo`) about high-performance backends
* Initialize (`onnxInitBackend`) and deinitialize (`onnxReleaseBackend`) high-performance backends
* Query if a backend supports an ONNX operator with particular parameters and input shapes (`onnxGetBackendCompatibility`)
* Convert an ONNX graph to opaque vendor-specific representation of a backend (`onnxInitGraph`)
* Specify memory locations and metadata about graph inputs and outputs (`onnxSetGraphIO`)
* Run an ONNX graph, converted to vendor-specific representation (`onnxRunGraph`)
* Release the vendor-specific representation of a graph and associated resources (`onnxReleaseGraph`)

## General Use Pattern for Deep Learning Frameworks

1. The user (deep learning framework) iterates operators in a model graph one-by-one, convert them to ONNX, and calls `onnxGetBackendCompatibility` to check which of the operators can be offloaded to the backend.
2. The user constructs connected subgraphs of operators that can be offloaded to the backend.
3. (Optional) For each subgraph, the user estimates if it is beneficial to offload it to the optimized backend:

    a. The user queries the backend about it high-level performance characteristics using `ONNX_BACKEND_MACS_*` and `ONNX_BACKEND_MEMORY_BANDWIDTH` information queries. These data let the user build a simple roofline model of backend performance.

    b. For every subgraph the user estimates time to do inference using the roofline model.

    c. The user additionally estimates time to transfer subgraph inputs to the backend using `ONNX_BACKEND_CPU_MEMORY_READ_BANDWIDTH` information query and to transfer subgraph outputs from the backend using `ONNX_BACKEND_CPU_MEMORY_WRITE_BANDWIDTH`.

    d. If predicted time to transfer inputs to the backend, do inference, and transfer outputs from the backend exceeds predicted time to do the inference on default engine (e.g. CPU), the user falls back to a different ONNX backend, or to the default engine.


4. The user initialized the backend, and offloads the subgraph execution to the ONNX backend by calling `onnxInitGraph`, `onnxSetGraphIO` and `onnxRunGraph`

## Implementation Notes

### Backend object

Backend is a combination of software library and hardware device. The same device (e.g. "NVIDIA Tesla P100 on CUDA index #0" accessed though different software libraries would be seen as different backends. A single software library can expose multiple backends, one per device  (e.g. each CUDA GPU in a system is exposed as a separate backend, or CPU, GPU, and DSP on a mobile chipset are exposed as three different backends).

We recommend that vendors make the backend object reference-counted, and use `uint32_t magic` as the first data field of the object:

```c
struct MyBackend {
  uint32_t magic;
  uint64_t referenceCount;
  ...
};

/* This line won't compile, but gives you an idea of relation between MyBackend structure and onnxBackend type. */
typedef MyBackend* onnxBackend;
```

Magic is an arbitrary 32-bit integer unique for a library implementing the API. It should be used to verify that the backend object passed to `onnxInitGraph` was created by `onnxInitBackend` in the same library.

### Graph object

Graph object is a vendor-specific representation of ONNX ModelProto message. Graph is logically related to the backend used to create it, and a typical implementation of a graph object would hold a reference to its backend object.

We recommend that vendors use `uint32_t magic` as the first data field of the graph object:

```c
struct MyGraph {
  uint32_t magic;
  struct MyBackend* backend;
  ...
};

/* This line won't compile, but gives you an idea of relation between MyGraph structure and onnxGraph type. */
typedef MyGraph* onnxGraph;
```

Magic is an arbitrary 32-bit integer unique for a library implementing the API. It should be used to verify that the backend object passed to `onnxInitGraph` was created by `onnxInitBackend` in the same library. Magic for a graph object should be different from magic of a backend object of the same library.

### Library initialization

During one-time library initialization, the implementation of the API would detect `n` supported devices and map them to backend indices in `0...(n-1)` range. The implementation of device discovery and checking required device characteristics is highly vendor- and platform-specific, e.g.:
- A CPU implementation may always expose 1 device.
- A CUDA-based implementation may call `cudaGetDeviceCount` to get the number of CUDA-enabled devices, then
 call `cudaGetDeviceProperties` for each device, and map CUDA devices which satisfy the minimum required functionality, such as compute capability, to backend indices.
- An OpenCL-based implementation for a mobile GPU would try to load OpenCL library, call `clGetPlatformIDs` and `clGetPlatformInfo` to find a supported platform, then call `clGetDeviceIDs` and `clGetDeviceInfo` to find a supported GPU device, and map it to the only exposed backend if such device exists, or expose 0 devices otherwise.
- An implementation for hardware neural network accelerators would call vendor-specific driver API to discover accelerator devices installed in the system and map them to backend indices.

We recommend that library initialization is triggered on the first call to `onnxGetNumBackends`, `onnxGetBackendInfo`, or `onnxInitBackend`. Using a global static C++ object for initialization may hurt portability if library initialization involves loading other shared libraries (DLLs): on Windows `LoadLibrary` function can't be used in initializers of global static objects.

### onnxGetNumBackends

Implementation would [initialize the library](#library-initialization), if it wasn't initialized already, and return the number `n` of available backends.

### onnxGetBackendInfo

Implementation would [initialize the library](#library-initialization), if it wasn't initialized already, and query information about the backend using vendor- or platform-specific API (e.g. `cudaGetDeviceProperties`, `clGetDeviceInfo`, CPUID instruction). Implementation can cache this information when it is first queried or during initialization, and return the cached value.
