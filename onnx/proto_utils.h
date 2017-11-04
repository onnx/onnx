#pragma once

#include <pybind11/pybind11.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>

namespace onnx {
  namespace py = pybind11;

  template<typename Proto>
    bool ParseProtoFromPyBytes(Proto* proto, const py::bytes& bytes) {

    // Get the buffer from Python bytes object
    char* buffer = nullptr;
    Py_ssize_t length;
    PyBytes_AsStringAndSize(bytes.ptr(), &buffer, &length);

    // Total bytes hard limit / warning limit are set to 1GB and 512MB
    // respectively.
    ::google::protobuf::io::CodedInputStream coded_stream(
      new google::protobuf::io::ArrayInputStream(buffer, length));
    coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
    return proto->ParseFromCodedStream(&coded_stream);
  }
}
