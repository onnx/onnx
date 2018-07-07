#pragma once

#include <pybind11/pybind11.h>
#include "onnx/proto_utils.h"

namespace ONNX_NAMESPACE {
namespace py = pybind11;

template <typename Proto>
bool ParseProtoFromPyBytes(Proto* proto, const py::bytes& bytes) {
  // Get the buffer from Python bytes object
  char* buffer = nullptr;
  Py_ssize_t length;
  PyBytes_AsStringAndSize(bytes.ptr(), &buffer, &length);

  return ParseProtoFromBytes(proto, buffer, length);
}

bool ParseOpSetIDFromPyBytes(OpSetID* opsetid, const py::bytes& bytes) {
  // Get the buffer from Python bytes object
  char* buffer = nullptr;
  Py_ssize_t length;
  PyBytes_AsStringAndSize(bytes.ptr(), &buffer, &length);
  // Split char* on "$"
  char * pch;
  pch = strtok (buffer,"$");
  std::string domain(pch);
  opsetid->domain = domain;
  pch = strtok(NULL, "$");
  int version;
  sscanf(pch, "%d", &version);
  opsetid->version = version;
  // TODO: Are there any cases where this would fail?
  return true;
}
} // namespace ONNX_NAMESPACE
