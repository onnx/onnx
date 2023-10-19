# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import enum
import os
import struct
from typing import Any, Dict, Optional, Tuple

import numpy as np

from onnx import GraphProto, ModelProto, TensorProto, checker
from onnx.helper import make_model, tensor_dtype_to_np_dtype


def make_large_tensor_proto(
    location: str, name: str, tensor_type: int, shape: Tuple[int, ...]
) -> TensorProto:
    """
    Create an external tensor.

    :param location: unique identifier (not necessary a path)
    :param name: initializer name in the graph
    :param tensor_type: onnx type
    :param shape: shape the of the initializer
    :return: the created tensor
    """
    tensor_location = location
    tensor = TensorProto()
    tensor.name = name

    del tensor.external_data[:]
    tensor.data_location = TensorProto.EXTERNAL
    for k, v in {
        "location": tensor_location,
        "offset": None,
        "length": None,
        "checksum": None,
        "basepath": None,
    }.items():
        if v is not None:
            entry = tensor.external_data.add()
            entry.key = k
            entry.value = str(v)

    tensor.data_type = tensor_type
    tensor.dims.extend(shape)
    return tensor


class LargeOnnxFileFormat(enum.IntEnum):
    # One single file with extension `.lonnx`.
    # This format has no constraint on the initializer size.
    # However, the main graph still need to be under 2Gb as it
    # relies on protobuf.
    LARGE_ONNX = 1

    # Multiple files, one file with extension `.onnx` for the
    # main graph and one file `.weight` for every large initializer.
    ONNX_AND_WEIGHTS = 2


class LargeModelProto:
    """
    Implements an API to save large onnx models into a single file.
    Avoids copying large initializers when defining the model.
    """

    def __init__(self):
        self.model_proto_: Optional[ModelProto] = None
        self.large_initializers: Dict[Tuple[str, str], np.ndarray] = {}

    def check_model(self):
        checker.check_model(self.model_proto)

    @property
    def model_proto(self):
        return self.model_proto_

    @model_proto.setter
    def model_proto(self, model_proto: ModelProto):
        self.model_proto_ = model_proto
        self.graphs_ = list(self.enumerate_graph_protos())

    @staticmethod
    def _enumerate_subgraphs(graph):
        for node in graph.node:
            for att in node.attribute:
                if att.g:
                    yield att.g
                    for g in LargeModelProto._enumerate_subgraphs(att.g):
                        yield g

    def enumerate_graph_protos(self):
        """
        Enumerates all GraphProtos in a model.
        """
        yield self.model_proto.graph
        for g in self._enumerate_subgraphs(self.model_proto.graph):
            yield g

    @staticmethod
    def element_size(data_type: int) -> int:
        values = {
            TensorProto.FLOAT16: 4,
            TensorProto.FLOAT: 4,
            TensorProto.DOUBLE: 8,
            TensorProto.INT8: 1,
            TensorProto.INT16: 2,
            TensorProto.INT32: 4,
            TensorProto.INT64: 8,
            TensorProto.UINT8: 1,
            TensorProto.UINT16: 2,
            TensorProto.UINT32: 4,
            TensorProto.UINT64: 8,
        }
        if data_type not in values:
            raise RuntimeError(f"Element type {data_type} is not supported yet.")
        return values[data_type]

    @staticmethod
    def get_tensor_location(tensor) -> Optional[str]:
        if tensor.data_location != TensorProto.EXTERNAL:
            return None
        if len(tensor.external_data) == 0:
            return None
        for ext in tensor.external_data:
            if ext.key == "location":
                return ext.value
        raise RuntimeError(
            f"Unable to find a location for tensor name {tensor.name!r}."
        )

    def add_external_data(self, large_initializers: Dict[Tuple[str, str], np.ndarray]):
        """
        Adds all large tensors (not stored in the model).
        """
        for k in large_initializers.keys():
            if not k.startswith("#"):
                raise ValueError(
                    f"The location {k!r} must start with '#' to be ignored by check model."
                )
        self.large_initializers = large_initializers

    def _save_lonnx(self, file_path: str):
        """
        Save the large model into a single file.

        :param file_path: model file
        """
        ext = os.path.splitext(file_path)[-1]
        if ext != ".lonnx":
            raise ValueError(f"file_path {file_path} must have extension '.lonnx'.")
        with open(file_path, "wb") as f:
            proto = self.model_proto.SerializeToString()
            size = len(proto)
            f.write(struct.pack("Q", size))
            f.write(proto)
            f.write(struct.pack("I", len(self.large_initializers)))
            found_tensor = None
            graph_index = -1
            tensor_index = -1
            for location, np_tensor in self.large_initializers.items():
                for index, graph in enumerate(self.enumerate_graph_protos()):
                    for i, init in enumerate(graph.initializer):
                        loc = self.get_tensor_location(init)
                        if loc == location:
                            found_tensor = init
                            graph_index = index
                            tensor_index = i
                            break
                    if found_tensor is not None:
                        break
                if graph_index == -1:
                    raise RuntimeError(
                        f"Unable to find tensor location {location!r}. Did you change the structure?"
                    )
                init = found_tensor
                size = np.prod(init.dims) * self.element_size(init.data_type)
                buffer = np_tensor.tobytes()
                if len(buffer) != size:
                    raise RuntimeError(
                        f"Tensor of shape {tuple(init.dims)} of type {init.data_type} "
                        f"is expected to have size {size} but larger tensor has size {len(buffer)}."
                    )
                f.write(struct.pack("I", graph_index))
                f.write(struct.pack("I", tensor_index))
                f.write(buffer)

    def save(
        self,
        file_path: str,
        file_format: LargeOnnxFileFormat = LargeOnnxFileFormat.LARGE_ONNX,
    ):
        """
        Save the large model into a single file.

        :param file_path: model file
        :param file_format: format to use
        """
        if file_format == LargeOnnxFileFormat.LARGE_ONNX:
            self._save_lonnx(file_path)
            return
        raise ValueError(
            f"Unsupported format {file_format}. It is not implemented yet."
        )

    def _load_lonnx(
        self, file_path: str, load_large_initializers: bool = True
    ) -> "LargeModelProto":
        ext = os.path.splitext(file_path)[-1]
        if ext != ".lonnx":
            raise ValueError(f"file_path {file_path} must have extensions '.lonnx'.")
        with open(file_path, "rb") as f:
            size = struct.unpack("Q", f.read(8))[0]
            proto = f.read(size)
            self.model_proto = ModelProto()
            self.model_proto.ParseFromString(proto)
            self.graphs_ = list(self.enumerate_graph_protos())
            n_large_initializers = struct.unpack("I", f.read(4))[0]
            self.large_initializers = {}
            if load_large_initializers:
                for i in range(n_large_initializers):
                    graph_index = struct.unpack("I", f.read(4))[0]
                    init_index = struct.unpack("I", f.read(4))[0]
                    graph = self.graphs_[graph_index]
                    init = graph.initializer[init_index]
                    size = np.prod(init.dims) * self.element_size(init.data_type)
                    buffer = f.read(size)
                    np_tensor = np.frombuffer(
                        buffer,
                        dtype=tensor_dtype_to_np_dtype(init.data_type),
                    ).reshape(tuple(init.dims))
                    location = self.get_tensor_location(init)
                    self.large_initializers[location] = np_tensor

    def load(
        self, file_path: str, load_large_initializers: bool = True
    ) -> "LargeModelProto":
        """
        Load the large model from a single file.
        The format is guessed based on the file extension.
        `.lonnx` means the large onnx file format.

        :param file_path: model file
        :param load_large_initializers: loads the large initializers,
            if not done, the model is incomplete but it can be used to
            look into the model without executing it
        """
        ext = os.path.splitext(file_path)[-1]
        if ext == ".lonnx":
            self._load_lonnx(file_path, load_large_initializers=load_large_initializers)
            return
        raise ValueError(
            f"Unsupported format {file_format}. It is not implemented yet."
        )


def make_large_model(
    graph: GraphProto,
    large_initializers: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
    **kwargs: Any,
) -> LargeModelProto:
    """Construct a LargeModelProto

    C API and Python API of protobuf do not operate without serializing
    the protos. This function uses the Python API of LargeModelProto.

    Arguments:
        graph: *make_graph* returns
        large_initializers: dictionary `(name, location): large tensor`,
            large tensor is any python object supporting the DLPack protocol,
            the ownership the tensor is transfered to the LargeModelProto,
            the tensor must define method `tobytes` like numpy tensors
        **kwargs: any attribute to add to the returned instance
    Returns:
        LargeModelProto
    """
    model = make_model(graph, **kwargs)
    large_model = LargeModelProto()
    large_model.model_proto = model
    if large_initializers:
        large_model.add_external_data(large_initializers)
    return large_model
