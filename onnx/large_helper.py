# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import enum
import os
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import numpy as np
import onnx
import onnx.external_data_helper as ext_data
import onnx.helper


def make_large_tensor_proto(
    location: str, tensor_name: str, tensor_type: int, shape: Tuple[int, ...]
) -> onnx.TensorProto:
    """
    Create an external tensor.

    :param location: unique identifier (not necessary a path)
    :param tensor_name: initializer tensor_name in the graph
    :param tensor_type: onnx type
    :param shape: shape the of the initializer
    :return: the created tensor
    """
    tensor_location = location
    tensor = onnx.TensorProto()
    tensor.name = tensor_name

    del tensor.external_data[:]
    tensor.data_location = onnx.TensorProto.EXTERNAL
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


class LargeModelFileFormat(enum.IntEnum):
    # One file for all the weights.
    SINGLE_TENSOR_FILE = 2

    # Multiple files, one file with extension `.onnx` for the
    # main graph and one file `.weight` for every large initializer.
    # It uses the same format as `write_external_data_tensors`.
    ONE_TENSOR_PER_FILE = 3


def _set_external_data(
    tensor: onnx.TensorProto,
    location: str,
    offset: Optional[int] = None,
    length: Optional[int] = None,
    checksum: Optional[str] = None,
    basepath: Optional[str] = None,
) -> None:
    del tensor.external_data[:]
    tensor.data_location = onnx.TensorProto.EXTERNAL
    for k, v in {
        "location": location,
        "offset": offset,
        "length": length,
        "checksum": checksum,
        "basepath": basepath,
    }.items():
        if v is not None:
            entry = tensor.external_data.add()
            entry.key = k
            entry.value = str(v)


class LargeModelContainer:
    """
    Implements an API to save large onnx models.
    Avoids copying large initializers when defining the model.
    """

    def __init__(self):
        self.model_proto_: Optional[onnx.ModelProto] = None
        self.large_initializers: Dict[str, np.ndarray] = {}

    def check_model(self):
        if self.model_proto is not None:
            onnx.checker.check_model(self.model_proto)

    @property
    def model_proto(self) -> onnx.ModelProto:
        if self.model_proto_ is None:
            raise RuntimeError("LargeModelContainer is empty.")
        return self.model_proto_

    @model_proto.setter
    def model_proto(self, model_proto: onnx.ModelProto):
        self.model_proto_ = model_proto
        self.graphs_ = list(self.enumerate_graph_protos())

    @staticmethod
    def _enumerate_subgraphs(graph):
        for node in graph.node:
            for att in node.attribute:
                if att.g:
                    yield att.g
                    yield from LargeModelContainer._enumerate_subgraphs(att.g)

    def enumerate_graph_protos(self) -> Iterable[onnx.GraphProto]:
        """
        Enumerates all GraphProtos in a model.
        """
        yield self.model_proto.graph
        yield from self._enumerate_subgraphs(self.model_proto.graph)

    def set_large_initializers(self, large_initializers: Dict[str, np.ndarray]):
        """
        Adds all large tensors (not stored in the model).
        """
        for k in large_initializers:
            if not k.startswith("#"):
                raise ValueError(
                    f"The location {k!r} must start with '#' to be ignored by check model."
                )
        self.large_initializers = large_initializers

    def check_large_initializers(self):
        for tensor in ext_data._get_all_tensors(self.model_proto):
            if not ext_data.uses_external_data(tensor):
                continue
            prop: Optional[onnx.StringStringEntryProto] = None
            for ext in tensor.external_data:  # type: ignore[assignment]
                if ext.key == "location":  # type: ignore[attr-defined]
                    prop = ext
            if prop is None:
                raise RuntimeError(
                    f"No location found for tensor name {tensor.name!r}."
                )
            if prop.value not in self.large_initializers:
                raise RuntimeError(
                    f"Unable to find large tensor named {tensor.name!r} "
                    f"with location {prop.value!r} in "
                    f"{sorted(self.large_initializers)}."
                )

    def _save_external(
        self, file_path: str, all_tensors_to_one_file: bool
    ) -> onnx.ModelProto:
        """
        Save the large model into a main onnx file and one file
        per tensor. Follows the same format as :func:`write_external_data_tensors
        <onnx.external_data_helper.write_external_data_tensors>`.
        The main model needs to be modified to update the file location,
        the function returns this modified copy.

        :param file_path: model file
        :param all_tensors_to_one_file: all tensors in one file
        :return: modified main model proto
        """
        _unique_names: Set[str] = set()

        def _clean_name(prefix: str, name: str) -> str:
            for c in ":/\\;,!":
                name = name.replace(c, "")
            base_name = name
            i = 0
            while name in _unique_names:
                i += 1
                name = f"{base_name}_{i}"
            _unique_names.add(name)
            return name

        ext = os.path.splitext(file_path)[-1]
        folder = os.path.dirname(file_path)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {folder!r} does not exist.")
        proto = self.model_proto.SerializeToString()
        copy = onnx.ModelProto()
        copy.ParseFromString(proto)
        prefix = os.path.splitext(os.path.split(file_path)[-1])[0]

        if all_tensors_to_one_file:
            file_weight = f"{os.path.split(file_path)[1]}.weight"
            full_file_weight = f"{file_path}.weight"
            offset = 0
            with open(full_file_weight, "wb") as f:
                pass

        for tensor in ext_data._get_all_tensors(copy):
            if not ext_data.uses_external_data(tensor):
                continue
            prop: Optional[onnx.StringStringEntryProto] = None
            for ext in tensor.external_data:  # type: ignore[assignment]
                if ext.key == "location":  # type: ignore[attr-defined]
                    prop = ext  # type: ignore[assignment]
            if prop is None:
                raise RuntimeError(
                    f"No location found for tensor name {tensor.name!r}."
                )
            if prop.value not in self.large_initializers:
                raise RuntimeError(
                    f"Unable to find large tensor named {tensor.name!r} "
                    f"with location {prop.value!r} in "
                    f"{sorted(self.large_initializers)}."
                )
            np_tensor = self.large_initializers[prop.value]

            if all_tensors_to_one_file:
                buffer = np_tensor.tobytes()
                _set_external_data(
                    tensor, location=file_weight, offset=offset, length=len(buffer)
                )
                offset += len(buffer)
                with open(full_file_weight, "ab") as f:
                    f.write(buffer)
            else:
                name = f"{_clean_name(prefix, prop.value)}.weight"
                _set_external_data(tensor, location=name)
                full_name = os.path.join(folder, name)
                prop.value = name
                with open(full_name, "wb") as f:
                    f.write(np_tensor.tobytes())

        with open(file_path, "wb") as f:
            f.write(copy.SerializeToString())
        return copy

    def save(
        self,
        file_path: str,
        file_format: LargeModelFileFormat = LargeModelFileFormat.ONE_TENSOR_PER_FILE,
    ) -> onnx.ModelProto:
        """
        Save the large model.
        The function returns a ModelProto,
        the current one if the model did not need any modification,
        a modified copy of it if it required changes such as giving file names
        to every external tensor.

        :param file_path: model file
        :param file_format: format to use
        :return: the saved ModelProto
        """
        if file_format in (
            LargeModelFileFormat.ONE_TENSOR_PER_FILE,
            LargeModelFileFormat.SINGLE_TENSOR_FILE,
        ):
            return self._save_external(
                file_path,
                all_tensors_to_one_file=file_format
                == LargeModelFileFormat.SINGLE_TENSOR_FILE,
            )
        raise ValueError(f"Unsupported file format {file_format}.")

    def load(self, file_path: str, load_large_initializers: bool = True):
        """
        Load the large model.

        :param file_path: model file
        :param load_large_initializers: loads the large initializers,
            if not done, the model is incomplete but it can be used to
            look into the model without executing it and method
            :meth:`_load_large_initializers` can be used to load them later
        """
        self.model_proto_ = onnx.load_model(file_path, load_external_data=False)
        if load_large_initializers:
            self._load_large_initializers(file_path)

    def _load_large_initializers(self, file_path):
        """
        Loads large initializers.

        :param file_path: model file, the weight are expected to be in the same folder as this file
        """
        if self.model_proto_ is None:
            raise RuntimeError("A model must be loaded before loading the weights.")
        self.large_initializers = {}
        base_dir = os.path.dirname(file_path)
        for i, tensor in enumerate(ext_data._get_all_tensors(self.model_proto_)):
            if not ext_data.uses_external_data(tensor):
                continue

            info = ext_data.ExternalDataInfo(tensor)
            file_location = ext_data._sanitize_path(info.location)
            external_data_file_path = os.path.join(base_dir, file_location)
            key = f"#t{i}"
            _set_external_data(tensor, location=key)

            with open(external_data_file_path, "rb") as data_file:
                if info.offset:
                    data_file.seek(info.offset)

                raw_data = (
                    data_file.read(info.length) if info.length else data_file.read()
                )

                dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
                shape = tuple(tensor.dims)
                self.large_initializers[key] = np.frombuffer(
                    raw_data, dtype=dtype
                ).reshape(shape)


def make_large_model(
    graph: onnx.GraphProto,
    large_initializers: Optional[Dict[str, np.ndarray]] = None,
    **kwargs: Any,
) -> LargeModelContainer:
    """Construct a LargeModelContainer

    C API and Python API of protobuf do not operate without serializing
    the protos. This function uses the Python API of LargeModelContainer.

    Arguments:
        graph: *make_graph* returns
        large_initializers: dictionary `(name, location): large tensor`,
            large tensor is any python object supporting the DLPack protocol,
            the ownership the tensor is transfered to the LargeModelContainer,
            the tensor must define method `tobytes` like numpy tensors
        **kwargs: any attribute to add to the returned instance
    Returns:
        LargeModelContainer
    """
    model = onnx.helper.make_model(graph, **kwargs)
    large_model = LargeModelContainer()
    large_model.model_proto = model
    if large_initializers:
        large_model.set_large_initializers(large_initializers)
        large_model.check_large_initializers()
    return large_model
