# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import tarfile
from collections import deque
from typing import TYPE_CHECKING

import onnx.checker
import onnx.helper
import onnx.shape_inference
from onnx import FunctionProto, ModelProto, NodeProto, TensorProto, ValueInfoProto

if TYPE_CHECKING:
    from google.protobuf.internal.containers import RepeatedCompositeFieldContainer


class Extractor:
    def __init__(self, model: ModelProto) -> None:
        self.model = model
        self.graph = self.model.graph
        self.wmap = self._build_name2obj_dict(self.graph.initializer)
        self.vimap = self._build_name2obj_dict(self.graph.value_info)
        self.outmap = self._build_output_dict(self.graph)

    @staticmethod
    def _build_name2obj_dict(objs) -> dict:
        return {obj.name: obj for obj in objs}

    @staticmethod
    def _build_output_dict(graph) -> dict[str, int]:
        output_to_index: dict[str, int] = {}
        for index, node in enumerate(graph.node):
            for output_name in node.output:
                if output_name == "":
                    continue
                assert output_name not in output_to_index  # output_name is unique
                output_to_index[output_name] = index
        return output_to_index

    def _collect_new_io_core(
        self,
        original_io: RepeatedCompositeFieldContainer[ValueInfoProto],
        io_names_to_extract: list[str],
    ) -> list[ValueInfoProto]:
        original_io_map = self._build_name2obj_dict(original_io)
        new_io_tensors = []
        for io_name_to_extract in io_names_to_extract:
            if io_name_to_extract in original_io_map:
                new_io_tensors.append(original_io_map[io_name_to_extract])
            else:
                new_io_tensors.append(self.vimap[io_name_to_extract])
        return new_io_tensors  # same order as io_names_to_extract

    def _collect_new_inputs(self, names: list[str]) -> list[ValueInfoProto]:
        return self._collect_new_io_core(self.graph.input, names)

    def _collect_new_outputs(self, names: list[str]) -> list[ValueInfoProto]:
        return self._collect_new_io_core(self.graph.output, names)

    def _dfs_search_reachable_nodes(
        self,
        node_output_name: str,
        graph_input_names: set[str],
        reachable: set[int],
    ) -> None:
        """Helper function to find nodes which are connected to an output

        Arguments:
            node_output_name (str): The name of the output
            graph_input_names (set of string): The names of all inputs of the graph
            reachable (set of int): The set of indexes to reachable nodes in `nodes`
        """
        stack = [node_output_name]
        while stack:
            current_output_name = stack.pop()
            # finish search at inputs
            if current_output_name in graph_input_names:
                continue
            # find nodes connected to this output
            if current_output_name in self.outmap:
                index = self.outmap[current_output_name]
                if index not in reachable:
                    # add nodes connected to this output to sets
                    reachable.add(index)
                    stack += [
                        input_name
                        for input_name in self.graph.node[index].input
                        if input_name != ""
                    ]

    def _collect_reachable_nodes(
        self,
        input_names: list[str],
        output_names: list[str],
    ) -> list[NodeProto]:
        _input_names = set(input_names)
        reachable: set[int] = set()
        for name in output_names:
            self._dfs_search_reachable_nodes(name, _input_names, reachable)
        # needs to be topologically sorted
        return [self.graph.node[index] for index in sorted(reachable)]

    def _collect_referred_local_functions(
        self,
        nodes: list[NodeProto],
    ) -> list[FunctionProto]:
        # a node in a model graph may refer a function.
        # a function contains nodes, some of which may in turn refer a function.
        # we need to find functions referred by graph nodes and
        # by nodes used to define functions.
        function_map: dict[tuple[str, str], FunctionProto] = {}
        for function in self.model.functions:
            function_map[(function.name, function.domain)] = function
        referred_local_functions: list[FunctionProto] = []
        queue = deque(nodes)
        while queue:
            node = queue.popleft()
            # check if the node is a function op
            if (node.op_type, node.domain) in function_map:
                function = function_map.pop((node.op_type, node.domain))
                referred_local_functions.append(function)
                queue.extend(function.node)
        # needs to be topologically sorted
        return referred_local_functions

    def _collect_reachable_tensors(
        self,
        nodes: list[NodeProto],
    ) -> tuple[list[TensorProto], list[ValueInfoProto]]:
        all_tensors_names: set[str] = set()
        for node in nodes:
            all_tensors_names.update(node.input)
            all_tensors_names.update(node.output)
        initializer = [self.wmap[t] for t in self.wmap if t in all_tensors_names]
        value_info = [self.vimap[t] for t in self.vimap if t in all_tensors_names]
        len_sparse_initializer = len(self.graph.sparse_initializer)
        if len_sparse_initializer != 0:
            raise ValueError(
                f"len_sparse_initializer is {len_sparse_initializer}, it must be 0."
            )
        len_quantization_annotation = len(self.graph.quantization_annotation)
        if len_quantization_annotation != 0:
            raise ValueError(
                f"len_quantization_annotation is {len_quantization_annotation}, it must be 0."
            )
        return initializer, value_info

    def _make_model(
        self,
        nodes: list[NodeProto],
        inputs: list[ValueInfoProto],
        outputs: list[ValueInfoProto],
        initializer: list[TensorProto],
        value_info: list[ValueInfoProto],
        local_functions: list[FunctionProto],
    ) -> ModelProto:
        name = "Extracted from {" + self.graph.name + "}"
        graph = onnx.helper.make_graph(
            nodes, name, inputs, outputs, initializer=initializer, value_info=value_info
        )
        meta = {
            "ir_version": self.model.ir_version,
            "opset_imports": self.model.opset_import,
            "producer_name": "onnx.utils.extract_model",
            "functions": local_functions,
        }
        return onnx.helper.make_model(graph, **meta)

    def extract_model(
        self,
        input_names: list[str],
        output_names: list[str],
    ) -> ModelProto:
        inputs = self._collect_new_inputs(input_names)
        outputs = self._collect_new_outputs(output_names)
        nodes = self._collect_reachable_nodes(input_names, output_names)
        initializer, value_info = self._collect_reachable_tensors(nodes)
        local_functions = self._collect_referred_local_functions(nodes)
        model = self._make_model(
            nodes, inputs, outputs, initializer, value_info, local_functions
        )
        return model


def extract_model(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    input_names: list[str],
    output_names: list[str],
    check_model: bool = True,
    infer_shapes: bool = True,
) -> None:
    """Extracts sub-model from an ONNX model.

    The sub-model is defined by the names of the input and output tensors *exactly*.

    Note: For control-flow operators, e.g. If and Loop, the _boundary of sub-model_,
    which is defined by the input and output tensors, should not _cut through_ the
    subgraph that is connected to the _main graph_ as attributes of these operators.

    Note: When the extracted model size is larger than 2GB, the extra data will be saved in "output_path.data".

    Arguments:
        input_path (str | os.PathLike): The path to original ONNX model.
        output_path (str | os.PathLike): The path to save the extracted ONNX model.
        input_names (list of string): The names of the input tensors that to be extracted.
        output_names (list of string): The names of the output tensors that to be extracted.
        check_model (bool): Whether to run model checker on the original model and the extracted model.
        infer_shapes (bool): Whether to infer the shapes of the original model.
    """
    if not os.path.exists(input_path):
        raise ValueError(f"Invalid input model path: {input_path}")
    if not output_path:
        raise ValueError("Output model path shall not be empty!")
    if not input_names:
        raise ValueError("Input tensor names shall not be empty!")
    if not output_names:
        raise ValueError("Output tensor names shall not be empty!")

    if len(input_names) != len(set(input_names)):
        raise ValueError("Duplicate names found in the input tensor names.")
    if len(output_names) != len(set(output_names)):
        raise ValueError("Duplicate names found in the output tensor names.")

    if check_model:
        onnx.checker.check_model(input_path)

    if infer_shapes and os.path.getsize(input_path) > onnx.checker.MAXIMUM_PROTOBUF:
        onnx.shape_inference.infer_shapes_path(input_path, output_path)
        model = onnx.load(output_path)
    elif infer_shapes:
        model = onnx.load(input_path, load_external_data=False)
        model = onnx.shape_inference.infer_shapes(model)
        base_dir = os.path.dirname(input_path)
        onnx.load_external_data_for_model(model, base_dir)
    else:
        model = onnx.load(input_path)

    e = Extractor(model)
    extracted = e.extract_model(input_names, output_names)

    if extracted.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF:
        location = os.path.basename(output_path) + ".data"
        onnx.save(extracted, output_path, save_as_external_data=True, location=location)
    else:
        onnx.save(extracted, output_path)

    if check_model:
        onnx.checker.check_model(output_path)


def _tar_members_filter(
    tar: tarfile.TarFile, base: str | os.PathLike
) -> list[tarfile.TarInfo]:
    """Check that the content of ``tar`` will be extracted safely

    Args:
        tar: The tarball file
        base: The directory where the tarball will be extracted

    Returns:
        list of tarball members
    """
    result = []
    for member in tar:
        member_path = os.path.join(base, member.name)
        abs_base = os.path.abspath(base)
        abs_member = os.path.abspath(member_path)
        if not abs_member.startswith(abs_base):
            raise RuntimeError(
                f"The tarball member {member_path} in downloading model contains "
                f"directory traversal sequence which may contain harmful payload."
            )
        if member.issym() or member.islnk():
            raise RuntimeError(
                f"The tarball member {member_path} in downloading model contains "
                f"symbolic links which may contain harmful payload."
            )
        result.append(member)
    return result


def _extract_model_safe(
    model_tar_path: str | os.PathLike, local_model_with_data_dir_path: str | os.PathLike
) -> None:
    """Safely extracts a tar file to a specified directory.

    This function ensures that the extraction process mitigates against
    directory traversal vulnerabilities by validating or sanitizing paths
    within the tar file. It also provides compatibility for different versions
    of the tarfile module by checking for the availability of certain attributes
    or methods before invoking them.

    Args:
        model_tar_path: The path to the tar file to be extracted.
        local_model_with_data_dir_path: The directory path where the tar file
      contents will be extracted to.
    """
    with tarfile.open(model_tar_path) as model_with_data_zipped:
        # Mitigate tarball directory traversal risks
        if hasattr(tarfile, "data_filter"):
            model_with_data_zipped.extractall(
                path=local_model_with_data_dir_path, filter="data"
            )
        else:
            model_with_data_zipped.extractall(
                path=local_model_with_data_dir_path,
                members=_tar_members_filter(
                    model_with_data_zipped, local_model_with_data_dir_path
                ),
            )
