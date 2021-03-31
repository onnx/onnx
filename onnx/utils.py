# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from typing import List, Tuple, Text

import onnx.checker
import onnx.helper
import onnx.shape_inference

from onnx import ModelProto, NodeProto, TensorProto, ValueInfoProto


class Extractor:
    def __init__(self, model):  # type: (ModelProto) -> None
        self.model = onnx.shape_inference.infer_shapes(model)
        self.graph = self.model.graph
        self.wmap = self._build_name2obj_dict(self.graph.initializer)
        self.vimap = self._build_name2obj_dict(self.graph.value_info)

    @staticmethod
    def _build_name2obj_dict(objs):  # type: ignore
        return {obj.name: obj for obj in objs}

    def _collect_new_io_core(self, original_io, io_names_to_extract):  # type: ignore
        original_io_map = self._build_name2obj_dict(original_io)
        original_io_names = set(original_io_map.keys())
        s_io_names_to_extract = set(io_names_to_extract)
        io_names_to_keep = s_io_names_to_extract & original_io_names
        new_io_names_to_add = s_io_names_to_extract - original_io_names

        new_io_tensors = []
        for name in io_names_to_keep:
            new_io_tensors.append(original_io_map[name])
        for name in new_io_names_to_add:
            # activation become input or output
            new_io_tensors.append(self.vimap[name])

        # adjust sequence
        new_io_tensors_map = self._build_name2obj_dict(new_io_tensors)
        return [new_io_tensors_map[name] for name in io_names_to_extract]

    def _collect_new_inputs(self, names):  # type: (List[Text]) -> List[ValueInfoProto]
        return self._collect_new_io_core(self.graph.input, names)  # type: ignore

    def _collect_new_outputs(self, names):  # type: (List[Text]) -> List[ValueInfoProto]
        return self._collect_new_io_core(self.graph.output, names)  # type: ignore

    def _dfs_search_reachable_nodes(
            self,
            node_output_name,  # type: Text
            graph_input_names,  # type: List[Text]
            reachable_nodes,  # type: List[NodeProto]
    ):  # type: (...) -> None
        if node_output_name in graph_input_names:
            return
        for node in self.graph.node:
            if node in reachable_nodes:
                continue
            if node_output_name not in node.output:
                continue
            reachable_nodes.append(node)
            for name in node.input:
                self._dfs_search_reachable_nodes(name, graph_input_names, reachable_nodes)

    def _collect_reachable_nodes(
            self,
            input_names,  # type: List[Text]
            output_names,  # type: List[Text]
    ):  # type: (...) -> List[NodeProto]
        reachable_nodes = list()  # type: ignore
        for name in output_names:
            self._dfs_search_reachable_nodes(name, input_names, reachable_nodes)
        # needs to be topology sorted.
        nodes = [n for n in self.graph.node if n in reachable_nodes]
        return nodes

    def _collect_reachable_tensors(
            self,
            nodes,  # type: List[NodeProto]
    ):  # type: (...) -> Tuple[List[TensorProto], List[ValueInfoProto]]
        all_tensors_name = set()
        for node in nodes:
            for name in node.input:
                all_tensors_name.add(name)
            for name in node.output:
                all_tensors_name.add(name)

        initializer = [self.wmap[t] for t in self.wmap.keys() if t in all_tensors_name]
        value_info = [self.vimap[t] for t in self.vimap.keys() if t in all_tensors_name]
        assert(len(self.graph.sparse_initializer) == 0)
        assert(len(self.graph.quantization_annotation) == 0)
        return (initializer, value_info)

    def _make_model(
            self,
            nodes,  # type: List[NodeProto]
            inputs,  # type: List[ValueInfoProto]
            outputs,  # type: List[ValueInfoProto]
            initializer,  # type: List[TensorProto]
            value_info  # type: List[ValueInfoProto]
    ):  # type: (...) -> ModelProto
        name = 'Extracted from {' + self.graph.name + '}'
        graph = onnx.helper.make_graph(nodes, name, inputs, outputs, initializer=initializer,
                                      value_info=value_info)

        meta = {
            'ir_version': self.model.ir_version,
            'opset_imports': self.model.opset_import,
            'producer_name': 'onnx.utils.extract_model',
        }
        return onnx.helper.make_model(graph, **meta)

    def extract_model(
            self,
            input_names,  # type: List[Text]
            output_names,  # type: List[Text]
    ):  # type: (...) -> ModelProto
        inputs = self._collect_new_inputs(input_names)
        outputs = self._collect_new_outputs(output_names)
        nodes = self._collect_reachable_nodes(input_names, output_names)
        initializer, value_info = self._collect_reachable_tensors(nodes)
        model = self._make_model(nodes, inputs, outputs, initializer, value_info)

        return model


def extract_model(
        input_path,  # type: Text
        output_path,  # type: Text
        input_names,  # type: List[Text]
        output_names  # type: List[Text]
):  # type: (...) -> None
    """Extracts sub-model from an ONNX model.

    The sub-model is defined by the names of the input and output tensors *exactly*.

    Note: For control-flow operators, e.g. If and Loop, the _boundary of sub-model_,
    which is defined by the input and output tensors, should not _cut through_ the
    subgraph that is connected to the _main graph_ as attributes of these operators.

    Arguments:
        input_path (string): The path to original ONNX model.
        output_path (string): The path to save the extracted ONNX model.
        input_names (list of string): The names of the input tensors that to be extracted.
        output_names (list of string): The names of the output tensors that to be extracted.
    """
    if not os.path.exists(input_path):
        raise ValueError("Invalid input model path: %s" % input_path)
    if not output_path:
        raise ValueError("Output model path shall not be empty!")
    if not output_names:
        raise ValueError("Output tensor names shall not be empty!")

    onnx.checker.check_model(input_path)
    model = onnx.load(input_path)

    e = Extractor(model)
    extracted = e.extract_model(input_names, output_names)

    onnx.save(extracted, output_path)
    onnx.checker.check_model(output_path)
