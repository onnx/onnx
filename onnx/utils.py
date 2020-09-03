from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from typing import Text, Sequence

import onnx.checker
import onnx.helper
import onnx.optimizer
import onnx.shape_inference

from onnx import ModelProto


def polish_model(model):  # type: (ModelProto) -> ModelProto
    '''
        This function combines several useful utility functions together.
    '''
    onnx.checker.check_model(model)
    onnx.helper.strip_doc_string(model)
    model = onnx.shape_inference.infer_shapes(model)
    model = onnx.optimizer.optimize(model)
    onnx.checker.check_model(model)
    return model


class Extractor:
    def __init__(self, model):  # type: ignore
        self.model = onnx.shape_inference.infer_shapes(model)
        self.graph = self.model.graph
        self.wmap = self._buildNameDict(self.graph.initializer)
        self.vimap = self._buildNameDict(self.graph.value_info)

    @staticmethod
    def _buildNameDict(objs):  # type: ignore
        return {obj.name: obj for obj in objs}

    def _filterTensors(self, originals, new_names):  # type: ignore
        tmap = self._buildNameDict(originals)
        original_names = set(tmap.keys())
        s_new_names = set(new_names)
        names_keep = s_new_names & original_names
        names_add = s_new_names - original_names

        tensors = []
        for name in names_keep:
            tensors.append(tmap[name])
        for name in names_add:
            # activation become input or output
            tensors.append(self.vimap[name])

        # sort the tensors
        new_tmap = self._buildNameDict(tensors)
        sorted_tensors = [new_tmap[name] for name in new_names]
        return sorted_tensors

    def _filterInputs(self, names):  # type: ignore
        return self._filterTensors(self.graph.input, names)

    def _filterOutputs(self, names):  # type: ignore
        return self._filterTensors(self.graph.output, names)

    def _dfsSearchReachableNodes(self, node_output_name, graph_input_names, reachable_nodes):  # type: ignore
        if node_output_name in graph_input_names:
            return
        for node in self.graph.node:
            if node in reachable_nodes:
                continue
            if node_output_name not in node.output:
                continue
            reachable_nodes.append(node)
            for name in node.input:
                self._dfsSearchReachableNodes(name, graph_input_names, reachable_nodes)

    def _filterNodes(self, input_names, output_names):  # type: ignore
        reachable_nodes = list()  # type: ignore
        for name in output_names:
            self._dfsSearchReachableNodes(name, input_names, reachable_nodes)
        # needs to be topology sorted.
        nodes = [n for n in self.graph.node if n in reachable_nodes]
        return nodes

    def _searchReachableTensors(self, nodes):  # type: ignore
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

    def _make_model(self, nodes, inputs, outputs, initializer, value_info):  # type: ignore
        name = 'Extracted from {' + self.graph.name + '}'
        graph = onnx.helper.make_graph(nodes, name, inputs, outputs, initializer=initializer,
                                      value_info=value_info)

        meta = {
            'ir_version': self.model.ir_version,
            'opset_imports': self.model.opset_import,
            'producer_name': 'onnx.utils.extract',
        }
        return onnx.helper.make_model(graph, **meta)

    def extract(self, input_names=None, output_names=None):  # type: ignore
        graph = self.model.graph
        if input_names is None:
            input_names = [t.name for t in graph.input]
        if output_names is None:
            output_names = [t.name for t in graph.output]

        inputs = self._filterInputs(input_names)
        outputs = self._filterOutputs(output_names)
        nodes = self._filterNodes(input_names, output_names)
        initializer, value_info = self._searchReachableTensors(nodes)
        model = self._make_model(nodes, inputs, outputs, initializer, value_info)

        return model


def extract(
        input_path,  # type: Text
        output_path,  # type: Text
        input_names,  # type: Sequence[Text]
        output_names,  # type: Sequence[Text]
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
        raise ValueError("Wrong input model path: %s" % input_path)
    if not output_path:
        raise ValueError("Output model path shall not be empty!")
    if not input_names or not output_names:
        raise ValueError("Input/output tensor names shall not be empty!")

    onnx.checker.check_model(input_path)
    model = onnx.load(input_path)

    e = Extractor(model)
    extracted = e.extract(input_names, output_names)

    onnx.save(extracted, output_path)
    onnx.checker.check_model(output_path)
