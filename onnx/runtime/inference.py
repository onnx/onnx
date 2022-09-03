# SPDX-License-Identifier: Apache-2.0
# pylint: disable=C0415,R0902,R0912,R0914,R0915
from io import BytesIO
from typing import Any, Dict, List, Tuple, Union

import numpy as np  # type: ignore

from onnx import FunctionProto, GraphProto, ModelProto, load, NodeProto, numpy_helper
from onnx.defs import onnx_opset_version


class Inference:
    """
    Executes an onnx model. The implementation relies on numpy
    for the most past and C++ through pybind11.

    :param proto: onnx model or file name
    :param verbose: display intermediate results
        on the standard output during the execution
    :param opsets: if *proto* is an instance of *GraphProto*,
        opsets must be defined by a dictionary of
    :param functions: known onnx functions

    The class maps every node to its associated implementation.
    When a subgraph of a function is met,
    it uses this class to execute the subgraph or the function.
    """

    def __init__(self, proto: Any, verbose: int = 0, opsets: Union[None, Dict[str, int]] = None, functions=None):  # type: ignore
        if isinstance(proto, str):
            with open(proto, "rb") as f:
                proto = load(f)
        elif isinstance(proto, bytes):
            proto = load(BytesIO(proto))
        self.proto_ = proto
        self.functions_: Dict[Tuple[str, str], Inference] = {}
        if isinstance(proto, ModelProto):
            self.onnx_graph_ = proto.graph
            self.opsets_ = {d.domain: d.version for d in proto.opset_import}
            if opsets is not None:
                raise ValueError("opsets must be None if proto is ModelProto.")
            if functions is not None:
                raise ValueError("functions must be None if proto is ModelProto.")
            functions = proto.functions
        elif isinstance(proto, GraphProto):
            self.onnx_graph_ = proto
            if not isinstance(opsets, dict):
                raise ValueError("opsets must be a dictionary if proto is GraphProto.")
            self.opsets_ = opsets
        elif isinstance(proto, FunctionProto):
            self.onnx_graph_ = None  # type: ignore
            self.opsets_ = {d.domain: d.version for d in proto.opset_import}
            if opsets is not None:
                raise ValueError("opsets must be None if proto is FunctionProto.")
        else:
            raise TypeError(f"Unexpected type {type(proto)} for proto.")
        if self.onnx_graph_:
            self.input_names_ = [i.name for i in self.onnx_graph_.input]
            self.output_names_ = [o.name for o in self.onnx_graph_.output]
            self.inits_ = list(self.onnx_graph_.initializer) + list(
                self.onnx_graph_.sparse_initializer  # type: ignore
            )
            self.nodes_ = self.onnx_graph_.node
        else:
            self.input_names_ = list(proto.input)
            self.output_names_ = list(proto.output)
            self.inits_ = []
            self.nodes_ = proto.node
        if "" not in self.opsets:
            self.opsets[""] = onnx_opset_version()
        if functions is not None:
            for f in functions:
                if isinstance(f, FunctionProto):
                    existing_functions = list(self.functions_.values())
                    self.functions_[f.domain, f.name] = Inference(
                        f, verbose=verbose, functions=existing_functions
                    )
                elif isinstance(f, Inference):
                    onx = f.proto_  # type: ignore
                    self.functions_[onx.domain, onx.name] = f
                else:
                    raise TypeError(f"Unexpected type {type(f)!r} for a function.")
        self.verbose = verbose
        self._init()

    def _log_arg(self, a: Any) -> Any:
        if isinstance(a, (str, int, float)):
            return a
        if isinstance(a, np.ndarray):
            if self.verbose < 4:
                return f"{a.dtype}:{a.shape} in [{a.min()}, {a.max()}]"
            elements = a.ravel().tolist()
            if len(elements) > 5:
                elements = elements[:5]
                return f"{a.dtype}:{a.shape}:{','.join(map(str, elements))}..."
            return f"{a.dtype}:{a.shape}:{elements}"
        if isinstance(a, list) or a.__class__.__name__ == "RepeatedScalarContainer":
            return ", ".join(map(self._log_arg, a))
        return a

    def _log(self, level: int, pattern: str, *args: List[Any]) -> None:
        if level < self.verbose:
            new_args = [self._log_arg(a) for a in args]
            print(pattern % tuple(new_args))

    @property
    def input_names(self):  # type: ignore
        "Returns the input names."
        return self.input_names_

    @property
    def output_names(self):  # type: ignore
        "Returns the output names."
        return self.output_names_

    @property
    def opsets(self):  # type: ignore
        "Returns the opsets."
        return self.opsets_

    def _init(self) -> None:
        """
        Loads the implementation for every node in the graph.
        """
        self.rt_inits_ = {}
        self.rt_nodes_ = []
        for init in self.inits_:
            self.rt_inits_[init.name] = numpy_helper.to_array(init)
        for node in self.nodes_:
            cl = self._load_impl(node)
            inst = cl(node, lambda pattern, *args: self._log(10, pattern, *args))
            self.rt_nodes_.append(inst)

    def _load_impl(self, node: NodeProto) -> type:
        """
        Loads the implementation for a specified runtime.
        """
        if node.domain not in self.opsets:
            raise RuntimeError(
                f"Domain {node.domain!r} (node type: {node.op_type!r}) "
                f"is not specified. Known opsets: {self.opsets!r}."
            )
        version = self.opsets[node.domain]
        if node.domain == "":
            from .aionnx import load_op

            return load_op(node.domain, node.op_type, version)
        if node.domain == "ai.onnx.ml":
            raise NotImplementedError(
                f"No implemented for domain {node.domain!r} is available yet."
            )
        # It has to be a function.
        key = node.domain, node.op_type
        if key in self.functions_:
            from .aionnx import load_op

            impl = self.functions_[key]
            return load_op(node.domain, node.op_type, version, custom=impl)
        raise RuntimeError(
            f"Node type {node.op_type!r} from domain {node.domain!r} "
            f"is unknown, known functions: {list(sorted(self.functions_))}."
        )

    def run(self, output_names, feed_inputs):  # type: ignore
        """
        Executes the onnx model.

        :param output_names: requested outputs by names,
            None for all
        :param feed_inputs: dictionary `{ input name: input value }`
        :return: list of requested outputs
        """
        if output_names is None:
            output_names = self.output_names

        # step 1: inputs and initalizers
        results = {"": None}  # optional input
        results.update(self.rt_inits_)
        results.update(feed_inputs)

        # step 2: execute nodes
        for node in self.rt_nodes_:
            self._log(1, "%s(%s) -> %s", node.op_type, node.input, node.output)
            inputs = [results[i] for i in node.input]
            outputs = node.run(*inputs)
            for name, value in zip(node.output, outputs):
                self._log(2, " + %s: %s", name, value)
                results[name] = value

        # return the results
        return [results[name] for name in output_names]
