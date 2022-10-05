# SPDX-License-Identifier: Apache-2.0
# pylint: disable=C0415,R0902,R0912,R0913,R0914,R0915
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np  # type: ignore

from .. import load, numpy_helper
from ..defs import onnx_opset_version
from ..onnx_pb import FunctionProto, GraphProto, ModelProto, NodeProto
from .op_run import OpRun


class ProtoRun:
    """
    Executes an onnx model. The implementation relies on numpy
    for the most past and C++ through pybind11.

    :param proto: ModelProto, GraphProto, FunctionProto, filename or bytes
    :param verbose: display intermediate results
        on the standard output during the execution
    :param opsets: if *proto* is an instance of *GraphProto*,
        opsets must be defined by a dictionary of
    :param functions: known onnx functions
    :param new_ops: this runtime can be used to test the implementations
        of new operators, *new_ops* is a list of classes
        derived from :class:`OpRun <onnx.funconnx.op_run.OpRun>`,
        every class must define the static attribute `domain`

    The class maps every node to its associated implementation.
    When a subgraph of a function is met,
    it uses this class to execute the subgraph or the function.
    Next example shows how to run `ProtoRun` with an onnx model
    stored in file `model.onnx`.

    ::

        import numpy as np
        from onnx.funconnx import ProtoRun

        X = np.array(...)
        sess = ProtoRun("model.onnx")
        results = sess.run(None, {"X": X})
        print(results[0])  # display the first result

    The class can be use any implementation available in folder
    `aionnx <https://github.com/onnx/onnx/tree/main/onnx/runtime/aionnx>`_.
    Adding an implementation requires two changes. The first is
    the implementation itself. Any existing node can be used as a template.
    The second is one line in file `_op_list.py
    <https://github.com/onnx/onnx/tree/main/onnx/runtime/aionnx/_op_file.py>`_
    to import the file and let the runtime know it exists.

    This class can also be used to test an implementation of
    a custom operator. Let's assume this new operator
    is `InvAlpha` from domain `custom`. The implementation
    must take place in a class inheriting from
    :class:`OpRun <onnx.funconnx.op_run.OpRun>`.
    It must also define attribute `op_domain`.
    Here is an example which computes :math:`\\frac{1}{X + \\alpha}`.

    .. exec_code::

        from onnx.funconnx.op_run import OpRun

        class InvAlpha(OpRun):

            op_domain = "custom"

            def _run(self, x, alpha=None):  # type: ignore
                # None must be the default value, it is automatically
                # replaced by class OpRun with either the default value
                # defined by ONNX or the value specified in the NodeProto.
                return (1 / (x + alpha),)

    `alpha` is an attribute. It can be defined by the onnx node or
    be defined by the function used this node. It is possible
    Line `alpha = alpha or self.alpha` selects first the value defined
    the onnx function if that's the case of falls back to the default
    value defined by the onnx node.
    to link a function attribute to a node attribute. In that case,
    Class `ProtoRun` must know about this new implementation
    and this can be done by specified argument *new_ops*.

    ::

        sess = ProtoRun(onnx_model, new_ops=[InvAlpha])
        got = sess.run(None, {"X": x})[0]

    A specific node can be simply evaluated.

    .. exec_code::

        import numpy as np
        from onnx.funconnx.aionnx._op_list import Celu

        x = np.array([[0, 1], [-1, 2]], dtype=np.float32)
        y = Celu.eval(x, alpha=0.5)
        print(y)

    This can also be expressed as:

    .. exec_code::

        import numpy as np
        from onnx.funconnx.aionnx import load_op

        Celu = load_op("", "Celu")  # domain is ""
        x = np.array([[0, 1], [-1, 2]], dtype=np.float32)
        y = Celu.eval(x, alpha=0.5)
        print(y)

    Method :meth:`eval <onnx.ProtoRun.op_run.eval>` creates an onnx node
    returned by method :meth:`make_node <onnx.funconnx.op_run.make_node>`.

    .. exec_code::

        import numpy as np
        from onnx.funconnx.aionnx._op_list import Celu

        onnx_node = Celu.make_node(alpha=0.5)
        print(onnx_node)
    """

    def __init__(  # type: ignore
        self,
        proto: Any,
        opsets: Union[None, Dict[str, int]] = None,
        functions=None,
        verbose: int = 0,
        new_ops: Optional[List[OpRun]] = None,
    ):
        self.output_types_ = None
        self.input_types_ = None
        if isinstance(proto, str):
            with open(proto, "rb") as f:
                proto = load(f)
        elif isinstance(proto, bytes):
            proto = load(BytesIO(proto))
        self.proto_ = proto
        self.functions_: Dict[Tuple[str, str], ProtoRun] = {}
        self.attributes_: List[str] = []
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
            self.attributes_ = list(proto.attribute)
        else:
            raise TypeError(f"Unexpected type {type(proto)} for proto.")
        if self.onnx_graph_:
            self.input_names_ = [i.name for i in self.onnx_graph_.input]
            self.input_types_ = [i.type for i in self.onnx_graph_.input]
            self.output_names_ = [o.name for o in self.onnx_graph_.output]
            self.output_types_ = [i.type for i in self.onnx_graph_.output]
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
                    self.functions_[f.domain, f.name] = ProtoRun(
                        f, verbose=verbose, functions=existing_functions
                    )
                elif isinstance(f, ProtoRun):
                    onx = f.proto_  # type: ignore
                    self.functions_[onx.domain, onx.name] = f
                else:
                    raise TypeError(f"Unexpected type {type(f)!r} for a function.")
        self.verbose = verbose
        self.new_ops_: Dict[Tuple[str, str], OpRun] = {}
        if new_ops is not None:
            for cl in new_ops:
                if not issubclass(cl, OpRun):  # type: ignore
                    raise TypeError(
                        f"Class {type(cl)} must inherit from OpRun (in new_ops)."
                    )
                if not hasattr(cl, "op_domain"):
                    raise AttributeError(
                        f"Class {type(cl)} must define attribute 'op_domain'."
                    )
                key = cl.op_domain, cl.__name__  # type: ignore
                if key in self.new_ops_:
                    raise ValueError(
                        f"Operator {cl.__name__!r} from domain {cl.op_domain!r} already exsits."  # type: ignore
                    )
                self.new_ops_[key] = cl
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
        if hasattr(a, "append"):
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
    def input_types(self):  # type: ignore
        "Returns the input types if any specified."
        return self.input_types_

    @property
    def output_names(self):  # type: ignore
        "Returns the output names."
        return self.output_names_

    @property
    def output_types(self):  # type: ignore
        "Returns the output types."
        return self.output_types_

    @property
    def opsets(self):  # type: ignore
        "Returns the opsets."
        return self.opsets_

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(self.input_names)}) -> {', '.join(self.output_names)}"

    def _init(self) -> None:
        """
        Loads the implementation for every node in the graph.
        """
        self.rt_inits_ = {}
        self.rt_nodes_ = []
        for init in self.inits_:
            self.rt_inits_[init.name] = numpy_helper.to_array(init)
        run_params = {
            "log": lambda pattern, *args: self._log(10, pattern, *args),
            "opsets": self.opsets,
            "verbose": self.verbose,
        }
        for node in self.nodes_:
            cl = self._load_impl(node)
            inst = cl(node, run_params)
            self.rt_nodes_.append(inst)

    def _load_impl(self, node: NodeProto) -> Any:
        """
        Loads the implementation for a specified runtime.
        """
        if node.domain not in self.opsets:
            raise RuntimeError(
                f"Domain {node.domain!r} (node type: {node.op_type!r}) "
                f"is not specified. Known opsets: {self.opsets!r}."
            )
        version = self.opsets[node.domain]
        key = node.domain, node.op_type
        if key in self.new_ops_:
            # This operator has a custom implementation.
            # This mechanism can be used to implement a custom onnx node
            # or to overwrite an existing one.
            return self.new_ops_[key]

        if node.domain == "":
            from .aionnx import load_op

            return load_op(node.domain, node.op_type, version)

        if node.domain == "ai.onnx.preview.training":
            from .aionnx_preview_training import load_op as load_op_pt

            return load_op_pt(node.domain, node.op_type, version)

        if node.domain == "experimental":
            from .experimental import load_op as load_op_exp

            return load_op_exp(node.domain, node.op_type, version)

        # It has to be a function.
        if key in self.functions_:
            from .aionnx import load_op

            impl = self.functions_[key]
            return load_op(node.domain, node.op_type, version, custom=impl)
        raise NotImplementedError(
            f"Node type {node.op_type!r} from domain {node.domain!r} "
            f"is unknown, known functions: {list(sorted(self.functions_))}."
        )

    def run(self, output_names, feed_inputs: Dict[str, Any], attributes: Dict[str, Any] = None):  # type: ignore
        """
        Executes the onnx model.

        :param output_names: requested outputs by names,
            None for all
        :param feed_inputs: dictionary `{ input name: input value }`
        :param attributes: attributes value if the instance runs a FunctionProto
        :return: list of requested outputs
        """
        if output_names is None:
            output_names = self.output_names
        if isinstance(self.proto_, FunctionProto) and attributes is None:
            raise TypeError()

        # step 1: inputs and initializers
        results = {"": None}  # optional input
        results.update(self.rt_inits_)
        results.update(feed_inputs)

        # step 2: execute nodes
        for node in self.rt_nodes_:
            self._log(1, "%s(%s) -> %s", node.op_type, node.input, node.output)
            inputs = [results[i] for i in node.input]
            linked_attributes = {}
            if node.has_linked_attribute and attributes:
                linked_attributes["linked_attributes"] = attributes
            if node.need_context():
                outputs = node.run(*inputs, context=results, **linked_attributes)
            else:
                outputs = node.run(*inputs, **linked_attributes)
            for name, value in zip(node.output, outputs):
                if isinstance(value, tuple):
                    raise TypeError(
                        f"Unexected type {type(value)} for output {name!r}."
                    )
                self._log(2, " + %s: %s", name, value)
                results[name] = value

        # return the results
        list_results: List[Any] = []
        for name in output_names:
            if name not in results:
                raise RuntimeError(
                    f"Unable to find output {name!r} in {list(sorted(results))}."
                )
            list_results.append(results[name])
        return list_results
