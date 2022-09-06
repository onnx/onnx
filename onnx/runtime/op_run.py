# SPDX-License-Identifier: Apache-2.0
# pylint: disable=C0415

from typing import Any, Dict, Iterable, List

import numpy as np  # type: ignore

from ..onnx_pb import AttributeProto, GraphProto, NodeProto
from ..defs import get_all_schemas_with_history
from ..numpy_helper import to_array


class RuntimeTypeError(RuntimeError):
    """
    Raised when a type of a variable is unexpected.
    """


class DefaultNone:
    """
    Default value for parameters when the parameter is not set
    but the operator has a default behavior for it.
    """


class RefAttrName:
    """
    Implements a link between a parameter of a function
    and an attribute in node.

    :param name: name of the input
    """

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        "usual"
        return f"{self.__class__.__name__}({self.name!r})"


def _build_schemas() -> Dict[str, type]:
    res: Dict[str, type] = {}
    for schema in get_all_schemas_with_history():
        # Multiple version can coexist. The last one is kept.
        if schema.name in res:
            if schema.since_version > res[schema.name].since_version:  # type: ignore
                # We keep the most recent one.
                res[schema.name] = schema  # type: ignore
        else:
            res[schema.name] = schema  # type: ignore
        res[schema.name + "_" + str(schema.since_version)] = schema  # type: ignore
    return res


_schemas = _build_schemas()


class OpRun:
    """
    Ancestor to all operators in this subfolder.
    The runtime for every node can checked into
    `ONNX unit tests
    <https://github.com/onnx/onnx/tree/master/onnx/backend/test/case/node>`_.

    :param onnx_node: :epkg:`onnx` node
    :param log_function: function used to log information while
        executing the onnx graph
    """

    _attribute_conversion_functions = {
        AttributeProto.FLOAT: lambda att: np.float32(att.f),
        AttributeProto.FLOATS: lambda att: [np.float32(f) for f in att.floats],
        AttributeProto.INT: lambda att: int(att.i),
        AttributeProto.INTS: lambda att: [int(i) for i in att.ints],
        AttributeProto.TENSOR: lambda att: to_array(att.t),
        AttributeProto.STRING: lambda att: str(att.s),
        AttributeProto.STRINGS: lambda att: [str(s) for s in att.strings],
    }

    def __init__(
        self, onnx_node: NodeProto, run_params: Dict[str, Any], schema: Any = None
    ):
        if not isinstance(run_params, dict):
            raise TypeError(f"run_params must be a dictionary not {type(run_params)}.")
        if "log" not in run_params:
            raise KeyError("run_params must contains key 'log'.")
        self.onnx_node = onnx_node
        self.run_params = run_params
        if schema is None:
            if onnx_node.op_type in _schemas:
                self._schema = _schemas[onnx_node.op_type]
            else:
                self._schema = None  # type: ignore
        else:
            self._schema = schema
        self._load_attributes()

    def _log(self, pattern, *args):  # type: ignore
        self.run_params["log"](pattern, *args)

    def _extract_attribute_value(self, att: AttributeProto) -> Any:
        """
        Converts an attribute value into a python value.
        """
        if att.type == AttributeProto.GRAPH:
            from .inference import Inference  # type: ignore

            return Inference(att.g, opsets=self.run_params["opsets"])
        if att.type in OpRun._attribute_conversion_functions:
            return OpRun._attribute_conversion_functions[att.type](att)  # type: ignore
        raise NotImplementedError(
            f"Unable to convert attribute {att.name!r} type {att.type!r} "
            f"from node type {self.onnx_node.op_type!r}, "
            f"domain {self.onnx_node.domain!r}."
        )

    def _load_attributes(self) -> None:
        "Checks and loads attributes."
        for att in self.onnx_node.attribute:
            name = att.name
            if att.ref_attr_name:
                value = RefAttrName(att.ref_attr_name)
            else:
                value = self._extract_attribute_value(att)
            setattr(self, name, value)
            if att.type == AttributeProto.GRAPH:
                setattr(
                    self,
                    f"_run_{att.name}",
                    lambda context, value=value: value.run(None, context or {}),
                )

        if self._schema and self.onnx_node.op_type not in {"Constant"}:
            for k, v in self._schema.attributes.items():  # type: ignore
                if not hasattr(self, k):
                    if getattr(v, "required", True):
                        raise RuntimeError(
                            f"Attribute {k!r} is expected based on ONNX specifications "
                            f"for node {self.onnx_node.op_type!r}."
                        )
                    if hasattr(v, "default_value"):
                        name = k
                        value = self._extract_attribute_value(v.default_value)
                        setattr(self, k, value)

    @staticmethod
    def local_inputs(graph: GraphProto) -> List[str]:
        """
        Returns all varibles not registered as inputs and not produced by
        an node inside the graph. This inputs are part of the context
        existing in the graph calling this one.
        """
        if not isinstance(graph, GraphProto):
            raise TypeError(f"Unexpected type {type(graph)!r}.")
        local = set()
        known = set()
        for init in graph.initializer:
            known.add(init.name)
        for sparse_init in graph.sparse_initializer:
            known.add(sparse_init.name)  # type: ignore
        for inp in graph.input:
            known.add(inp.name)
        for node in graph.node:
            for o in node.output:
                known.add(o)
            for i in node.input:
                if i not in known:
                    local.add(i)
        return list(local)

    @property
    def input(self) -> Iterable[str]:
        "Returns node attribute `input`."
        return self.onnx_node.input  # type: ignore

    @property
    def output(self) -> Iterable[str]:
        "Returns node attribute `output`."
        return self.onnx_node.output  # type: ignore

    @property
    def op_type(self) -> str:
        "Returns node attribute `op_type`."
        return self.onnx_node.op_type  # type: ignore

    @property
    def domain(self) -> str:
        "Returns node attribute `domain`."
        return self.onnx_node.domain  # type: ignore

    def need_context(self) -> bool:
        """
        Tells the runtime if this node needs the context
        (all the results produced so far) as it may silently access
        one of them (operator Scan, If, Loop).
        The default answer is `False`.
        """
        return False

    def is_constant(self) -> bool:
        """
        Tells if the node is a constant.
        """
        return False

    def __str__(self) -> str:
        atts = [self.__class__.__name__ + "(", f"    op_type={self.onnx_node.op_type}"]
        for k, v in sorted(self.__dict__.items()):
            if k in {"desc", "onnx_node"}:
                continue
            if "a" <= k[0] <= "z" and k[-1] != "_":
                atts.append(f"    {k}={v},")
        atts.append(")")
        return "\n".join(atts)

    def _run(self, *args, **kwargs):  # type: ignore
        """
        Should be overwritten.
        """
        raise NotImplementedError(
            f"Method '_run' or 'to_python' should be overwritten for operator {self.__class__.__name__!r}."
        )

    def run(self, *args, **kwargs):  # type: ignore
        """
        Calls method ``_run``, catches exceptions,
        displays a longer error message.
        """
        self._log("-- begin %s.run(%d inputs)", self.__class__.__name__, len(args))
        try:
            res = self._run(*args, **kwargs)
        except (TypeError, AttributeError) as e:
            raise TypeError(
                f"Issues with types {[type(_) for _ in args]} "
                f"(operator {self.__class__.__name__!r})."
            ) from e
        self._log("-- done %s.run -> %d outputs", self.__class__.__name__, len(res))
        return res


class OpFunction(OpRun):
    """
    Runs a custom function.
    """

    def __init__(self, onnx_node: NodeProto, log_function: Any, impl: Any = None):
        if impl is None:
            raise RuntimeError(
                f"impl cannot be None for node type {onnx_node.op_type!r} "
                f"from domain {onnx_node.domain!r}."
            )
        OpRun.__init__(self, onnx_node, log_function)
        self.impl_ = impl
        # The function implementation is the same whenever the function is called
        # but the attributes may be different at every call.
        self.attributes_ = {
            name: getattr(self, name) for name in self.impl_.attributes_
        }

    def _run(self, *inputs):  # type: ignore # pylint: disable=W0221
        if len(self.impl_.input_names) != len(inputs):
            raise RuntimeError(
                f"Mismatch lengths between the number of inputs {len(inputs)} "
                f"and the expected number of inputs {len(self.impl_.inputs)} "
                f"for node {self.op_type!r} from domain {self.domain!r}."
            )
        feeds = dict(zip(self.impl_.input_names, inputs))
        results = self.impl_.run(None, feeds, attributes=self.attributes_)
        if len(self.impl_.output_names) != len(results):
            raise RuntimeError(
                f"Mismatch lengths between the number of outputs {len(results)} "
                f"and the expected number of outputs {len(self.impl_.output_names)} "
                f"for node {self.op_type!r} from domain {self.domain!r}."
            )
        return tuple(results)
