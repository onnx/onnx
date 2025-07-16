# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import numpy as np

import onnx

if TYPE_CHECKING:
    from collections.abc import Sequence


class RuntimeTypeError(RuntimeError):
    """Raised when a type of a variable is unexpected."""


class RuntimeContextError(RuntimeError):
    """Raised when the context is missing but an context dependent implementation is defined for an operator."""


class RuntimeImplementationError(NotImplementedError):
    """Raised when no implementation was found for an operator."""


class DefaultNone:
    """Default value for parameters when the parameter is not set but the operator has a default behavior for it."""


class RefAttrName:
    """Implements a link between a parameter of a function and an attribute in node.

    Args:
        name: name of the input
    """

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"


def _build_schemas() -> dict[str, onnx.defs.OpSchema]:
    res: dict[str, onnx.defs.OpSchema] = {}
    for schema in onnx.defs.onnx.defs.get_all_schemas_with_history():
        # Multiple version can coexist. The last one is kept.
        if schema.name in res:
            if schema.domain != res[schema.name].domain:
                raise NotImplementedError(
                    f"This function assumes every operator has a unique name {schema.name!r} "
                    f"even across multiple domains {schema.domain!r} and {res[schema.name].domain!r}."
                )
            if schema.since_version > res[schema.name].since_version:
                # We keep the most recent one.
                res[schema.name] = schema
        else:
            res[schema.name] = schema
        res[schema.name + "_" + str(schema.since_version)] = schema
    return res


_schemas = _build_schemas()


class OnnxType:
    def __init__(self, type_proto: onnx.TypeProto):
        if not isinstance(type_proto, onnx.TypeProto):
            raise TypeError(
                f"type_proto {type(type_proto)} must be of type onnx.TypeProto."
            )
        self.type_proto = type_proto

    def __repr__(self) -> str:
        return f"OnnxType({self.type_proto!r})"


class SparseTensor:
    """Simple representation of a sparse tensor.
    It is based on numpy but does not require scipy.
    """

    def __init__(
        self, values: np.ndarray, indices: np.ndarray, shape: tuple[int]
    ) -> None:
        self.values = values
        self.indices = indices
        self.shape = shape

    @property
    def dtype(self) -> Any:
        return self.values.dtype


def to_sparse_tensor(att: onnx.AttributeProto) -> SparseTensor:
    """Hosts a sparse tensor."""
    shape = tuple(d for d in att.dims)  # type: ignore[attr-defined]
    return SparseTensor(
        onnx.numpy_helper.to_array(att.values),  # type: ignore[attr-defined]
        onnx.numpy_helper.to_array(att.indices),  # type: ignore[attr-defined]
        shape,
    )


def _attribute_conversion_function(attr_type: onnx.AttributeProto.AttributeType):
    return {
        onnx.AttributeProto.FLOAT: lambda att: np.float32(att.f),
        onnx.AttributeProto.FLOATS: lambda att: [np.float32(f) for f in att.floats],
        onnx.AttributeProto.GRAPH: lambda att: Graph(att.g),
        onnx.AttributeProto.GRAPHS: lambda att: [Graph(g) for g in att.graphs],
        onnx.AttributeProto.INT: lambda att: int(att.i),
        onnx.AttributeProto.INTS: lambda att: [int(i) for i in att.ints],
        onnx.AttributeProto.SPARSE_TENSOR: lambda att: to_sparse_tensor(
            att.sparse_tensor
        ),
        onnx.AttributeProto.SPARSE_TENSORS: lambda att: [
            to_sparse_tensor(t) for t in att.sparse_tensors
        ],
        onnx.AttributeProto.STRING: lambda att: att.s.decode("utf-8"),
        onnx.AttributeProto.STRINGS: lambda att: [
            s.decode("utf-8") for s in att.strings
        ],
        onnx.AttributeProto.TENSOR: lambda att: onnx.numpy_helper.to_array(att.t),
        onnx.AttributeProto.TENSORS: lambda att: [
            onnx.numpy_helper.to_array(t) for t in att.tensors
        ],
        onnx.AttributeProto.TYPE_PROTO: lambda att: OnnxType(att.tp),
        onnx.AttributeProto.TYPE_PROTOS: lambda att: [
            OnnxType(t) for t in att.type_protos
        ],
    }[attr_type]


class Graph:
    __slots__ = ("g",)

    def __init__(self, g: onnx.GraphProto) -> None:
        self.g = g


class OpRun(abc.ABC):
    """Ancestor to all operators in this subfolder.

    Args:
        onnx_node: `onnx` node
        run_params: additional parameters such as `verbose`, `opsets`
            (it can be more than one if the operator has a subgraph),
            `log` for a logging function
        schema: operator schema
    """

    op_domain = ""

    def __init__(
        self, onnx_node: onnx.NodeProto, run_params: dict[str, Any], schema: Any = None
    ):
        if not isinstance(run_params, dict):
            raise TypeError(f"run_params must be a dictionary not {type(run_params)}.")
        for att in ["opsets", "new_ops"]:
            if att not in run_params:
                raise RuntimeError(
                    f"Attribute {att!r} must be in run_params, only "
                    f"{sorted(run_params)} was found."
                )
        if "log" not in run_params:
            raise KeyError("run_params must contains key 'log'.")
        self.onnx_node = onnx_node
        self.run_params = run_params
        if schema is None:
            if hasattr(self.__class__, "op_schema"):
                self._schema = self.__class__.op_schema
            elif self.__class__.__name__ in _schemas:
                self._schema = _schemas[self.__class__.__name__]
            elif onnx_node.op_type in _schemas:
                self._schema = _schemas[onnx_node.op_type]
            else:
                self._schema = None
        else:
            self._schema = schema
        self.has_subgraph = False
        self._load_attributes()

    def _log(self, pattern, *args):
        self.run_params["log"](pattern, *args)

    def _extract_attribute_value(
        self, att: onnx.AttributeProto, ref_att: onnx.AttributeProto | None = None
    ) -> Any:
        """Converts an attribute value into a python value."""
        if att.type == onnx.AttributeProto.GRAPH:
            new_ops = self.run_params.get("new_ops", None)
            if "existing_functions" in self.run_params:
                functions = list(self.run_params["existing_functions"].values())
            else:
                functions = None
            evaluator_cls = self.run_params.get("evaluator_cls", None)
            assert evaluator_cls is not None, (
                f"evaluator_cls must be specified to evaluate att={att}"
            )
            return evaluator_cls(
                att.g,
                opsets=self.run_params["opsets"],
                verbose=max(0, self.run_params.get("verbose", 0) - 2),
                new_ops=None if new_ops is None else list(new_ops.values()),
                functions=functions,
            )

        conversion_function = _attribute_conversion_function(att.type)
        if conversion_function is not None:
            return conversion_function(att)

        if ref_att is None:
            raise AttributeError(
                f"Unable to convert attribute {att.name!r} type {att.type!r} "
                f"from node type {self.onnx_node.op_type!r}, "
                f"domain {self.onnx_node.domain!r}\n{att}."
            )

        raise AttributeError(
            f"Unable to convert default value for {ref_att.name!r} type {att.type!r} "
            f"from node type {self.onnx_node.op_type!r}, "
            f"domain {self.onnx_node.domain!r}\n{att}\n{ref_att}."
        )

    @staticmethod
    def _evaluate_subgraph(context, value, attributes):
        return value.run(None, context or {}, attributes=attributes)

    def _load_attributes(self) -> None:
        """Checks and loads attributes."""
        self.has_linked_attribute = False
        added_attributes = []
        for att in self.onnx_node.attribute:
            name = att.name
            if att.ref_attr_name:
                value = RefAttrName(att.ref_attr_name)
                self.has_linked_attribute = True
            else:
                value = self._extract_attribute_value(att)
            setattr(self, name, value)
            added_attributes.append(name)
            if att.type == onnx.AttributeProto.GRAPH:
                self.has_subgraph = True
                self.has_linked_attribute |= value.has_linked_attribute  # type: ignore[attr-defined]
                setattr(
                    self,
                    f"_run_{att.name}",
                    lambda context,
                    value=value,
                    attributes=None: OpRun._evaluate_subgraph(
                        context, value, attributes
                    ),
                )

        if self._schema and self.onnx_node.op_type not in {"Constant"}:
            for k, v in self._schema.attributes.items():
                if not hasattr(self, k):
                    if getattr(v, "required", True):
                        raise RuntimeError(
                            f"Attribute {k!r} is expected based on ONNX specifications "
                            f"for node {self.onnx_node.op_type!r}."
                        )
                    if hasattr(v, "default_value"):
                        if v.default_value.type == 0 or (
                            v.default_value.type == 4  # noqa: PLR2004
                            and v.default_value.t.data_type == 0
                        ):
                            # default value is undefined, it depends on the inputs
                            value = None  # type: ignore[assignment]
                        else:
                            value = self._extract_attribute_value(v.default_value, v)
                        setattr(self, k, value)
                        added_attributes.append(k)
        self.attributes_names_ = set(added_attributes)

    @staticmethod
    def implicit_inputs(graph: onnx.GraphProto) -> list[str]:
        """Returns all variables not registered as inputs and not produced by
        an node inside the graph. This inputs are part of the context
        existing in the graph calling this one.
        """
        if not isinstance(graph, onnx.GraphProto):
            raise TypeError(f"Unexpected type {type(graph)!r}.")
        local = set()
        known = set()
        for init in graph.initializer:
            known.add(init.name)
        for sparse_init in graph.sparse_initializer:
            known.add(sparse_init.name)  # type: ignore[attr-defined]
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
    def input(self) -> Sequence[str]:
        """Returns node attribute `input`."""
        return self.onnx_node.input  # type: ignore[no-any-return]

    @property
    def output(self) -> Sequence[str]:
        """Returns node attribute `output`."""
        return self.onnx_node.output  # type: ignore[no-any-return]

    @property
    def op_type(self) -> str:
        """Returns node attribute `op_type`."""
        return self.onnx_node.op_type

    @property
    def domain(self) -> str:
        """Returns node attribute `domain`."""
        return self.onnx_node.domain

    def need_context(self) -> bool:
        """Tells the runtime if this node needs the context
        (all the results produced so far) as it may silently access
        one of them (operator Scan, If, Loop).
        The default answer is `False`.
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

    @abc.abstractmethod
    def _run(self, *args, **kwargs):
        """Should be overwritten.

        Args:
            *args: operator inputs
            **kwargs: optional inputs and overridden attributes, an
                attribute may be overridden if it belongs to a function,
                in this case, the same instance of OpRun can be called
                with different values of the same attribute.

        Returns:
            outputs
        """
        raise NotImplementedError(
            f"Method '_run' must be overwritten for operator {self.__class__.__name__!r}."
        )

    def _check_and_fix_outputs(self, res: tuple[Any, ...]) -> tuple[Any, ...]:
        """Checks the output are from the expected type."""
        if not isinstance(res, tuple):
            raise TypeError(
                f"Method '_run' of class {self.__class__.__name__!r} does not return a tuple but '{type(res)}'."
            )
        if not res:
            raise ValueError(
                f"Method '_run' of class {self.__class__.__name__!r} does not return any result."
            )
        if any(isinstance(t, tuple) for t in res):
            dtypes = [type(t) for t in res]
            raise TypeError(
                f"One of the results returned by method '_run' of class {self.__class__.__name__!r} "
                f"is a tuple, this is no ONNX corresponding type (Map, List, Tensor, SparseTensor). "
                f"All returned types: {dtypes!r}."
            )
        res = tuple(  # type: ignore[assignment]
            (np.array(x) if np.isscalar(x) else x) for x in res
        )
        if any(
            not (isinstance(t, (np.ndarray, list, dict)) or hasattr(t, "todense"))
            for t in res
        ):
            dtypes = [type(t) for t in res]
            raise TypeError(
                f"One of the results returned by method '_run' of class {self.__class__.__name__!r} "
                f"has an unexpected type, this is no ONNX corresponding type (Map, List, Tensor, SparseTensor). "
                f"All returned types: {dtypes!r}."
            )
        return res

    def run(self, *args, linked_attributes=None, context=None):
        """Calls method ``_run``, catches exceptions,
        displays a longer error message.

        Args:
            *args: inputs
            linked_attributes: used if this has an attriute linked to
                the attribute of the function it belongs to
            context: if this node is part of the subgraph, `context` is
                a dictionary with the values this node may use

        Returns:
            tuple of results
        """
        if self.need_context():
            if context is None:
                raise RuntimeError(
                    f"This node if type {type(self)} needs context to be filled."
                )
        elif context is not None:
            raise RuntimeError(
                f"This node if type {type(self)} does not need any contextbut one is given."
            )
        if self.has_linked_attribute and linked_attributes is None:
            raise ValueError(
                f"This node {type(self)} has linked attributes but None are given in parameter 'linked_attributes'."
            )
        if not self.has_linked_attribute and linked_attributes is not None:
            raise ValueError(
                f"This node {type(self)} has no linked attribute but some are given in parameter "
                f"'linked_attributes' {set(linked_attributes)}."
            )
        overridden_attributes = {}
        if self.has_linked_attribute:
            if linked_attributes is None:
                raise AttributeError(
                    f"One attribute is linked but no linked value is provided, "
                    f"in class {type(self)}."
                )
            for att in self.attributes_names_:
                v = getattr(self, att)
                if isinstance(v, RefAttrName):
                    if v.name not in linked_attributes:
                        raise ValueError(
                            f"Unable to find a value for linked attribute {att!r} in {linked_attributes!r} "
                            f"in node {type(self)}."
                        )
                    overridden_attributes[att] = linked_attributes[v.name]

        self._log("-- begin %s.run(%d inputs)", self.__class__.__name__, len(args))
        kwargs = {}
        for att in self.attributes_names_:
            if att in overridden_attributes:
                continue
            if not hasattr(self, att):
                raise NameError(
                    f"Attribute {att!r} is missing in operator {self.__class__.__name__!r}."
                )
            kwargs[att] = getattr(self, att)
        if self.has_subgraph:
            if self.has_linked_attribute and not linked_attributes:
                raise RuntimeError(
                    f"A subgraph has linked attribute but none was given to {type(self)}."
                )
            kwargs["attributes"] = linked_attributes
        if context is not None:
            kwargs["context"] = context
        try:
            if overridden_attributes:
                res = self._run(*args, **overridden_attributes, **kwargs)
            else:
                res = self._run(*args, **kwargs)
        except (TypeError, AttributeError) as e:
            raise TypeError(
                f"Issues with types {[type(_) for _ in args]} and attributes "
                f"{sorted(kwargs)} and linked attributes={sorted(overridden_attributes)} "
                f"(operator {self.__class__.__name__!r})."
            ) from e
        self._log(
            "-- done %s.run -> %d outputs",
            self.__class__.__name__,
            len(res) if res is not None else 0,
        )
        return self._check_and_fix_outputs(res)

    @classmethod
    def infer_name(cls):
        name = cls.__name__
        if "_" not in name:
            return name, onnx.defs.onnx_opset_version()
        name, vers = name.rsplit("_", 1)
        try:
            i_vers = int(vers)
        except ValueError:
            return cls.__name__, onnx.defs.onnx_opset_version()
        return name, i_vers

    @classmethod
    def make_node(
        cls,
        n_inputs: int | None = None,
        n_outputs: int | None = None,
        **kwargs: Any,
    ) -> onnx.NodeProto:
        """Creates an ONNX node for this class based on the given information.

        Args:
            n_inputs: number of inputs (default is defined by the
                operator schema)
            n_outputs: number of outputs (default is defined by the
                operator schema)
            verbose: verbosity
            **kwargs: node attributes

        Returns:
            NodeProto

        Method :meth:`eval <onnx.reference.op_run.OpRun.eval>` creates an onnx node
        returned by method :meth:`make_node <onnx.reference.op_run.OpRun.make_node>`.

        .. exec_code::

            import numpy as np
            from onnx.reference.ops._op_list import Celu

            onnx_node = Celu.make_node(alpha=0.5)
            print(onnx_node)
        """
        op_type, opset = cls.infer_name()
        domain = cls.op_domain
        schema = None
        if n_inputs is None:
            if schema is None:
                schema = onnx.defs.get_schema(op_type, opset, domain)
            n_inputs = schema.min_input
        if n_outputs is None:
            if schema is None:
                schema = onnx.defs.get_schema(op_type, opset, domain)
            n_outputs = schema.min_output

        names_in = [f"x{i}" for i in range(n_inputs)]
        names_out = [f"y{i}" for i in range(n_outputs)]
        node = onnx.helper.make_node(op_type, names_in, names_out, **kwargs)
        return node

    @classmethod
    def create(
        cls,
        n_inputs: int | None = None,
        n_outputs: int | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> Any:
        """Instantiates this class based on the given information.

        Args:
            n_inputs: number of inputs (default is defined by the
                operator schema)
            n_outputs: number of outputs (default is defined by the
                operator schema)
            verbose: verbosity
            **kwargs: node attributes

        Returns:
            NodeProto
        """

        def log_function(pattern: str, *args: Any) -> None:
            if verbose > 1:
                print(pattern % tuple(args))

        node = cls.make_node(n_inputs, n_outputs, **kwargs)
        run_params = {
            "verbose": verbose,
            "log": log_function,
            "new_ops": None,
            "opsets": {"": onnx.defs.onnx_opset_version()},
        }
        cl = cls(node, run_params)
        return cl

    @classmethod
    def eval(
        cls,
        *args: list[Any],
        n_outputs: int | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> Any:
        """Evaluates this operator.

        Args:
            *args: inputs
            n_outputs: number of outputs (default is defined by the
                operator schema)
            verbose: verbosity
            **kwargs: node attributes

        Returns:
            NodeProto
        """
        inst = cls.create(len(args), n_outputs=n_outputs, verbose=verbose, **kwargs)
        res = inst.run(*args)
        if len(res) == 1:
            return res[0]
        return res


class OpRunExpand(OpRun):
    """Class any operator to avoid must inherit from."""

    def __init__(self, *args, **kwargs):  # noqa: ARG002
        raise RuntimeError(
            f"The reference implementation must not use this node ({type(self)})."
        )

    def _run(self, *inputs, **kwargs):  # noqa: ARG002
        raise RuntimeError(
            f"The reference implementation must not use this node ({type(self)})."
        )


class OpFunction(OpRun):
    """Runs a custom function."""

    def __init__(
        self,
        onnx_node: onnx.NodeProto,
        run_params: dict[str, Any] | None,
        impl: Any = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if impl is None:
            raise RuntimeError(
                f"impl cannot be None for node type {onnx_node.op_type!r} "
                f"from domain {onnx_node.domain!r}."
            )
        OpRun.__init__(self, onnx_node, run_params)  # type: ignore[arg-type]
        self.impl_ = impl
        # The function implementation is the same whenever the function is called
        # but the attributes may be different at every call.
        self.attributes_ = {
            name: getattr(self, name)
            for name in getattr(self.impl_, "attributes_", attributes)  # type: ignore[union-attr]
        }

    def _run(self, *inputs, **kwargs):
        return self._run_impl(self.impl_, *inputs, **kwargs)

    def _run_impl(self, impl, *inputs, **kwargs):
        if len(impl.input_names) != len(inputs):
            raise RuntimeError(
                f"Mismatch lengths between the number of inputs {len(inputs)} "
                f"and the expected number of inputs {len(impl.input_names)} "
                f"for node {self.op_type!r} from domain {self.domain!r}."
            )
        feeds = dict(zip(impl.input_names, inputs))
        attributes = self.attributes_.copy()
        attributes.update(kwargs)
        results = impl.run(None, feeds, attributes=attributes)
        if len(impl.output_names) != len(results):
            raise RuntimeError(
                f"Mismatch lengths between the number of outputs {len(results)} "
                f"and the expected number of outputs {len(impl.output_names)} "
                f"for node {self.op_type!r} from domain {self.domain!r}."
            )
        return tuple(results)


class OpFunctionContextDependant(OpFunction):
    """The function can be instantiated but only at execution time.
    An instance of OpFunction is created everytime to node is executed.
    This is needed when the schema of an operator defines a context dependent function.
    """

    def __init__(
        self,
        onnx_node: onnx.NodeProto,
        run_params: dict[str, Any] | None,
        parent: Any = None,
    ):
        OpFunction.__init__(self, onnx_node, run_params, impl=self, attributes={})
        self.parent = parent
        version = parent.opsets[onnx_node.domain]
        self.schema_ = onnx.defs.get_schema(
            onnx_node.op_type, version, onnx_node.domain
        )

    def _run(self, *inputs, **kwargs):
        # Input types are known. They are used to properly
        # created the body for this operator.
        types = []
        for t in inputs:
            dtype = onnx.helper.np_dtype_to_tensor_dtype(t.dtype)
            types.append(onnx.helper.make_tensor_type_proto(dtype, t.shape))
        cl = self.parent._load_impl(self.onnx_node, types)
        inst = cl(self.onnx_node, self.run_params)
        return self._run_impl(inst.impl_, *inputs, **kwargs)
