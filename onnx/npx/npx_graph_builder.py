# SPDX-License-Identifier: Apache-2.0
# pylint: disable=too-many-branches,protected-access,too-many-statements

from inspect import Parameter, signature
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from onnx import (  # pylint: disable=E0611
    IR_VERSION,
    AttributeProto,
    FunctionProto,
    ModelProto,
    NodeProto,
    TypeProto,
    ValueInfoProto,
)
from onnx.checker import C as onnxC
from onnx.checker import check_model, check_node, check_value_info
from onnx.defs import onnx_opset_version
from onnx.helper import (
    OP_SET_ID_VERSION_MAP,
    make_attribute,
    make_function,
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.npx.npx_constants import _OPSET_TO_IR_VERSION, FUNCTION_DOMAIN, ONNX_DOMAIN
from onnx.npx.npx_function_implementation import get_function_implementation
from onnx.npx.npx_helper import (
    iter_nodes,
    onnx_convert_model_for_opsets,
    onnx_model_to_function,
    rename_in_onnx_graph,
)
from onnx.npx.npx_types import (
    ElemType,
    OptParType,
    ParType,
    SequenceType,
    TensorType,
    TupleType,
)
from onnx.npx.npx_var import Cst, Input, ManyIdentity, Par, Var
from onnx.numpy_helper import from_array
from onnx.onnx_cpp2py_export.checker import (  # pylint: disable=E0611,E0401
    ValidationError,
)
from onnx.onnx_cpp2py_export.shape_inference import (  # pylint: disable=E0611,E0401
    InferenceError,
)
from onnx.shape_inference import infer_shapes


class _FunctionIO:
    """
    Wrapper around a string.

    :param name: name
    """

    def __init__(self, name):
        if not isinstance(name, str):
            raise TypeError(f"name is not a string but {type(name)} - {name!r}.")
        self.name = name

    def __str__(self):
        "usual"
        return f"{self.__class__.__name__}({self.name!r})"


class _GraphBuilder:
    """
    Intermediate class to build an onnx graph.

    :param target_opsets: dictionary `{ domain: version}`
    :param as_function: export as :class:`onnx.FunctionProto`
        or :class:`onnx.GraphProto`
    :param name: function name if *as_function* is True
    :param domain: function domain if *as_function* is True
    :param constraints: specifies a precise type for the type
        constraints when a function allows more than one type,
        this works if there is only one variable to be converted
    :param ir_version: defines the IR version to use ot build
        the ONNX graph
    """

    def __init__(
        self,
        target_opsets: Optional[Dict[str, int]] = None,
        as_function: bool = False,
        name: Optional[str] = None,
        domain: Optional[str] = None,
        attributes: Optional[List[str]] = None,
        constraints: Optional[Dict[Any, TensorType]] = None,
        ir_version: Optional[int] = None,
    ):
        if ir_version is None:
            if (
                target_opsets is not None
                and "" in target_opsets
                and target_opsets[""] in _OPSET_TO_IR_VERSION
            ):
                ir_version = _OPSET_TO_IR_VERSION[target_opsets[""]]
        if ir_version is None:
            raise ValueError(
                f"Not default value for ir_version and "
                f"target_opsets={target_opsets}. "
                f"ir_version must be defined."
            )

        self.target_opsets = (
            target_opsets if target_opsets is None else target_opsets.copy()
        )
        self.ir_version = ir_version

        check_opsets = target_opsets or {"": onnx_opset_version()}
        main_opset = check_opsets.get("", None)
        if domain is not None and domain not in check_opsets:
            check_opsets[domain] = 1
        self.check_context = onnxC.CheckerContext()
        self.check_context.opset_imports = check_opsets
        self.check_context.ir_version = (
            OP_SET_ID_VERSION_MAP.get(main_opset, IR_VERSION)
            if main_opset is not None
            else IR_VERSION
        )

        self.as_function = as_function
        self.constraints = constraints
        if as_function:
            if name is None:
                raise ValueError("name cannot be None if as_function is specified.")
            if domain is None:
                raise ValueError("domain cannot be None if as_function is specified.")
        self.function_name = name
        self.function_domain = domain
        self.attributes = attributes
        self._names = set()
        self._id_vars = {}
        self._vars = []

    def _unique(self, prefix):
        if prefix in ("", None):
            prefix = "r"
        if "__" in prefix:
            raise NameError("prefix {prefix!r} cannot contain '__'.")
        name = f"{prefix}__{len(self._names)}"
        self._names.add(name)
        return name

    def append(self, var):
        "Appends an instruction to the list."
        i = id(var)
        for index in range(var.n_var_outputs):
            if (i, index) in self._id_vars:
                # an input or result used twice
                return
            self._id_vars[i, index] = None
        self._vars.append(var)

    def add_function(
        self, key: Tuple[str, str], values: Tuple[FunctionProto, Any, Any, Any]
    ):
        if not isinstance(values, tuple):
            raise TypeError(f"values must be a tuple not {type(values)}.")
        if len(values) != 4:
            raise TypeError(f"values must have 4 elements not {len(values)}.")
        if key in self.functions_:
            f1 = self.functions_[key][0].SerializeToString()
            f2 = values[0].SerializeToString()
            if f1 == f2:
                return
            raise KeyError(
                f"Function {key!r} is already registered and "
                f"the definition is not the same. Registered functions: "
                f"{list(sorted(self.functions_))}."
            )
        self.functions_[key] = values

    def _reset(self):
        self.inputs_ = []  # pylint: disable=attribute-defined-outside-init
        self.outputs_ = []  # pylint: disable=attribute-defined-outside-init
        self.nodes_ = []  # pylint: disable=attribute-defined-outside-init
        self.functions_ = {}  # pylint: disable=attribute-defined-outside-init
        self.attributes_ = []  # pylint: disable=attribute-defined-outside-init
        self.onnx_names_ = {}  # pylint: disable=attribute-defined-outside-init

    def make_node(
        self,
        op: str,
        inputs,
        outputs,
        domain: str = "",
        opset: int = 1,
        attribute_protos=None,
        **kwargs,
    ):
        """
        Inserts a node in the graph.
        """
        if self.target_opsets is not None and self.target_opsets.get(domain, 1) < opset:
            raise ValueError(
                f"opset value is too low: opset={opset} <= "
                f"{self.target_opsets.get(domain, 1)} "
                f"for domain={domain!r} and op={op!r}."
            )
        # checks inputs are known
        for i, inp in enumerate(inputs):
            if inp and inp not in self.onnx_names_:
                names = "\n".join(sorted(self.onnx_names_))
                raise RuntimeError(
                    f"Input {i} {inp!r} of node {op!r} does not exist in "
                    f"function {self.function_name!r} from domain "
                    f"{self.function_domain!r}. Known names:\n{names}\n."
                )

        new_kwargs = {}
        protos = []
        for k, v in kwargs.items():
            if isinstance(v, Par):
                if self.as_function:
                    att = AttributeProto()
                    att.name = k
                    att.ref_attr_name = v.name
                    try:
                        att.type = v.onnx_type
                    except TypeError as e:
                        raise TypeError(f"Unexected type {v.onnx_type}: {v}.") from e
                    protos.append(att)
                elif v.value is not None:
                    new_kwargs[k] = v.value
            else:
                new_kwargs[k] = v

        # make node
        if op == "Identity" and (len(inputs) != 1 or len(outputs) != 1):
            raise RuntimeError(
                f"Cannot create a node Identity for {len(inputs)} input(s) and "
                f"{len(outputs)} output(s)."
            )
        node = make_node(op, inputs, outputs, domain=domain, **new_kwargs)
        for p in protos:
            node.attribute.append(p)
        if attribute_protos is not None:
            for att in attribute_protos:
                node.attribute.append(att)

        for out in outputs:
            if out:
                self.onnx_names_[out] = node

        # check context
        context = self.check_context
        if domain is not None and domain not in context.opset_imports:
            d = dict(self.check_context.opset_imports)
            d[domain] = opset
            context = onnxC.CheckerContext()
            context.opset_imports = d
            context.ir_version = self.check_context.ir_version
        try:
            check_node(node, context)
        except ValidationError as e:
            raise RuntimeError(f"Node type {node.op_type!r} is wrong ({node})") from e
        self.nodes_.append(node)

    def _io(
        self, index: int, name: str, tensor_type: Optional[type], is_input: bool
    ) -> ValueInfoProto:
        """
        Converts an input or outut into :class:`onnx.ValueInfoProto`.

        :param index: index of the input or output to add
        :param name: input or output name
        :param tensor_type: type of the tensor
        :param is_input: True to tell *name* is an input, False
            for an output
        :return: an instance of :class:`ValueInfoProto`
        """
        if self.as_function:
            return _FunctionIO(name)
        if tensor_type is not None and not issubclass(tensor_type, TensorType):
            raise TypeError(
                f"Unexpected type {tensor_type.type_name()} for tensor_type. "
                f"This may happen if you specialised the function based on "
                f"contraints and not on input."
            )
        if self.constraints is not None:
            if is_input and index in self.constraints:
                new_type = self.constraints[index]
            elif (index, is_input) in self.constraints:
                new_type = self.constraints[index, is_input]
            elif name in self.constraints:
                new_type = self.constraints[name]
            elif tensor_type is not None and tensor_type.name in self.constraints:
                new_type = self.constraints[tensor_type.name]
            elif is_input:
                raise RuntimeError(
                    f"tensor_type is not specific enough (tensor_type={tensor_type!r}) "
                    f"and constraints do not precise this type for "
                    f"{'input' if is_input else 'output'} {index} "
                    f"with name={name!r} and constraints={self.constraints!r}."
                )
            else:
                new_type = None
            if tensor_type is not None and new_type is not None:
                if not tensor_type.issuperset(new_type):
                    exc = True
                    if tensor_type.dtypes == new_type.dtypes:
                        # shape are different, we keep the most
                        # restrictive one
                        if new_type.issuperset(tensor_type):
                            new_type = tensor_type
                            exc = False
                    if exc and is_input:
                        raise RuntimeError(
                            f"tensor_type is not specific enough {tensor_type!r} "
                            f"and constraint={new_type!r} and not consistent for "
                            f"{'input' if is_input else 'output'} {index} "
                            f"with name={name!r}."
                        )
            tensor_type = new_type
        if tensor_type is None:
            if is_input:
                raise RuntimeError(
                    f"tensor_type cannot be None for name={name!r} and "
                    f"input or output {index}."
                )
            tensor_type = TensorType["undefined"]
        if len(tensor_type.dtypes) != 1:
            raise RuntimeError(
                f"tensor_type is not specific enough ({str(tensor_type)} "
                f"or its full representation {tensor_type!r})."
            )
        if tensor_type.shape is None:
            type_proto = TypeProto()
            tensor_type_proto = type_proto.tensor_type
            tensor_type_proto.elem_type = tensor_type.dtypes[0].dtype
            value_info_proto = ValueInfoProto()
            value_info_proto.name = name
            # tensor_type_proto.shape.dim.extend([])
            value_info_proto.type.CopyFrom(type_proto)
            info = value_info_proto
        else:
            # Every runtime must allow inputs of different shapes but
            # with fixed rank. This can be changed here and in methods `make_key`.
            shape = [None for _ in tensor_type.shape]
            info = make_tensor_value_info(name, tensor_type.dtypes[0].dtype, shape)
            # check_value_info fails if the shape is left undefined
            check_value_info(info, self.check_context)
        return info

    def make_input(self, name: str, tensor_type: type):
        """
        Inserts a node in the graph.
        """
        if name is None or len(name) == 0:
            raise RuntimeError(
                f"Empty input name in function {self.function_name!r} "
                f"from domain {self.function_domain!r}."
            )
        existing_names = {i.name for i in self.inputs_}
        if name not in existing_names:
            self.inputs_.append(self._io(len(self.inputs_), name, tensor_type, True))
        self.onnx_names_[name] = None

    def make_output(self, name: str, tensor_type: type):
        """
        Inserts a node in the graph.
        """
        if name is None or len(name) == 0:
            raise RuntimeError(
                f"Empty output name in function {self.function_name!r} "
                f"from domain {self.function_domain!r}."
            )
        self.outputs_.append(self._io(len(self.outputs_), name, tensor_type, False))

    def _make_onnx(self):
        """
        Makes the final onnx.
        """
        if self.target_opsets is None:
            opset_imports = [make_opsetid("", onnx_opset_version())]
        else:
            opset_imports = [make_opsetid(k, v) for k, v in self.target_opsets.items()]
        set_domains = set(d.domain for d in opset_imports)
        for f in self.functions_.values():
            domain = f[0].domain
            if domain not in set_domains:
                set_domains.add(domain)
                opset_imports.append(make_opsetid(domain, 1))

        # adds missing domain
        only_domains = set()
        for node in iter_nodes(self.nodes_):
            only_domains.add(node.domain)
            if node.domain not in set_domains:
                set_domains.add(node.domain)
                opset_imports.append(make_opsetid(node.domain, 1))
        opset_imports = [d for d in opset_imports if d.domain in only_domains]

        if self.as_function:
            inputs = []
            for i, inp in enumerate(self.inputs_):
                name = inp.name
                if name is None:
                    raise RuntimeError(
                        f"Input {i} is None for function " f"{self.function_name!r}."
                    )
                inputs.append(name)

            fct = make_function(
                self.function_domain,
                self.function_name,
                inputs,
                [o.name for o in self.outputs_],
                self.nodes_,
                opset_imports,
                (
                    None
                    if self.attributes is None
                    else [p.name for p in self.attributes]
                ),
            )
            return fct

        graph = make_graph(self.nodes_, "npx", self.inputs_, self.outputs_)
        model = make_model(
            graph,
            opset_imports=opset_imports,
            functions=list(f[0] for f in self.functions_.values()),
            ir_version=self.ir_version,
        )
        try:
            check_model(model)
        except ValidationError as e:
            if "Field 'shape' of 'type' is required but missing" in str(e):
                # checker does like undefined shape
                pass
            else:
                raise RuntimeError(f"Model is not valid\n{model}") from e
        has_undefined = 0 in set(
            o.type.tensor_type.elem_type for o in model.graph.output
        )
        if has_undefined:
            # an output has undefined type, run shape inference to fix it
            try:
                shapes = infer_shapes(model)
            except InferenceError as e:
                raise RuntimeError(
                    f"Unable to determine output shape of\n{model}"
                ) from e
            model = shapes
            if model.graph.value_info:
                # let's remove unnecessary information
                del model.graph.value_info[:]
        return model

    def _function_to_onnx(self, fct: Callable, n_inputs: int, n_outputs: int):
        """
        Converts a function to onnx.

        :param fct: a function
        :param n_inputs: number of inputs, needed information in case
            there is an undefined number of inputs
        """
        sig = signature(fct)
        if any(
            map(
                lambda t: issubclass(t.annotation, SequenceType),
                sig.parameters.values(),
            )
        ):
            # onnx does not allow undefined number of inputs
            key = fct.__module__, fct.__name__, n_inputs
        else:
            key = fct.__module__, fct.__name__
        if key in self.functions_:
            return self.functions_[key]
        domain = fct.__module__

        inputs = []
        input_types = []
        kwargs = {}
        attributes = []
        for idx, (name, par) in enumerate(sig.parameters.items()):
            value = par.default
            anno = par.annotation
            if not issubclass(
                anno,
                (ElemType, OptParType, ParType, SequenceType, TensorType, TupleType),
            ):
                raise TypeError(
                    f"Annotation must of a known not {type(anno)} for "
                    f"parameter {name!r} in function {fct.__name__!r}."
                )
            if issubclass(anno, SequenceType):
                # undefined number of parameters
                for i in range(idx, n_inputs):
                    new_name = f"{name}:{i - idx}"
                    inputs.append(Input(new_name))
                    input_types.append(anno.elem_type)
                continue
            if value == Parameter.empty or value is None:
                inputs.append(Input(name))
            else:
                p = Par(name, anno, value, parent_op=(fct.__module__, fct.__name__, 1))
                kwargs[name] = p
                attributes.append(p)
            input_types.append(anno)

        if issubclass(sig.return_annotation, TupleType):
            if sig.return_annotation.len() != n_outputs:
                raise TypeError(
                    f"Mismatched number of outputs {sig.return_annotation.len()} "
                    f"!= n_outputs={n_outputs} for fct={fct}."
                )
            output_types = [sig.return_annotation[i] for i in range(n_outputs)]
        elif n_outputs != 1:
            raise TypeError(
                f"Inconsistency between return type {sig.return_annotation} "
                f"and n_outputs={n_outputs} for fct={fct}."
            )
        else:
            output_types = [sig.return_annotation]
        applied = fct(*inputs, **kwargs)
        name_fct = fct.__name__ if len(key) == 2 else f"{fct.__name__}_{n_inputs}"

        onx = applied.to_onnx(
            self.target_opsets,
            as_function=True,
            name=name_fct,
            domain=domain,
            attributes=attributes,
        )
        if isinstance(onx, list):
            # This function calls other functions.
            if len(onx) != 2:
                raise RuntimeError(f"onx is a list with {len(onx)} elements.")
            d = onx[0]
            for k, v in d.items():
                self.add_function(k, v)
            onx = onx[1]
        self.add_function(key, (onx, input_types, output_types, attributes))
        return onx, input_types, output_types, attributes

    def _to_onnx_make_node(self, domop, node_inputs, node_outputs, kwargs):
        if domop == ("", "Identity") and len(node_inputs) > 1:
            if len(node_inputs) != len(node_outputs):
                raise RuntimeError(
                    f"Mismatch between {node_inputs} and {node_outputs}."
                )
            for ni, no in zip(node_inputs, node_outputs):
                self.make_node(
                    domop[1],
                    [ni],
                    [no],
                    domain=domop[0],
                    opset=self.target_opsets[""],
                    **kwargs,
                )
        elif domop[0] == FUNCTION_DOMAIN:
            proto = get_function_implementation(
                domop, node_inputs, node_outputs, opsets=self.target_opsets, **kwargs
            )
            self.add_function(
                domop,
                (
                    proto,
                    (None for i in node_inputs),
                    (None for i in node_outputs),
                    list(sorted(kwargs)),
                ),
            )
            self.make_node(
                proto.name,
                node_inputs,
                node_outputs,
                domain=proto.domain,
                opset=1,
                **{k: v for k, v in kwargs.items() if k in proto.attribute},
            )
        elif domop[0] == ONNX_DOMAIN:
            if isinstance(domop[1], NodeProto):
                node = domop[1]
                repls = dict(zip(node.input, node_inputs))
                atts = []
                for att in node.attribute:
                    if (
                        att.type == AttributeProto.GRAPH
                        and hasattr(att, "g")
                        and att.g is not None
                    ):
                        new_g = rename_in_onnx_graph(att.g, repls)
                        if new_g is None:
                            atts.append(att)
                            continue
                        att = make_attribute(
                            att.name, new_g
                        )  # pylint: disable=undefined-variable
                    atts.append(att)

                self.make_node(
                    node.op_type,
                    node_inputs,
                    node_outputs,
                    domain=node.domain,
                    attribute_protos=atts,
                )
            elif isinstance(domop[1], FunctionProto):
                fct = domop[1]
                key = fct.domain, fct.name
                self.add_function(
                    key,
                    (
                        fct,
                        (None for i in node_inputs),
                        (None for i in node_outputs),
                        [],
                    ),
                )
                self.make_node(fct.name, node_inputs, node_outputs, domain=fct.domain)
            elif isinstance(domop[1], ModelProto):
                onnx_convert_model_for_opsets(
                    domop[1], target_opsets=self.target_opsets
                )
                if "name" not in kwargs or kwargs["name"] is None:
                    raise ValueError(
                        "Parameter 'name' must be specified when "
                        "calling function 'compute'."
                    )
                name = kwargs["name"]
                domain = kwargs.get("domain", "LOCAL")
                key = domain, name
                if key in self.functions_:
                    raise ValueError(f"Function {key!r} was already added.")
                f1, fs = onnx_model_to_function(
                    domop[1], name=name, domain=domain, opset_imports=self.target_opsets
                )
                # needed functions are added first
                if fs is not None and len(fs) > 0:
                    for f in fs:
                        keyf = f.domain, f.name
                        if keyf in self.functions_:
                            raise ValueError(f"Function {keyf!r} was already added.")
                        self.add_function(
                            keyf,
                            (
                                f,
                                (None for i in f.input),
                                (None for i in f.output),
                                list(f.attribute),
                            ),
                        )
                # then the main function is added
                self.add_function(
                    key,
                    (f1, (None for i in node_inputs), (None for i in node_outputs), []),
                )
                self.make_node(name, node_inputs, node_outputs, domain=domain)
            else:
                raise TypeError(f"Unexpected proto type {type(domop[1])!r}.")

        else:
            self.make_node(
                domop[1],
                node_inputs,
                node_outputs,
                domain=domop[0],
                opset=self.target_opsets[domop[0] or ""],
                **kwargs,
            )

    def to_onnx(
        self, output_vars: Optional[List[Var]] = None
    ) -> Union[FunctionProto, ModelProto]:
        """
        Conversion to onnx.

        :param output_vars: list of :class:`Var` holding the final outputs
        :return: onnx graph
        """
        # _GraphBuilder.to_onnx
        self._reset()
        possible_inputs = []
        possible_outputs = []
        possible_types = []

        for var in self._vars:
            key = id(var)

            if isinstance(var, Cst):
                name = self._unique(var._prefix)
                self._id_vars[key, 0] = name
                self.make_node(
                    "Constant",
                    [],
                    [name],
                    value=from_array(var.inputs[0]),
                    opset=self.target_opsets[""],
                )
                self.onnx_names_[name] = var
                continue

            if isinstance(var, Input):
                name = var.name or self._unique(var._prefix)
                self._id_vars[key, 0] = name
                self.onnx_names_[name] = var
                possible_inputs.append((var, 0, None))
                continue

            out_types = None
            if isinstance(var, ManyIdentity):
                # an operator
                domop = ("", "Identity")
                att_types = None
                for v, ind in zip(var.inputs, var.input_indices):
                    inp = v, ind
                    possible_types.append((var, 0, inp))
            elif var.onnx_op[0] is None:
                # a function is converted into FunctionProto
                # and then a node is inserted in the main graph
                packed = self._function_to_onnx(
                    var.onnx_op[1], len(var.inputs), var.n_var_outputs
                )
                (onx_fn, in_types, out_types, att_types) = packed
                domop = (onx_fn.domain, onx_fn.name)

                for inp, index, dt in zip(var.inputs, var.input_indices, in_types):
                    if isinstance(inp, Input):
                        possible_types.append((inp, index, dt))
                for i, o in enumerate(out_types):
                    if isinstance(o, TupleType):
                        possible_types.append((var, i, o[i]))
                    else:
                        possible_types.append((var, i, o))
            else:
                # an operator
                domop = var.onnx_op
                att_types = None
                if domop == ("", "Identity"):
                    inp = var.inputs[0], var.input_indices[0]
                    possible_types.append((var, 0, inp))

            # an operator is to be inserted
            # preprocess the inputs
            node_inputs = []
            node_outputs = []
            for i, index in zip(var.inputs, var.input_indices):
                if i is None:
                    # optional input
                    node_inputs.append("")
                    continue
                if isinstance(i, Var):
                    kv = id(i)
                    if (kv, index) not in self._id_vars or self._id_vars[
                        kv, index
                    ] is None:
                        raise RuntimeError(
                            f"A variable of type {type(i)} id={kv} "
                            f"index={index} was not registered, i={i}."
                        )
                    input_name = self._id_vars[kv, index]
                    node_inputs.append(input_name)
                    continue

                if isinstance(i, np.ndarray):
                    c = Cst(i)
                    input_name = self._unique(var._prefix)
                    self._id_vars[id(i), index] = input_name
                    self._id_vars[id(c), index] = input_name
                    self.make_node(
                        "Constant",
                        [],
                        [input_name],
                        value=from_array(i),
                        opset=self.target_opsets[""],
                    )
                    self.onnx_names_[input_name] = c
                    node_inputs.append(input_name)
                    continue

                if isinstance(i, (int, float)):
                    ni = np.array(i)
                    c = Cst(ni)
                    input_name = self._unique(var._prefix)
                    self._id_vars[id(i), index] = input_name
                    self._id_vars[id(c), index] = input_name
                    self.make_node(
                        "Constant",
                        [],
                        [input_name],
                        value=from_array(ni),
                        opset=self.target_opsets[""],
                    )
                    self.onnx_names_[input_name] = c
                    node_inputs.append(input_name)
                    continue

                raise NotImplementedError(
                    f"Unexpected type {type(i)} for node={domop}."
                )

            # preprocess the argument
            kwargs = var.onnx_op_kwargs

            key = id(var)

            if var.n_var_outputs == 1:
                name = self._unique(var._prefix or "r")
                self._id_vars[key, 0] = name
                node_outputs = [name]
            else:
                node_outputs = []
                for no in range(var.n_var_outputs):
                    name = self._unique(f"{var._prefix or 'rm'}{no}")
                    node_outputs.append(name)
                    self._id_vars[key, no] = name

            # creates the node
            if att_types is not None and len(att_types) > 0:
                # functions do not accept default values,
                # all of them need to be defined or added
                # with the default value
                for par in att_types:
                    if par.name in kwargs:
                        continue
                    if par.value is None:
                        raise RuntimeError(
                            f"Default value for parameter {par.name!r} "
                            f"of function {domop[1]!r} and domain "
                            f"{domop[0]!r}."
                        )
                    kwargs[par.name] = par.value

            self._to_onnx_make_node(domop, node_inputs, node_outputs, kwargs)

        # the output is the last variable
        last_vars = output_vars or [self._vars[-1]]
        possible_outputs = []
        for var in last_vars:  # pylint: disable=consider-using-enumerate
            if isinstance(var, ManyIdentity):
                for i in range(len(var)):  # pylint: disable=C0200
                    possible_outputs.append((var[i], var.input_indices[i], None))
            else:
                possible_outputs.extend(
                    [(var, i, None) for i in range(var.n_var_outputs)]
                )

        if len(possible_types) > 0:
            # converts possibles types into a dictionary
            map_types = {}
            for var, i, dt in possible_types:
                if isinstance(dt, tuple):
                    # shortcut to pass the type along an identity node
                    ref, ind = dt
                    k = id(ref), ind
                    if k in map_types:
                        map_types[id(var), i] = map_types[k]
                    continue
                map_types[id(var), i] = dt

            # replace input types when known
            new_possible_inputs = []
            for var, index, dt in possible_inputs:
                if dt is None and (id(var), index) in map_types:
                    dt = map_types[id(var), index]
                new_possible_inputs.append((var, index, dt))
            possible_inputs = new_possible_inputs

            # replace output types when known
            new_possible_outputs = []
            for var, index, dt in possible_outputs:
                if dt is None and not self.as_function:
                    if isinstance(var, ManyIdentity):
                        raise RuntimeError("Cannot add multiple variables.")
                    if isinstance(var, Var):  # pylint: disable=consider-using-get
                        k = id(var), index
                        if k in map_types:  # pylint: disable=R1715
                            dt = map_types[k]
                    else:
                        k = id(var[0]), var[1]
                        if k in map_types:  # pylint: disable=R1715
                            dt = map_types[k]
                new_possible_outputs.append((var, index, dt))
            possible_outputs = new_possible_outputs

        for inp, index, dt in possible_inputs:
            self.make_input(self._id_vars[id(inp), index], dt)
        for out, index, dt in possible_outputs:
            self.make_output(self._id_vars[id(out), index], dt)
        onx = self._make_onnx()
        return onx
