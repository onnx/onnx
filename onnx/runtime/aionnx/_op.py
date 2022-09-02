# SPDX-License-Identifier: Apache-2.0

import numpy as np
from onnx import TensorProto, GraphProto
from onnx.defs import get_all_schemas_with_history


def _build_schemas():
    res = {}
    for schema in get_all_schemas_with_history():
        # Multiple version can coexist. The last one is kept.
        if schema.name in res:
            if schema.since_version > res[schema.name].since_version:
                # We keep the most recent one.
                res[schema.name] = schema
        else:
            res[schema.name] = schema
        res[schema.name + '_' + str(schema.since_version)] = schema
    return res


_schemas = _build_schemas()


class RuntimeTypeError(RuntimeError):
    """
    Raised when a type of a variable is unexpected.
    """
    pass


class DefaultNone:
    """
    Default value for parameters when the parameter is not set
    but the operator has a default behaviour for it.
    """
    pass


class RefAttrName:
    """
    Implements a link between a parameter of a function
    and an attribute in node.

    :param name: name of the input
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        "usual"
        return f"{self.__class__.__name__}({self.name!r})"


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
        TensorProto.FLOAT: lambda att: np.float32(att.f),
        TensorProto.INT64: lambda att: np.float32(att.i),
    }

    def __init__(self, onnx_node, log_function):
        self.onnx_node = onnx_node
        self.log_function = log_function
        if onnx_node.op_type in _schemas:
            self._schema = _schemas[onnx_node.op_type]
        else:
            self._schema = None
        self._load_attributes()

    def _extract_attribute_value(self, att):
        """
        Converts an attribute value into a python value.
        """
        if att.type in OpRun._attribute_conversion_functions:
            return OpRun._attribute_conversion_functions[att.type](att)
        raise NotImplementedError(
            f"Unable to convert attribute {att.name!r} type {att.type!r} "
            f"from node type {self.onnx_node.op_type!r}, "
            f"domain {self.onnx_node.domain!r}.")

    def _load_attributes(self):
        "Checks and loads attributes."
        for att in self.onnx_node.attribute:
            name = att.name
            value = self._extract_attribute_value(att)
            setattr(self, name, value)

        if self._schema and self.onnx_node.op_type not in {'Constant'}:
            for k, v in self._schema.attributes.items():
                if not hasattr(self, k) and getattr(v, 'required', True):
                    raise RuntimeError(
                        "Attribute '{}' is expected based on ONNX specifications "
                        "for node '{}'.".format(k, self.onnx_node.op_type))

    @staticmethod
    def local_inputs(graph):
        """
        Returns all varibles not registered as inputs and not produced by
        an node inside the graph. This inputs are part of the context
        existing in the graph calling this one.
        """
        if not isinstance(graph, GraphProto):
            raise TypeError(
                f"Unexpected type {type(graph)!r}.")
        local = set()
        known = set()
        for init in graph.initializer:
            known.add(init.name)
        for init in graph.input:
            known.add(init.name)
        for node in graph.node:
            for o in node.output:
                known.add(o)
            for i in node.input:
                if i not in known:
                    local.add(i)
        return list(local)

    @property
    def input(self):
        "Returns node attribute `input`."
        return self.onnx_node.input

    @property
    def output(self):
        "Returns node attribute `output`."
        return self.onnx_node.output

    @property
    def op_type(self):
        "Returns node attribute `op_type`."
        return self.onnx_node.op_type

    @property
    def domain(self):
        "Returns node attribute `domain`."
        return self.onnx_node.domain

    def need_context(self):
        """
        Tells the runtime if this node needs the context
        (all the results produced so far) as it may silently access
        one of them (operator Scan, If, Loop).
        The default answer is `False`.
        """
        return False

    def __str__(self):
        atts = [self.__class__.__name__ + '(',
                f"    op_type={self.onnx_node.op_type}"]
        for k, v in sorted(self.__dict__.items()):
            if k in {'desc', 'onnx_node'}:
                continue
            if 'a' <= k[0] <= 'z' and k[-1] != '_':
                atts.append(f'    {k}={v},')
        atts.append(')')
        return "\n".join(atts)

    def _run(self, *args, attributes=None, **kwargs):
        """
        Should be overwritten.
        Parameter *attributes* is used by functions.
        """
        raise NotImplementedError(
            "Method '_run' or 'to_python' should be overwritten for operator %s."
            "" % self.__class__.__name__)

    def run(self, *args, **kwargs):
        """
        Calls method ``_run``, catches exceptions,
        displays a longer error message.
        """
        try:
            res = self._run(*args, **kwargs)
        except TypeError as e:
            raise TypeError(
                "Issues with types {} (operator {}).".format(
                    ", ".join(str(type(_)) for _ in args),
                    self.__class__.__name__)) from e
        except AttributeError as e:
            raise AttributeError(
                "Issues with types {} (operator {}).".format(
                    ", ".join(str(type(_)) for _ in args),
                    self.__class__.__name__)) from e
        return res

    @property
    def args_default(self):
        """
        Returns the list of arguments as well as
        the list of parameters with the default values
        (close to the signature).
        """
        inps = []
        if hasattr(self, 'atts'):
            for k, v in self.atts.items():
                if isinstance(v, (list, tuple, dict)) and len(v) == 0:
                    v = None
                inps.append(f'{k}={v!r}')
        return inps

    @property
    def args_default_modified(self):
        """
        Returns the list of modified parameters.
        """
        if not hasattr(self, 'atts'):
            return None

        inps = []
        for k, v in self.atts.items():
            val = getattr(self, k, None)
            if isinstance(val, np.ndarray) and isinstance(v, list):
                val = list(val)
            try:
                if val != v:
                    inps.append(f'{k}={val!r}')
            except ValueError as e:
                raise ValueError(
                    f"Unexpected value for v={v!r} and val={val!r}.") from e
        return inps

    @property
    def args_optional(self):
        """
        Returns the list of optional arguments.
        """
        inps = []
        if hasattr(self, 'optional_inputs'):
            for k, v in self.optional_inputs.items():
                inps.append(f'{k}={v!r}')
        return inps

    @property
    def args_mandatory(self):
        """
        Returns the list of optional arguments.
        """
        if hasattr(self, 'mandatory_inputs'):
            return self.mandatory_inputs
        return None

    @property
    def atts_value(self):
        "Returns all parameters in a dictionary."
        if hasattr(self, 'atts'):
            return {k: getattr(self, k)
                    for k in self.atts}
        return None


class OpFunction(OpRun):
    """
    Runs a custom function.
    """
    def __init__(self, onnx_node, log_function, impl=None):
        if impl is None:
            raise RuntimeError(
                f"impl cannot be None for node type {onnx_node.op_type!r} "
                f"from domain {onnx_node.domain!r}.")
        OpRun.__init__(self, onnx_node, log_function)
        self.impl_ = impl

    def _run(self, *inputs):
        if len(self.impl_.input_names) != len(inputs):
            raise RuntimeError(
                f"Mismatch lengths between the number of inputs {len(inputs)} "
                f"and the expected number of inputs {len(self.impl_.inputs)} "
                f"for node {self.op_type!r} from domain {self.domain!r}.")
        feeds = {name: value for name, value in zip(self.impl_.input_names, inputs)}
        results = self.impl_.run(None, feeds)
        if len(self.impl_.output_names) != len(results):
            raise RuntimeError(
                f"Mismatch lengths between the number of outputs {len(results)} "
                f"and the expected number of outputs {len(self.impl_.output_names)} "
                f"for node {self.op_type!r} from domain {self.domain!r}.")
        return tuple(results)


class OpRunUnary(OpRun):
    """
    Ancestor to all unary operators in this subfolder.
    Checks that inputs type are the same.
    """

    def __init__(self, onnx_node, logging_function):
        OpRun.__init__(self, onnx_node, logging_function)

    def run(self, x, attributes=None):
        """
        Calls method ``_run``, catches exceptions,
        displays a longer error message.
        Supports only unary operators.
        """
        try:
            res = self._run(x, attributes=attributes)
        except TypeError as e:
            raise TypeError(
                "Issues with types {} (binary operator {}).".format(
                    ", ".join(str(type(_)) for _ in [x]),
                    self.__class__.__name__)) from e
        return res


class OpRunArg(OpRunUnary):
    """
    Ancestor to all unary operators in this subfolder
    and which produces position of extremas (ArgMax, ...).
    Checks that inputs type are the same.
    The class must have attributes *axis*, *keepdim*.
    """

    def __init__(self, onnx_node, logging_function):
        OpRunUnary.__init__(self, onnx_node, logging_function)
        if not hasattr(self, 'keepdims'):
            raise AttributeError(
                "Attribute 'keepdims' is missing.")
        if not hasattr(self, 'axis'):
            raise AttributeError(
                "Attribute 'axis' is missing.")

    def run(self, x, attributes=None):
        """
        Calls method ``OpRunUnary.run``, catches exceptions,
        displays a longer error message.
        """
        res = OpRunUnary.run(self, x, attributes=attributes)
        if res[0].dtype != np.int64:
            raise RuntimeTypeError(
                "Output type mismatch: should be '{}' != output '{}' "
                "(operator '{}')".format(
                    np.int64, res[0].dtype, self.__class__.__name__))
        return res

    def _run_no_checks_(self, x, attributes=None):
        return OpRunUnary.run(self, x, attributes=attributes)


class OpRunUnaryNum(OpRunUnary):
    """
    Ancestor to all unary and numerical operators
    in this subfolder. Checks that inputs type
    are the same.
    """

    def __init__(self, onnx_node, logging_function):
        OpRunUnary.__init__(self, onnx_node, logging_function)

    def run(self, x, attributes=None):
        """
        Calls method ``OpRunUnary.run``, catches exceptions,
        displays a longer error message.
        Checks that the result is not empty.
        """
        res = OpRunUnary.run(self, x, attributes=attributes)
        if len(res) == 0 or res[0] is None:
            return res
        if not isinstance(res[0], list) and res[0].dtype != x.dtype:
            raise RuntimeTypeError(
                "Output type mismatch: input '{}' != output '{}' "
                "(operator '{}')".format(
                    x.dtype, res[0].dtype, self.__class__.__name__))
        return res

    def _run_no_checks_(self, x, attributes=None):
        return OpRunUnary.run(self, x, attributes=attributes)


class OpRunBinary(OpRun):
    """
    Ancestor to all binary operators in this subfolder.
    Checks that inputs type are the same.
    """

    def __init__(self, onnx_node, logging_function):
        OpRun.__init__(self, onnx_node, logging_function)

    def run(self, x, y, attributes=None):
        """
        Calls method ``_run``, catches exceptions,
        displays a longer error message.
        Supports only binary operators.
        """
        if x is None or y is None:
            raise RuntimeError(
                f"x and y have different dtype: {type(x)} != {type(y)} ({type(self)})")
        if x.dtype != y.dtype:
            raise RuntimeTypeError(
                "Input type mismatch: {} != {} (operator '{}', shapes {}, {})".format(
                    x.dtype, y.dtype, self.__class__.__name__,
                    x.shape, y.shape))
        try:
            res = self._run(x, y, attributes=attributes)
        except (TypeError, ValueError) as e:
            raise TypeError(
                "Issues with types {} (binary operator {}).".format(
                    ", ".join(str(type(_)) for _ in [x, y]),
                    self.__class__.__name__)) from e
        return res

    def _run_no_checks_(self, x, y, attributes=None):
        """
        Calls method ``_run``.
        """
        try:
            res = self._run(x, y, attributes=attributes)
        except TypeError as e:
            raise TypeError(
                "Issues with types {} (binary operator {}).".format(
                    ", ".join(str(type(_)) for _ in [x, y]),
                    self.__class__.__name__)) from e
        return res


class OpRunBinaryComparison(OpRunBinary):
    """
    Ancestor to all binary operators in this subfolder
    comparing tensors.
    """

    def __init__(self, onnx_node, logging_function):
        OpRunBinary.__init__(self, onnx_node, logging_function)


class OpRunBinaryNum(OpRunBinary):
    """
    Ancestor to all binary operators in this subfolder.
    Checks that inputs type are the same.
    """

    def __init__(self, onnx_node, logging_function):
        OpRunBinary.__init__(self, onnx_node, logging_function)

    def run(self, x, y, attributes=None):
        """
        Calls method ``OpRunBinary.run``, catches exceptions,
        displays a longer error message.
        """
        res = OpRunBinary.run(self, x, y, attributes=attributes)
        if res[0].dtype != x.dtype:
            raise RuntimeTypeError(
                "Output type mismatch: {} != {} or {} (operator '{}')"
                " type(x)={} type(y)={}".format(
                    x.dtype, res[0].dtype, y.dtype,
                    self.__class__.__name__, type(x), type(y)))
        return res

    def _run_no_checks_(self, x, y, attributes=None):
        return OpRunBinary._run_no_checks_(self, x, y, attributes=attributes)


class OpRunBinaryNumpy(OpRunBinaryNum):
    """
    *numpy_fct* is a binary numpy function which
    takes two matrices.
    """

    def __init__(self, numpy_fct, onnx_node, logging_function):
        OpRunBinaryNum.__init__(self, onnx_node, logging_function)
        self.numpy_fct = numpy_fct

    def _run(self, a, b, attributes=None):
        return (self.numpy_fct(a, b), )


class OpRunReduceNumpy(OpRunUnaryNum):
    """
    Implements the reduce logic.
    It must have a parameter *axes*.
    """

    def __init__(self, onnx_node, logging_function):
        OpRunUnaryNum.__init__(self, onnx_node, logging_function)
        if isinstance(self.axes, np.ndarray):
            if len(self.axes.shape) == 0 or self.axes.shape[0] == 0:
                self.axes = None
            else:
                self.axes = tuple(self.axes)
        elif self.axes in [[], tuple()]:
            self.axes = None
        elif isinstance(self.axes, list):
            self.axes = tuple(self.axes)
