from __future__ import annotations

from onnx.ir import _enums, _protocols
from typing import Sequence


# NOTE: None of these classes will have a "to_onnx" method because
# We cannot assume that the build tool chain has protoc installed.


class Attribute(_protocols.AttributeProtocol):
    """Base class for ONNX attributes."""

    # NOTE: We use primitive types for T

    def __init__(
        self,
        name: str,
        typ: _enums.AttributeType,
        value,
        *,
        ref_attr_name: str,
        doc_string: str,
    ):
        self.name = name
        self.type = typ
        self.value = value
        self.ref_attr_name = ref_attr_name
        self.doc_string = doc_string

    # TODO: How do we represent ref attributes? Do we need to?


# NOTE: The following classes are just supporting classes (partially applied) for convenience
# But I think they would be useful to have in the IR by having the type info
# explicitly in the class type.
# Arguably, they can also be functions that return Attribute objects.
# TODO: We need something that is easy for pattern matchers, otherwise they
# can just be make_attribute functions.
class AttrFloat32(Attribute):
    def __init__(
        self, name: str, value: float, ref_attr_name: str = "", doc_string: str = ""
    ):
        super().__init__(
            name,
            _enums.AttributeType.FLOAT,
            value,
            ref_attr_name=ref_attr_name,
            doc_string=doc_string,
        )


class AttrInt64(Attribute):
    def __init__(
        self, name: str, value: int, ref_attr_name: str = "", doc_string: str = ""
    ):
        super().__init__(
            name,
            _enums.AttributeType.INT,
            value,
            ref_attr_name=ref_attr_name,
            doc_string=doc_string,
        )


class AttrString(Attribute):
    def __init__(
        self, name: str, value: str, ref_attr_name: str = "", doc_string: str = ""
    ):
        super().__init__(
            name,
            _enums.AttributeType.STRING,
            value,
            ref_attr_name=ref_attr_name,
            doc_string=doc_string,
        )


# NOTE: Tensor should be a tensor proto
class AttrTensor(Attribute):
    def __init__(
        self,
        name: str,
        value: _protocols.TensorProtocol,
        ref_attr_name: str = "",
        doc_string: str = "",
    ):
        super().__init__(
            name,
            _enums.AttributeType.TENSOR,
            value,
            ref_attr_name=ref_attr_name,
            doc_string=doc_string,
        )


class AttrGraph(Attribute):
    def __init__(
        self,
        name: str,
        value: _protocols.GraphProtocol,
        ref_attr_name: str = "",
        doc_string: str = "",
    ):
        super().__init__(
            name,
            _enums.AttributeType.GRAPH,
            value,
            ref_attr_name=ref_attr_name,
            doc_string=doc_string,
        )


class AttrFloat32s(Attribute):
    def __init__(
        self,
        name: str,
        value: Sequence[float],
        ref_attr_name: str = "",
        doc_string: str = "",
    ):
        super().__init__(
            name,
            _enums.AttributeType.FLOATS,
            value,
            ref_attr_name=ref_attr_name,
            doc_string=doc_string,
        )


class AttrInt64s(Attribute):
    def __init__(
        self,
        name: str,
        value: Sequence[int],
        ref_attr_name: str = "",
        doc_string: str = "",
    ):
        super().__init__(
            name,
            _enums.AttributeType.INTS,
            value,
            ref_attr_name=ref_attr_name,
            doc_string=doc_string,
        )


class AttrStrings(Attribute):
    def __init__(
        self,
        name: str,
        value: Sequence[str],
        ref_attr_name: str = "",
        doc_string: str = "",
    ):
        super().__init__(
            name,
            _enums.AttributeType.STRINGS,
            value,
            ref_attr_name=ref_attr_name,
            doc_string=doc_string,
        )


class AttrTensors(Attribute):
    def __init__(
        self,
        name: str,
        value: Sequence[_protocols.TensorProtocol],
        ref_attr_name: str = "",
        doc_string: str = "",
    ):
        super().__init__(
            name,
            _enums.AttributeType.TENSORS,
            value,
            ref_attr_name=ref_attr_name,
            doc_string=doc_string,
        )


class AttrGraphs(Attribute):
    def __init__(
        self,
        name: str,
        value: Sequence[_protocols.GraphProtocol],
        ref_attr_name: str = "",
        doc_string: str = "",
    ):
        super().__init__(
            name,
            _enums.AttributeType.GRAPHS,
            value,
            ref_attr_name=ref_attr_name,
            doc_string=doc_string,
        )


# NOTE: SparseTensor should be a sparse tensor proto
class AttrSparseTensor(Attribute):
    def __init__(
        self,
        name: str,
        value: Sequence[_protocols.SparseTensorProtocol],
        ref_attr_name: str = "",
        doc_string: str = "",
    ):
        super().__init__(
            name,
            _enums.AttributeType.SPARSE_TENSOR,
            value,
            ref_attr_name=ref_attr_name,
            doc_string=doc_string,
        )


class AttrSparseTensors(Attribute):
    def __init__(
        self,
        name: str,
        value: Sequence[_protocols.SparseTensorProtocol],
        ref_attr_name: str = "",
        doc_string: str = "",
    ):
        super().__init__(
            name,
            _enums.AttributeType.SPARSE_TENSORS,
            value,
            ref_attr_name=ref_attr_name,
            doc_string=doc_string,
        )


class AttrTypeProto(Attribute):
    def __init__(
        self,
        name: str,
        value: _protocols.TypeProtocol,
        ref_attr_name: str = "",
        doc_string: str = "",
    ):
        super().__init__(
            name,
            _enums.AttributeType.TYPE_PROTO,
            value,
            ref_attr_name=ref_attr_name,
            doc_string=doc_string,
        )
