from typing import Any

from .message import Message

class Error(Exception): ...
class TypeTransformationError(Error): ...

class DescriptorMetaclass(type):
    def __instancecheck__(cls, obj): ...

class DescriptorBase:
    __metaclass__ = DescriptorMetaclass
    has_options = ...  # type: Any
    def __init__(self, options, options_class_name) -> None: ...
    def GetOptions(self): ...

class _NestedDescriptorBase(DescriptorBase):
    name = ...  # type: Any
    full_name = ...  # type: Any
    file = ...  # type: Any
    containing_type = ...  # type: Any
    def __init__(self, options, options_class_name, name, full_name, file, containing_type, serialized_start=..., serialized_end=...) -> None: ...
    def GetTopLevelContainingType(self): ...
    def CopyToProto(self, proto): ...

class Descriptor(_NestedDescriptorBase):
    def __new__(cls, name, full_name, filename, containing_type, fields, nested_types, enum_types, extensions, options=..., is_extendable=..., extension_ranges=..., oneofs=..., file=..., serialized_start=..., serialized_end=..., syntax=...): ...
    fields = ...  # type: Any
    fields_by_number = ...  # type: Any
    fields_by_name = ...  # type: Any
    nested_types = ...  # type: Any
    nested_types_by_name = ...  # type: Any
    enum_types = ...  # type: Any
    enum_types_by_name = ...  # type: Any
    enum_values_by_name = ...  # type: Any
    extensions = ...  # type: Any
    extensions_by_name = ...  # type: Any
    is_extendable = ...  # type: Any
    extension_ranges = ...  # type: Any
    oneofs = ...  # type: Any
    oneofs_by_name = ...  # type: Any
    syntax = ...  # type: Any
    def __init__(self, name, full_name, filename, containing_type, fields, nested_types, enum_types, extensions, options=..., is_extendable=..., extension_ranges=..., oneofs=..., file=..., serialized_start=..., serialized_end=..., syntax=...) -> None: ...
    def EnumValueName(self, enum, value): ...
    def CopyToProto(self, proto): ...

class FieldDescriptor(DescriptorBase):
    TYPE_DOUBLE = ...  # type: Any
    TYPE_FLOAT = ...  # type: Any
    TYPE_INT64 = ...  # type: Any
    TYPE_UINT64 = ...  # type: Any
    TYPE_INT32 = ...  # type: Any
    TYPE_FIXED64 = ...  # type: Any
    TYPE_FIXED32 = ...  # type: Any
    TYPE_BOOL = ...  # type: Any
    TYPE_STRING = ...  # type: Any
    TYPE_GROUP = ...  # type: Any
    TYPE_MESSAGE = ...  # type: Any
    TYPE_BYTES = ...  # type: Any
    TYPE_UINT32 = ...  # type: Any
    TYPE_ENUM = ...  # type: Any
    TYPE_SFIXED32 = ...  # type: Any
    TYPE_SFIXED64 = ...  # type: Any
    TYPE_SINT32 = ...  # type: Any
    TYPE_SINT64 = ...  # type: Any
    MAX_TYPE = ...  # type: Any
    CPPTYPE_INT32 = ...  # type: Any
    CPPTYPE_INT64 = ...  # type: Any
    CPPTYPE_UINT32 = ...  # type: Any
    CPPTYPE_UINT64 = ...  # type: Any
    CPPTYPE_DOUBLE = ...  # type: Any
    CPPTYPE_FLOAT = ...  # type: Any
    CPPTYPE_BOOL = ...  # type: Any
    CPPTYPE_ENUM = ...  # type: Any
    CPPTYPE_STRING = ...  # type: Any
    CPPTYPE_MESSAGE = ...  # type: Any
    MAX_CPPTYPE = ...  # type: Any
    LABEL_OPTIONAL = ...  # type: Any
    LABEL_REQUIRED = ...  # type: Any
    LABEL_REPEATED = ...  # type: Any
    MAX_LABEL = ...  # type: Any
    MAX_FIELD_NUMBER = ...  # type: Any
    FIRST_RESERVED_FIELD_NUMBER = ...  # type: Any
    LAST_RESERVED_FIELD_NUMBER = ...  # type: Any
    def __new__(cls, name, full_name, index, number, type, cpp_type, label, default_value, message_type, enum_type, containing_type, is_extension, extension_scope, options=..., file=..., has_default_value=..., containing_oneof=...): ...
    name = ...  # type: Any
    full_name = ...  # type: Any
    index = ...  # type: Any
    number = ...  # type: Any
    type = ...  # type: Any
    cpp_type = ...  # type: Any
    label = ...  # type: Any
    has_default_value = ...  # type: Any
    default_value = ...  # type: Any
    containing_type = ...  # type: Any
    message_type = ...  # type: Any
    enum_type = ...  # type: Any
    is_extension = ...  # type: Any
    extension_scope = ...  # type: Any
    containing_oneof = ...  # type: Any
    def __init__(self, name, full_name, index, number, type, cpp_type, label, default_value, message_type, enum_type, containing_type, is_extension, extension_scope, options=..., file=..., has_default_value=..., containing_oneof=...) -> None: ...
    @staticmethod
    def ProtoTypeToCppProtoType(proto_type): ...

class EnumDescriptor(_NestedDescriptorBase):
    def __new__(cls, name, full_name, filename, values, containing_type=..., options=..., file=..., serialized_start=..., serialized_end=...): ...
    values = ...  # type: Any
    values_by_name = ...  # type: Any
    values_by_number = ...  # type: Any
    def __init__(self, name, full_name, filename, values, containing_type=..., options=..., file=..., serialized_start=..., serialized_end=...) -> None: ...
    def CopyToProto(self, proto): ...

class EnumValueDescriptor(DescriptorBase):
    def __new__(cls, name, index, number, type=..., options=...): ...
    name = ...  # type: Any
    index = ...  # type: Any
    number = ...  # type: Any
    type = ...  # type: Any
    def __init__(self, name, index, number, type=..., options=...) -> None: ...

class OneofDescriptor:
    def __new__(cls, name, full_name, index, containing_type, fields): ...
    name = ...  # type: Any
    full_name = ...  # type: Any
    index = ...  # type: Any
    containing_type = ...  # type: Any
    fields = ...  # type: Any
    def __init__(self, name, full_name, index, containing_type, fields) -> None: ...

class ServiceDescriptor(_NestedDescriptorBase):
    index = ...  # type: Any
    methods = ...  # type: Any
    def __init__(self, name, full_name, index, methods, options=..., file=..., serialized_start=..., serialized_end=...) -> None: ...
    def FindMethodByName(self, name): ...
    def CopyToProto(self, proto): ...

class MethodDescriptor(DescriptorBase):
    name = ...  # type: Any
    full_name = ...  # type: Any
    index = ...  # type: Any
    containing_service = ...  # type: Any
    input_type = ...  # type: Any
    output_type = ...  # type: Any
    def __init__(self, name, full_name, index, containing_service, input_type, output_type, options=...) -> None: ...

class FileDescriptor(DescriptorBase):
    def __new__(cls, name, package, options=..., serialized_pb=..., dependencies=..., syntax=...): ...
    _options = ...  # type: Any
    message_types_by_name = ...  # type: Any
    name = ...  # type: Any
    package = ...  # type: Any
    syntax = ...  # type: Any
    serialized_pb = ...  # type: Any
    enum_types_by_name = ...  # type: Any
    extensions_by_name = ...  # type: Any
    dependencies = ...  # type: Any
    def __init__(self, name, package, options=..., serialized_pb=..., dependencies=..., syntax=...) -> None: ...
    def CopyToProto(self, proto): ...

def MakeDescriptor(desc_proto, package=..., build_file_if_cpp=..., syntax=...): ...
def _ParseOptions(message: Message, string: str) -> Message: ...
