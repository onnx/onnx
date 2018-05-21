from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer,
)
from google.protobuf.message import (
    Message,
)
from typing import (
    Iterable,
    List,
    Optional,
    Text,
    Tuple,
    cast,
)


class FileDescriptorSet(Message):

    @property
    def file(self) -> RepeatedCompositeFieldContainer[FileDescriptorProto]: ...

    def __init__(self,
                 file: Optional[Iterable[FileDescriptorProto]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> FileDescriptorSet: ...


class FileDescriptorProto(Message):
    name = ...  # type: Text
    package = ...  # type: Text
    dependency = ...  # type: RepeatedScalarFieldContainer[Text]
    public_dependency = ...  # type: RepeatedScalarFieldContainer[int]
    weak_dependency = ...  # type: RepeatedScalarFieldContainer[int]
    syntax = ...  # type: Text

    @property
    def message_type(
        self) -> RepeatedCompositeFieldContainer[DescriptorProto]: ...

    @property
    def enum_type(
        self) -> RepeatedCompositeFieldContainer[EnumDescriptorProto]: ...

    @property
    def service(
        self) -> RepeatedCompositeFieldContainer[ServiceDescriptorProto]: ...

    @property
    def extension(
        self) -> RepeatedCompositeFieldContainer[FieldDescriptorProto]: ...

    @property
    def options(self) -> FileOptions: ...

    @property
    def source_code_info(self) -> SourceCodeInfo: ...

    def __init__(self,
                 name: Optional[Text] = ...,
                 package: Optional[Text] = ...,
                 dependency: Optional[Iterable[Text]] = ...,
                 public_dependency: Optional[Iterable[int]] = ...,
                 weak_dependency: Optional[Iterable[int]] = ...,
                 message_type: Optional[Iterable[DescriptorProto]] = ...,
                 enum_type: Optional[Iterable[EnumDescriptorProto]] = ...,
                 service: Optional[Iterable[ServiceDescriptorProto]] = ...,
                 extension: Optional[Iterable[FieldDescriptorProto]] = ...,
                 options: Optional[FileOptions] = ...,
                 source_code_info: Optional[SourceCodeInfo] = ...,
                 syntax: Optional[Text] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> FileDescriptorProto: ...


class DescriptorProto(Message):

    class ExtensionRange(Message):
        start = ...  # type: int
        end = ...  # type: int

        @property
        def options(self) -> ExtensionRangeOptions: ...

        def __init__(self,
                     start: Optional[int] = ...,
                     end: Optional[int] = ...,
                     options: Optional[ExtensionRangeOptions] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> DescriptorProto.ExtensionRange: ...

    class ReservedRange(Message):
        start = ...  # type: int
        end = ...  # type: int

        def __init__(self,
                     start: Optional[int] = ...,
                     end: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> DescriptorProto.ReservedRange: ...
    name = ...  # type: Text
    reserved_name = ...  # type: RepeatedScalarFieldContainer[Text]

    @property
    def field(
        self) -> RepeatedCompositeFieldContainer[FieldDescriptorProto]: ...

    @property
    def extension(
        self) -> RepeatedCompositeFieldContainer[FieldDescriptorProto]: ...

    @property
    def nested_type(
        self) -> RepeatedCompositeFieldContainer[DescriptorProto]: ...

    @property
    def enum_type(
        self) -> RepeatedCompositeFieldContainer[EnumDescriptorProto]: ...

    @property
    def extension_range(
        self) -> RepeatedCompositeFieldContainer[DescriptorProto.ExtensionRange]: ...

    @property
    def oneof_decl(
        self) -> RepeatedCompositeFieldContainer[OneofDescriptorProto]: ...

    @property
    def options(self) -> MessageOptions: ...

    @property
    def reserved_range(
        self) -> RepeatedCompositeFieldContainer[DescriptorProto.ReservedRange]: ...

    def __init__(self,
                 name: Optional[Text] = ...,
                 field: Optional[Iterable[FieldDescriptorProto]] = ...,
                 extension: Optional[Iterable[FieldDescriptorProto]] = ...,
                 nested_type: Optional[Iterable[DescriptorProto]] = ...,
                 enum_type: Optional[Iterable[EnumDescriptorProto]] = ...,
                 extension_range: Optional[Iterable[DescriptorProto.ExtensionRange]] = ...,
                 oneof_decl: Optional[Iterable[OneofDescriptorProto]] = ...,
                 options: Optional[MessageOptions] = ...,
                 reserved_range: Optional[Iterable[DescriptorProto.ReservedRange]] = ...,
                 reserved_name: Optional[Iterable[Text]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> DescriptorProto: ...


class ExtensionRangeOptions(Message):

    @property
    def uninterpreted_option(
        self) -> RepeatedCompositeFieldContainer[UninterpretedOption]: ...

    def __init__(self,
                 uninterpreted_option: Optional[Iterable[UninterpretedOption]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> ExtensionRangeOptions: ...


class FieldDescriptorProto(Message):

    class Type(int):

        @classmethod
        def Name(cls, number: int) -> str: ...

        @classmethod
        def Value(cls, name: str) -> FieldDescriptorProto.Type: ...

        @classmethod
        def keys(cls) -> List[str]: ...

        @classmethod
        def values(cls) -> List[FieldDescriptorProto.Type]: ...

        @classmethod
        def items(cls) -> List[Tuple[str, FieldDescriptorProto.Type]]: ...
    TYPE_DOUBLE: Type
    TYPE_FLOAT: Type
    TYPE_INT64: Type
    TYPE_UINT64: Type
    TYPE_INT32: Type
    TYPE_FIXED64: Type
    TYPE_FIXED32: Type
    TYPE_BOOL: Type
    TYPE_STRING: Type
    TYPE_GROUP: Type
    TYPE_MESSAGE: Type
    TYPE_BYTES: Type
    TYPE_UINT32: Type
    TYPE_ENUM: Type
    TYPE_SFIXED32: Type
    TYPE_SFIXED64: Type
    TYPE_SINT32: Type
    TYPE_SINT64: Type

    class Label(int):

        @classmethod
        def Name(cls, number: int) -> str: ...

        @classmethod
        def Value(cls, name: str) -> FieldDescriptorProto.Label: ...

        @classmethod
        def keys(cls) -> List[str]: ...

        @classmethod
        def values(cls) -> List[FieldDescriptorProto.Label]: ...

        @classmethod
        def items(cls) -> List[Tuple[str, FieldDescriptorProto.Label]]: ...
    LABEL_OPTIONAL: Label
    LABEL_REQUIRED: Label
    LABEL_REPEATED: Label
    name = ...  # type: Text
    number = ...  # type: int
    label = ...  # type: FieldDescriptorProto.Label
    type = ...  # type: FieldDescriptorProto.Type
    type_name = ...  # type: Text
    extendee = ...  # type: Text
    default_value = ...  # type: Text
    oneof_index = ...  # type: int
    json_name = ...  # type: Text

    @property
    def options(self) -> FieldOptions: ...

    def __init__(self,
                 name: Optional[Text] = ...,
                 number: Optional[int] = ...,
                 label: Optional[FieldDescriptorProto.Label] = ...,
                 type: Optional[FieldDescriptorProto.Type] = ...,
                 type_name: Optional[Text] = ...,
                 extendee: Optional[Text] = ...,
                 default_value: Optional[Text] = ...,
                 oneof_index: Optional[int] = ...,
                 json_name: Optional[Text] = ...,
                 options: Optional[FieldOptions] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> FieldDescriptorProto: ...


class OneofDescriptorProto(Message):
    name = ...  # type: Text

    @property
    def options(self) -> OneofOptions: ...

    def __init__(self,
                 name: Optional[Text] = ...,
                 options: Optional[OneofOptions] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> OneofDescriptorProto: ...


class EnumDescriptorProto(Message):

    class EnumReservedRange(Message):
        start = ...  # type: int
        end = ...  # type: int

        def __init__(self,
                     start: Optional[int] = ...,
                     end: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> EnumDescriptorProto.EnumReservedRange: ...
    name = ...  # type: Text
    reserved_name = ...  # type: RepeatedScalarFieldContainer[Text]

    @property
    def value(
        self) -> RepeatedCompositeFieldContainer[EnumValueDescriptorProto]: ...

    @property
    def options(self) -> EnumOptions: ...

    @property
    def reserved_range(
        self) -> RepeatedCompositeFieldContainer[EnumDescriptorProto.EnumReservedRange]: ...

    def __init__(self,
                 name: Optional[Text] = ...,
                 value: Optional[Iterable[EnumValueDescriptorProto]] = ...,
                 options: Optional[EnumOptions] = ...,
                 reserved_range: Optional[Iterable[EnumDescriptorProto.EnumReservedRange]] = ...,
                 reserved_name: Optional[Iterable[Text]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> EnumDescriptorProto: ...


class EnumValueDescriptorProto(Message):
    name = ...  # type: Text
    number = ...  # type: int

    @property
    def options(self) -> EnumValueOptions: ...

    def __init__(self,
                 name: Optional[Text] = ...,
                 number: Optional[int] = ...,
                 options: Optional[EnumValueOptions] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> EnumValueDescriptorProto: ...


class ServiceDescriptorProto(Message):
    name = ...  # type: Text

    @property
    def method(
        self) -> RepeatedCompositeFieldContainer[MethodDescriptorProto]: ...

    @property
    def options(self) -> ServiceOptions: ...

    def __init__(self,
                 name: Optional[Text] = ...,
                 method: Optional[Iterable[MethodDescriptorProto]] = ...,
                 options: Optional[ServiceOptions] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> ServiceDescriptorProto: ...


class MethodDescriptorProto(Message):
    name = ...  # type: Text
    input_type = ...  # type: Text
    output_type = ...  # type: Text
    client_streaming = ...  # type: bool
    server_streaming = ...  # type: bool

    @property
    def options(self) -> MethodOptions: ...

    def __init__(self,
                 name: Optional[Text] = ...,
                 input_type: Optional[Text] = ...,
                 output_type: Optional[Text] = ...,
                 options: Optional[MethodOptions] = ...,
                 client_streaming: Optional[bool] = ...,
                 server_streaming: Optional[bool] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> MethodDescriptorProto: ...


class FileOptions(Message):

    class OptimizeMode(int):

        @classmethod
        def Name(cls, number: int) -> str: ...

        @classmethod
        def Value(cls, name: str) -> FileOptions.OptimizeMode: ...

        @classmethod
        def keys(cls) -> List[str]: ...

        @classmethod
        def values(cls) -> List[FileOptions.OptimizeMode]: ...

        @classmethod
        def items(cls) -> List[Tuple[str, FileOptions.OptimizeMode]]: ...
    SPEED: OptimizeMode
    CODE_SIZE: OptimizeMode
    LITE_RUNTIME: OptimizeMode
    java_package = ...  # type: Text
    java_outer_classname = ...  # type: Text
    java_multiple_files = ...  # type: bool
    java_generate_equals_and_hash = ...  # type: bool
    java_string_check_utf8 = ...  # type: bool
    optimize_for = ...  # type: FileOptions.OptimizeMode
    go_package = ...  # type: Text
    cc_generic_services = ...  # type: bool
    java_generic_services = ...  # type: bool
    py_generic_services = ...  # type: bool
    php_generic_services = ...  # type: bool
    deprecated = ...  # type: bool
    cc_enable_arenas = ...  # type: bool
    objc_class_prefix = ...  # type: Text
    csharp_namespace = ...  # type: Text
    swift_prefix = ...  # type: Text
    php_class_prefix = ...  # type: Text
    php_namespace = ...  # type: Text

    @property
    def uninterpreted_option(
        self) -> RepeatedCompositeFieldContainer[UninterpretedOption]: ...

    def __init__(self,
                 java_package: Optional[Text] = ...,
                 java_outer_classname: Optional[Text] = ...,
                 java_multiple_files: Optional[bool] = ...,
                 java_generate_equals_and_hash: Optional[bool] = ...,
                 java_string_check_utf8: Optional[bool] = ...,
                 optimize_for: Optional[FileOptions.OptimizeMode] = ...,
                 go_package: Optional[Text] = ...,
                 cc_generic_services: Optional[bool] = ...,
                 java_generic_services: Optional[bool] = ...,
                 py_generic_services: Optional[bool] = ...,
                 php_generic_services: Optional[bool] = ...,
                 deprecated: Optional[bool] = ...,
                 cc_enable_arenas: Optional[bool] = ...,
                 objc_class_prefix: Optional[Text] = ...,
                 csharp_namespace: Optional[Text] = ...,
                 swift_prefix: Optional[Text] = ...,
                 php_class_prefix: Optional[Text] = ...,
                 php_namespace: Optional[Text] = ...,
                 uninterpreted_option: Optional[Iterable[UninterpretedOption]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> FileOptions: ...


class MessageOptions(Message):
    message_set_wire_format = ...  # type: bool
    no_standard_descriptor_accessor = ...  # type: bool
    deprecated = ...  # type: bool
    map_entry = ...  # type: bool

    @property
    def uninterpreted_option(
        self) -> RepeatedCompositeFieldContainer[UninterpretedOption]: ...

    def __init__(self,
                 message_set_wire_format: Optional[bool] = ...,
                 no_standard_descriptor_accessor: Optional[bool] = ...,
                 deprecated: Optional[bool] = ...,
                 map_entry: Optional[bool] = ...,
                 uninterpreted_option: Optional[Iterable[UninterpretedOption]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> MessageOptions: ...


class FieldOptions(Message):

    class CType(int):

        @classmethod
        def Name(cls, number: int) -> str: ...

        @classmethod
        def Value(cls, name: str) -> FieldOptions.CType: ...

        @classmethod
        def keys(cls) -> List[str]: ...

        @classmethod
        def values(cls) -> List[FieldOptions.CType]: ...

        @classmethod
        def items(cls) -> List[Tuple[str, FieldOptions.CType]]: ...
    STRING: CType
    CORD: CType
    STRING_PIECE: CType

    class JSType(int):

        @classmethod
        def Name(cls, number: int) -> str: ...

        @classmethod
        def Value(cls, name: str) -> FieldOptions.JSType: ...

        @classmethod
        def keys(cls) -> List[str]: ...

        @classmethod
        def values(cls) -> List[FieldOptions.JSType]: ...

        @classmethod
        def items(cls) -> List[Tuple[str, FieldOptions.JSType]]: ...
    JS_NORMAL: JSType
    JS_STRING: JSType
    JS_NUMBER: JSType
    ctype = ...  # type: FieldOptions.CType
    packed = ...  # type: bool
    jstype = ...  # type: FieldOptions.JSType
    lazy = ...  # type: bool
    deprecated = ...  # type: bool
    weak = ...  # type: bool

    @property
    def uninterpreted_option(
        self) -> RepeatedCompositeFieldContainer[UninterpretedOption]: ...

    def __init__(self,
                 ctype: Optional[FieldOptions.CType] = ...,
                 packed: Optional[bool] = ...,
                 jstype: Optional[FieldOptions.JSType] = ...,
                 lazy: Optional[bool] = ...,
                 deprecated: Optional[bool] = ...,
                 weak: Optional[bool] = ...,
                 uninterpreted_option: Optional[Iterable[UninterpretedOption]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> FieldOptions: ...


class OneofOptions(Message):

    @property
    def uninterpreted_option(
        self) -> RepeatedCompositeFieldContainer[UninterpretedOption]: ...

    def __init__(self,
                 uninterpreted_option: Optional[Iterable[UninterpretedOption]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> OneofOptions: ...


class EnumOptions(Message):
    allow_alias = ...  # type: bool
    deprecated = ...  # type: bool

    @property
    def uninterpreted_option(
        self) -> RepeatedCompositeFieldContainer[UninterpretedOption]: ...

    def __init__(self,
                 allow_alias: Optional[bool] = ...,
                 deprecated: Optional[bool] = ...,
                 uninterpreted_option: Optional[Iterable[UninterpretedOption]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> EnumOptions: ...


class EnumValueOptions(Message):
    deprecated = ...  # type: bool

    @property
    def uninterpreted_option(
        self) -> RepeatedCompositeFieldContainer[UninterpretedOption]: ...

    def __init__(self,
                 deprecated: Optional[bool] = ...,
                 uninterpreted_option: Optional[Iterable[UninterpretedOption]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> EnumValueOptions: ...


class ServiceOptions(Message):
    deprecated = ...  # type: bool

    @property
    def uninterpreted_option(
        self) -> RepeatedCompositeFieldContainer[UninterpretedOption]: ...

    def __init__(self,
                 deprecated: Optional[bool] = ...,
                 uninterpreted_option: Optional[Iterable[UninterpretedOption]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> ServiceOptions: ...


class MethodOptions(Message):

    class IdempotencyLevel(int):

        @classmethod
        def Name(cls, number: int) -> str: ...

        @classmethod
        def Value(cls, name: str) -> MethodOptions.IdempotencyLevel: ...

        @classmethod
        def keys(cls) -> List[str]: ...

        @classmethod
        def values(cls) -> List[MethodOptions.IdempotencyLevel]: ...

        @classmethod
        def items(cls) -> List[Tuple[str, MethodOptions.IdempotencyLevel]]: ...
    IDEMPOTENCY_UNKNOWN: IdempotencyLevel
    NO_SIDE_EFFECTS: IdempotencyLevel
    IDEMPOTENT: IdempotencyLevel
    deprecated = ...  # type: bool
    idempotency_level = ...  # type: MethodOptions.IdempotencyLevel

    @property
    def uninterpreted_option(
        self) -> RepeatedCompositeFieldContainer[UninterpretedOption]: ...

    def __init__(self,
                 deprecated: Optional[bool] = ...,
                 idempotency_level: Optional[MethodOptions.IdempotencyLevel] = ...,
                 uninterpreted_option: Optional[Iterable[UninterpretedOption]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> MethodOptions: ...


class UninterpretedOption(Message):

    class NamePart(Message):
        name_part = ...  # type: Text
        is_extension = ...  # type: bool

        def __init__(self,
                     name_part: Text,
                     is_extension: bool,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> UninterpretedOption.NamePart: ...
    identifier_value = ...  # type: Text
    positive_int_value = ...  # type: int
    negative_int_value = ...  # type: int
    double_value = ...  # type: float
    string_value = ...  # type: str
    aggregate_value = ...  # type: Text

    @property
    def name(
        self) -> RepeatedCompositeFieldContainer[UninterpretedOption.NamePart]: ...

    def __init__(self,
                 name: Optional[Iterable[UninterpretedOption.NamePart]] = ...,
                 identifier_value: Optional[Text] = ...,
                 positive_int_value: Optional[int] = ...,
                 negative_int_value: Optional[int] = ...,
                 double_value: Optional[float] = ...,
                 string_value: Optional[str] = ...,
                 aggregate_value: Optional[Text] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> UninterpretedOption: ...


class SourceCodeInfo(Message):

    class Location(Message):
        path = ...  # type: RepeatedScalarFieldContainer[int]
        span = ...  # type: RepeatedScalarFieldContainer[int]
        leading_comments = ...  # type: Text
        trailing_comments = ...  # type: Text
        leading_detached_comments = ...  # type: RepeatedScalarFieldContainer[Text]

        def __init__(self,
                     path: Optional[Iterable[int]] = ...,
                     span: Optional[Iterable[int]] = ...,
                     leading_comments: Optional[Text] = ...,
                     trailing_comments: Optional[Text] = ...,
                     leading_detached_comments: Optional[Iterable[Text]] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> SourceCodeInfo.Location: ...

    @property
    def location(
        self) -> RepeatedCompositeFieldContainer[SourceCodeInfo.Location]: ...

    def __init__(self,
                 location: Optional[Iterable[SourceCodeInfo.Location]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> SourceCodeInfo: ...


class GeneratedCodeInfo(Message):

    class Annotation(Message):
        path = ...  # type: RepeatedScalarFieldContainer[int]
        source_file = ...  # type: Text
        begin = ...  # type: int
        end = ...  # type: int

        def __init__(self,
                     path: Optional[Iterable[int]] = ...,
                     source_file: Optional[Text] = ...,
                     begin: Optional[int] = ...,
                     end: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> GeneratedCodeInfo.Annotation: ...

    @property
    def annotation(
        self) -> RepeatedCompositeFieldContainer[GeneratedCodeInfo.Annotation]: ...

    def __init__(self,
                 annotation: Optional[Iterable[GeneratedCodeInfo.Annotation]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> GeneratedCodeInfo: ...
