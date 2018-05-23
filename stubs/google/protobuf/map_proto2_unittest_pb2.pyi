from google.protobuf.message import (
    Message,
)
from google.protobuf.unittest_import_pb2 import (
    ImportEnumForMap,
)
from typing import (
    List,
    Mapping,
    MutableMapping,
    Optional,
    Text,
    Tuple,
    cast,
)


class Proto2MapEnum(int):

    @classmethod
    def Name(cls, number: int) -> str: ...

    @classmethod
    def Value(cls, name: str) -> Proto2MapEnum: ...

    @classmethod
    def keys(cls) -> List[str]: ...

    @classmethod
    def values(cls) -> List[Proto2MapEnum]: ...

    @classmethod
    def items(cls) -> List[Tuple[str, Proto2MapEnum]]: ...
PROTO2_MAP_ENUM_FOO: Proto2MapEnum
PROTO2_MAP_ENUM_BAR: Proto2MapEnum
PROTO2_MAP_ENUM_BAZ: Proto2MapEnum


class Proto2MapEnumPlusExtra(int):

    @classmethod
    def Name(cls, number: int) -> str: ...

    @classmethod
    def Value(cls, name: str) -> Proto2MapEnumPlusExtra: ...

    @classmethod
    def keys(cls) -> List[str]: ...

    @classmethod
    def values(cls) -> List[Proto2MapEnumPlusExtra]: ...

    @classmethod
    def items(cls) -> List[Tuple[str, Proto2MapEnumPlusExtra]]: ...
E_PROTO2_MAP_ENUM_FOO: Proto2MapEnumPlusExtra
E_PROTO2_MAP_ENUM_BAR: Proto2MapEnumPlusExtra
E_PROTO2_MAP_ENUM_BAZ: Proto2MapEnumPlusExtra
E_PROTO2_MAP_ENUM_EXTRA: Proto2MapEnumPlusExtra


class TestEnumMap(Message):

    class KnownMapFieldEntry(Message):
        key = ...  # type: int
        value = ...  # type: Proto2MapEnum

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[Proto2MapEnum] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestEnumMap.KnownMapFieldEntry: ...

    class UnknownMapFieldEntry(Message):
        key = ...  # type: int
        value = ...  # type: Proto2MapEnum

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[Proto2MapEnum] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestEnumMap.UnknownMapFieldEntry: ...

    @property
    def known_map_field(self) -> MutableMapping[int, Proto2MapEnum]: ...

    @property
    def unknown_map_field(self) -> MutableMapping[int, Proto2MapEnum]: ...

    def __init__(self,
                 known_map_field: Optional[Mapping[int, Proto2MapEnum]]=...,
                 unknown_map_field: Optional[Mapping[int, Proto2MapEnum]]=...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> TestEnumMap: ...


class TestEnumMapPlusExtra(Message):

    class KnownMapFieldEntry(Message):
        key = ...  # type: int
        value = ...  # type: Proto2MapEnumPlusExtra

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[Proto2MapEnumPlusExtra] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestEnumMapPlusExtra.KnownMapFieldEntry: ...

    class UnknownMapFieldEntry(Message):
        key = ...  # type: int
        value = ...  # type: Proto2MapEnumPlusExtra

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[Proto2MapEnumPlusExtra] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestEnumMapPlusExtra.UnknownMapFieldEntry: ...

    @property
    def known_map_field(self) -> MutableMapping[int, Proto2MapEnumPlusExtra]: ...

    @property
    def unknown_map_field(self) -> MutableMapping[int, Proto2MapEnumPlusExtra]: ...

    def __init__(self,
                 known_map_field: Optional[Mapping[int, Proto2MapEnumPlusExtra]]=...,
                 unknown_map_field: Optional[Mapping[int, Proto2MapEnumPlusExtra]]=...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> TestEnumMapPlusExtra: ...


class TestImportEnumMap(Message):

    class ImportEnumAmpEntry(Message):
        key = ...  # type: int
        value = ...  # type: ImportEnumForMap

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[ImportEnumForMap] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestImportEnumMap.ImportEnumAmpEntry: ...

    @property
    def import_enum_amp(self) -> MutableMapping[int, ImportEnumForMap]: ...

    def __init__(self,
                 import_enum_amp: Optional[Mapping[int, ImportEnumForMap]]=...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> TestImportEnumMap: ...


class TestIntIntMap(Message):

    class MEntry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestIntIntMap.MEntry: ...

    @property
    def m(self) -> MutableMapping[int, int]: ...

    def __init__(self,
                 m: Optional[Mapping[int, int]]=...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> TestIntIntMap: ...


class TestMaps(Message):

    class MInt32Entry(Message):
        key = ...  # type: int

        @property
        def value(self) -> TestIntIntMap: ...

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[TestIntIntMap] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestMaps.MInt32Entry: ...

    class MInt64Entry(Message):
        key = ...  # type: int

        @property
        def value(self) -> TestIntIntMap: ...

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[TestIntIntMap] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestMaps.MInt64Entry: ...

    class MUint32Entry(Message):
        key = ...  # type: int

        @property
        def value(self) -> TestIntIntMap: ...

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[TestIntIntMap] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestMaps.MUint32Entry: ...

    class MUint64Entry(Message):
        key = ...  # type: int

        @property
        def value(self) -> TestIntIntMap: ...

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[TestIntIntMap] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestMaps.MUint64Entry: ...

    class MSint32Entry(Message):
        key = ...  # type: int

        @property
        def value(self) -> TestIntIntMap: ...

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[TestIntIntMap] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestMaps.MSint32Entry: ...

    class MSint64Entry(Message):
        key = ...  # type: int

        @property
        def value(self) -> TestIntIntMap: ...

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[TestIntIntMap] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestMaps.MSint64Entry: ...

    class MFixed32Entry(Message):
        key = ...  # type: int

        @property
        def value(self) -> TestIntIntMap: ...

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[TestIntIntMap] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestMaps.MFixed32Entry: ...

    class MFixed64Entry(Message):
        key = ...  # type: int

        @property
        def value(self) -> TestIntIntMap: ...

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[TestIntIntMap] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestMaps.MFixed64Entry: ...

    class MSfixed32Entry(Message):
        key = ...  # type: int

        @property
        def value(self) -> TestIntIntMap: ...

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[TestIntIntMap] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestMaps.MSfixed32Entry: ...

    class MSfixed64Entry(Message):
        key = ...  # type: int

        @property
        def value(self) -> TestIntIntMap: ...

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[TestIntIntMap] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestMaps.MSfixed64Entry: ...

    class MBoolEntry(Message):
        key = ...  # type: bool

        @property
        def value(self) -> TestIntIntMap: ...

        def __init__(self,
                     key: Optional[bool] = ...,
                     value: Optional[TestIntIntMap] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestMaps.MBoolEntry: ...

    class MStringEntry(Message):
        key = ...  # type: Text

        @property
        def value(self) -> TestIntIntMap: ...

        def __init__(self,
                     key: Optional[Text] = ...,
                     value: Optional[TestIntIntMap] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestMaps.MStringEntry: ...

    @property
    def m_int32(self) -> MutableMapping[int, TestIntIntMap]: ...

    @property
    def m_int64(self) -> MutableMapping[int, TestIntIntMap]: ...

    @property
    def m_uint32(self) -> MutableMapping[int, TestIntIntMap]: ...

    @property
    def m_uint64(self) -> MutableMapping[int, TestIntIntMap]: ...

    @property
    def m_sint32(self) -> MutableMapping[int, TestIntIntMap]: ...

    @property
    def m_sint64(self) -> MutableMapping[int, TestIntIntMap]: ...

    @property
    def m_fixed32(self) -> MutableMapping[int, TestIntIntMap]: ...

    @property
    def m_fixed64(self) -> MutableMapping[int, TestIntIntMap]: ...

    @property
    def m_sfixed32(self) -> MutableMapping[int, TestIntIntMap]: ...

    @property
    def m_sfixed64(self) -> MutableMapping[int, TestIntIntMap]: ...

    @property
    def m_bool(self) -> MutableMapping[bool, TestIntIntMap]: ...

    @property
    def m_string(self) -> MutableMapping[Text, TestIntIntMap]: ...

    def __init__(self,
                 m_int32: Optional[Mapping[int, TestIntIntMap]]=...,
                 m_int64: Optional[Mapping[int, TestIntIntMap]]=...,
                 m_uint32: Optional[Mapping[int, TestIntIntMap]]=...,
                 m_uint64: Optional[Mapping[int, TestIntIntMap]]=...,
                 m_sint32: Optional[Mapping[int, TestIntIntMap]]=...,
                 m_sint64: Optional[Mapping[int, TestIntIntMap]]=...,
                 m_fixed32: Optional[Mapping[int, TestIntIntMap]]=...,
                 m_fixed64: Optional[Mapping[int, TestIntIntMap]]=...,
                 m_sfixed32: Optional[Mapping[int, TestIntIntMap]]=...,
                 m_sfixed64: Optional[Mapping[int, TestIntIntMap]]=...,
                 m_bool: Optional[Mapping[bool, TestIntIntMap]]=...,
                 m_string: Optional[Mapping[Text, TestIntIntMap]]=...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> TestMaps: ...


class TestSubmessageMaps(Message):

    @property
    def m(self) -> TestMaps: ...

    def __init__(self,
                 m: Optional[TestMaps] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> TestSubmessageMaps: ...
