from google.protobuf.descriptor_pb2 import (
    FileDescriptorProto,
)
from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer,
)
from google.protobuf.message import (
    Message,
)
from typing import (
    Iterable,
    Optional,
    Text,
)


class Version(Message):
    major = ...  # type: int
    minor = ...  # type: int
    patch = ...  # type: int
    suffix = ...  # type: Text

    def __init__(self,
                 major: Optional[int] = ...,
                 minor: Optional[int] = ...,
                 patch: Optional[int] = ...,
                 suffix: Optional[Text] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> Version: ...


class CodeGeneratorRequest(Message):
    file_to_generate = ...  # type: RepeatedScalarFieldContainer[Text]
    parameter = ...  # type: Text

    @property
    def proto_file(self) -> RepeatedCompositeFieldContainer[FileDescriptorProto]: ...

    @property
    def compiler_version(self) -> Version: ...

    def __init__(self,
                 file_to_generate: Optional[Iterable[Text]] = ...,
                 parameter: Optional[Text] = ...,
                 proto_file: Optional[Iterable[FileDescriptorProto]] = ...,
                 compiler_version: Optional[Version] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> CodeGeneratorRequest: ...


class CodeGeneratorResponse(Message):

    class File(Message):
        name = ...  # type: Text
        insertion_point = ...  # type: Text
        content = ...  # type: Text

        def __init__(self,
                     name: Optional[Text] = ...,
                     insertion_point: Optional[Text] = ...,
                     content: Optional[Text] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> CodeGeneratorResponse.File: ...
    error = ...  # type: Text

    @property
    def file(self) -> RepeatedCompositeFieldContainer[CodeGeneratorResponse.File]: ...

    def __init__(self,
                 error: Optional[Text] = ...,
                 file: Optional[Iterable[CodeGeneratorResponse.File]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> CodeGeneratorResponse: ...
