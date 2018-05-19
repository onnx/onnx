from typing import Any

def ReadTag(buffer, pos): ...
def EnumDecoder(field_number, is_repeated, is_packed, key, new_default): ...

Int32Decoder = ...  # type: Any
Int64Decoder = ...  # type: Any
UInt32Decoder = ...  # type: Any
UInt64Decoder = ...  # type: Any
SInt32Decoder = ...  # type: Any
SInt64Decoder = ...  # type: Any
Fixed32Decoder = ...  # type: Any
Fixed64Decoder = ...  # type: Any
SFixed32Decoder = ...  # type: Any
SFixed64Decoder = ...  # type: Any
FloatDecoder = ...  # type: Any
DoubleDecoder = ...  # type: Any
BoolDecoder = ...  # type: Any

def StringDecoder(field_number, is_repeated, is_packed, key, new_default): ...
def BytesDecoder(field_number, is_repeated, is_packed, key, new_default): ...
def GroupDecoder(field_number, is_repeated, is_packed, key, new_default): ...
def MessageDecoder(field_number, is_repeated, is_packed, key, new_default): ...

MESSAGE_SET_ITEM_TAG = ...  # type: Any

def MessageSetItemDecoder(extensions_by_number): ...
def MapDecoder(field_descriptor, new_default, is_message_map): ...

SkipField = ...  # type: Any
