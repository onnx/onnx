from . import AttributeProto, ModelProto, SparseTensorProto


class ValidationError(ValueError):
    "Invalid proto"


def check_attribute(att: AttributeProto):
    """Checks an AttributeProto is valid."""
    oneof = [
        att.has_f(),
        att.has_i(),
        att.has_s(),
        att.has_t(),
        att.has_sparse_tensor(),
        att.has_floats(),
        att.has_ints(),
        att.has_strings(),
        att.has_tensors(),
        att.has_sparse_tensors(),
    ]
    if not any(oneof):
        raise ValidationError(f"The attribute has no value: {att}")
    total = sum(int(i) for i in oneof)
    if total != 1:
        raise ValidationError(f"The attribute has more than one value: {att}")


def check_sparse_tensor(sp: SparseTensorProto):
    """Checks a SparseTensorProto is valid."""
    shape = tuple(sp.dims)
    if len(shape) != 2:
        raise ValidationError(f"Only 2D sparse tensors are allowed: {shape}")


def check_model(model: ModelProto):
    """Checks a ModelProto is valid."""
    meta = set(m.key for m in model.metadata_props)
    if len(meta) != len(model.metadata_props):
        raise ValidationError(
            f"Duplicated key in metadata_props: {model.metadata_props}"
        )
