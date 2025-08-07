import os
from pathlib import Path
from typing import Optional
from . import ModelProto, ParseOptions, SerializeOptions


def save(
    proto: ModelProto,
    f: str | Path,
    format: str = "protobuf",
    *,
    save_as_external_data: bool = False,
    all_tensors_to_one_file: bool = True,
    location: str | None = None,
    size_threshold: int = 1024,
    convert_attribute: bool = False,
) -> None:
    """
    Saves the ModelProto to the specified path and optionally,
    serializes tensors with raw data as external data before saving.

    :param proto: should be a in-memory ModelProto
    :param f: can be a file-like object (has "write" function) or a string containing
        a file name or a pathlike object
    :param format: The serialization format. When it is not specified, it is inferred
        from the file extension when ``f`` is a path. If not specified _and_
        ``f`` is not a path, 'protobuf' is used. The encoding is assumed to
        be "utf-8" when the format is a text format.
    :param save_as_external_data: If true, save tensors to external file(s).
        all_tensors_to_one_file: Effective only if save_as_external_data is True.
        If true, save all tensors to one external file specified by location.
        If false, save each tensor to a file named with the tensor name.
    :param all_tensors_to_one_file: if `save_as_external_data` is True,
        then saves all tensors into one file instead of a file per tensor
    :param location: Effective only if `save_as_external_data` is true.
        Specify the external file that all tensors to save to.
        Path is relative to the model path.
        If not specified, will use the model name.
    :param size_threshold: Effective only if save_as_external_data is True.
        Threshold for size of data. Only when tensor's data
        is >= the size_threshold it will be converted
        to external data. To convert every tensor with raw data
        to external data set size_threshold=0.
    :param convert_attribute: Effective only if save_as_external_data is True.
        If true, convert all tensors to external data
        If false, convert only non-attribute tensors to external data
    """
    assert isinstance(proto, ModelProto), f"Unexpected type {type(proto)} for proto."
    assert isinstance(f, (str, Path)), f"Unexpected type {type(f)} for f."
    assert format == "protobuf", f"Unsupported format={format!r}"
    assert (
        all_tensors_to_one_file
    ), f"all_tensors_to_one_file={all_tensors_to_one_file} is not implemented"
    if save_as_external_data or location:
        opts = SerializeOptions()
        opts.raw_data_threshold = size_threshold
        proto.SerializeToFile(str(f), opts, str(location))
    else:
        proto.SerializeToFile(str(f))


def load(
    f: str | Path,
    skip_raw_data: bool = False,
    raw_data_threshold: int = 1024,
    load_external_data: Optional[bool] = None,
    parallel: bool = False,
    num_threads: int = -1,
    location: str = "",
) -> ModelProto:
    """
    Loads a serialized ModelProto into memory.

    :param f: path or bytes
    :param skip_raw_data: skips the raw data of every tensor, this can be used
        to load only the architecture of the model even if the model is stored in
        one unique file
    :param raw_data_threshold: if `skip_raw_data` is True, still keeps the tensors
        smaller than this size (in bytes)
    :param load_external_data: Whether to load the external data.
            Set to True if the data is under the same directory of the model.
    :param parallel: parallelize the loading of the tensors
    :param num_threads: number of threads to use, -1 means the number of cores
    :param location: location of the external weights
        (can be different from the value stored in the main model),
        it must be specified if `load_external_data` is True
    :return: Loaded in-memory ModelProto.
    """
    assert isinstance(f, (str, bytes, Path)), f"Unexpected type {type(f)} for f."
    if load_external_data is None:
        load_external_data = bool(location)
    assert (
        not load_external_data or location
    ), f"'external_data_file' must be specified if load_external_data={location}"
    assert (
        not location or load_external_data
    ), f"'load_external_data' must be True if location={location!r}"
    if isinstance(f, Path):
        f = str(f)
    assert not isinstance(f, str) or os.path.splitext(f)[-1] in {
        ".onnx"
    }, f"File name must have the extension .onnx to be loaded but f={f!r}"
    model = ModelProto()
    if skip_raw_data or parallel:
        opts = ParseOptions()
        opts.skip_raw_data = skip_raw_data
        opts.raw_data_threshold = raw_data_threshold
        opts.parallel = parallel
        opts.num_threads = num_threads
        if isinstance(f, bytes):
            model.ParseFromString(f, opts)
        elif location:
            model.ParseFromFile(f, opts, external_data_file=location)
        else:
            model.ParseFromFile(f, opts)
    else:
        if isinstance(f, bytes):
            model.ParseFromString(f)
        elif location:
            model.ParseFromFile(f, external_data_file=location)
        else:
            model.ParseFromFile(f)
    return model
