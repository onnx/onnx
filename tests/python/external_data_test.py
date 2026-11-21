# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import uuid
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import onnx
from onnx import (
    ModelProto,
    NodeProto,
    TensorProto,
    checker,
    external_data_helper,
    helper,
    parser,
    shape_inference,
)
from onnx.external_data_helper import (
    _ALLOWED_EXTERNAL_DATA_KEYS,
    _MAX_EXTERNAL_DATA_PADDING,
    ExternalDataInfo,
    convert_model_from_external_data,
    convert_model_to_external_data,
    load_external_data_for_model,
    load_external_data_for_tensor,
    save_external_data,
    set_external_data,
    write_external_data_tensors,
)
from onnx.numpy_helper import from_array, to_array

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class TestLoadExternalDataBase:
    """Base class for testing external data related behaviors.

    Subclasses should be parameterized with a serialization format.
    """

    serialization_format: str = "protobuf"

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path):
        self.temp_dir = str(tmp_path)
        self.initializer_value = np.arange(6).reshape(3, 2).astype(np.float32) + 512
        self.attribute_value = np.arange(6).reshape(2, 3).astype(np.float32) + 256
        self.model_filename = self.create_test_model()

    def get_temp_model_filename(self) -> str:
        return os.path.join(self.temp_dir, str(uuid.uuid4()) + ".onnx")

    def create_external_data_tensor(
        self, value: list[Any], tensor_name: str, location: str = ""
    ) -> TensorProto:
        tensor = from_array(np.array(value))
        tensor.name = tensor_name
        tensor_filename = location or f"{tensor_name}.bin"
        set_external_data(tensor, location=tensor_filename)

        with open(os.path.join(self.temp_dir, tensor_filename), "wb") as data_file:
            data_file.write(tensor.raw_data)
        tensor.ClearField("raw_data")
        tensor.data_location = onnx.TensorProto.EXTERNAL
        return tensor

    def create_test_model(self, location: str = "") -> str:
        constant_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["values"],
            value=self.create_external_data_tensor(
                self.attribute_value,
                "attribute_value",
            ),
        )

        initializers = [
            self.create_external_data_tensor(
                self.initializer_value,
                "input_value",
                location,
            )
        ]
        inputs = [
            helper.make_tensor_value_info(
                "input_value", onnx.TensorProto.FLOAT, self.initializer_value.shape
            )
        ]

        graph = helper.make_graph(
            [constant_node],
            "test_graph",
            inputs=inputs,
            outputs=[],
            initializer=initializers,
        )
        model = helper.make_model(graph)

        model_filename = os.path.join(self.temp_dir, "model.onnx")
        onnx.save_model(model, model_filename, self.serialization_format)

        return model_filename

    def test_check_model(self) -> None:
        if self.serialization_format != "protobuf":
            pytest.skip(
                "check_model supports protobuf only as binary when provided as a path"
            )
        checker.check_model(self.model_filename)


class TestLoadExternalData(TestLoadExternalDataBase):
    @pytest.fixture(scope="class", params=["protobuf", "textproto"], autouse=True)
    def override_serialization_format(self, request):
        # Override the class' `serialization_format`.
        # This is not idiomatic pytest code which would structure all
        # these dependencies as explicit fixtures rather than setting
        # state on `self`. The code is as it is because it was
        # inherited from an earlier unittest/parameterized setup.
        self.serialization_format = request.param

    def test_load_external_data(self) -> None:
        model = onnx.load_model(self.model_filename, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)

        attribute_tensor = model.graph.node[0].attribute[0].t
        np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)

    def test_load_external_data_for_model(self) -> None:
        model = onnx.load_model(
            self.model_filename, self.serialization_format, load_external_data=False
        )
        load_external_data_for_model(model, self.temp_dir)
        initializer_tensor = model.graph.initializer[0]
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)

        attribute_tensor = model.graph.node[0].attribute[0].t
        np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)

    def test_load_external_data_for_model_reuses_directory_authority(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All tensors in one model load must use the same pinned base directory."""
        original_context = external_data_helper._ExternalDataContext
        constructed_contexts = []

        class CountingContext:
            def __init__(self, base_dir: str) -> None:
                self._context = original_context(base_dir)
                constructed_contexts.append(self)

            def open(self, location: str, tensor_name: str, read_only: bool) -> int:
                return self._context.open(location, tensor_name, read_only)

        monkeypatch.setattr(
            external_data_helper, "_ExternalDataContext", CountingContext
        )
        model = onnx.load_model(
            self.model_filename, self.serialization_format, load_external_data=False
        )

        load_external_data_for_model(model, self.temp_dir)

        assert len(constructed_contexts) == 1

    @pytest.mark.parametrize("replace_before_open", [True, False])
    def test_load_model_pins_main_model_and_external_data_to_same_directory(
        self, monkeypatch: pytest.MonkeyPatch, replace_before_open: bool
    ) -> None:
        """Replacing a pinned directory must not redirect the model or sidecars."""
        if os.name == "nt" and not replace_before_open:
            pytest.skip("Windows may deny renaming a directory with an open child file")
        original_context = onnx._ExternalDataContext
        base_path = pathlib.Path(self.temp_dir)
        original_path = base_path.with_name(f"{base_path.name}-original")
        replaced = False

        class ReplacingContext:
            def __init__(self, base_dir: str) -> None:
                self._context = original_context(base_dir)

            def open_model_file(self, location: str, read_only: bool) -> int:
                nonlocal replaced
                fd = None
                try:
                    if not replace_before_open:
                        fd = self._context.open_model_file(location, read_only)
                    base_path.rename(original_path)
                    replaced = True
                    base_path.mkdir()
                    (base_path / "input_value.bin").write_bytes(
                        bytes(self_initializer_nbytes)
                    )
                    (base_path / "attribute_value.bin").write_bytes(
                        bytes(self_attribute_nbytes)
                    )
                    if fd is None:
                        fd = self._context.open_model_file(location, read_only)
                except BaseException:
                    if fd is not None:
                        os.close(fd)
                    raise
                else:
                    return fd

            def open(self, location: str, tensor_name: str, read_only: bool) -> int:
                return self._context.open(location, tensor_name, read_only)

        self_initializer_nbytes = self.initializer_value.nbytes
        self_attribute_nbytes = self.attribute_value.nbytes
        monkeypatch.setattr(onnx, "_ExternalDataContext", ReplacingContext)
        try:
            model = onnx.load_model(self.model_filename, self.serialization_format)
        finally:
            if replaced:
                shutil.rmtree(base_path)
                original_path.rename(base_path)

        np.testing.assert_allclose(
            to_array(model.graph.initializer[0]), self.initializer_value
        )
        np.testing.assert_allclose(
            to_array(model.graph.node[0].attribute[0].t), self.attribute_value
        )

    def test_named_model_file_object_rejects_replaced_parent(self) -> None:
        """A named file handle must not load sidecars from a replacement directory."""
        replacement_path = pathlib.Path(self.temp_dir) / "replacement"
        replacement_path.mkdir()
        replacement_model = replacement_path / "model.onnx"
        shutil.copyfile(self.model_filename, replacement_model)
        (replacement_path / "input_value.bin").write_bytes(
            bytes(self.initializer_value.nbytes)
        )
        (replacement_path / "attribute_value.bin").write_bytes(
            bytes(self.attribute_value.nbytes)
        )

        class StaleNamedFile:
            def __init__(self, model_file) -> None:
                self._model_file = model_file
                self.name = str(replacement_model)

            def read(self):
                return self._model_file.read()

            def fileno(self) -> int:
                return self._model_file.fileno()

        with (
            open(self.model_filename, "rb") as model_file,
            pytest.raises(ValueError, match="does not belong"),
        ):
            onnx.load_model(StaleNamedFile(model_file), self.serialization_format)

    def test_named_model_file_object_loads_external_data(self) -> None:
        """A named file handle and its verified parent still support sidecars."""
        with open(self.model_filename, "rb") as model_file:
            model = onnx.load_model(model_file, self.serialization_format)
        np.testing.assert_allclose(
            to_array(model.graph.initializer[0]), self.initializer_value
        )

    def test_save_external_data(self) -> None:
        model = onnx.load_model(self.model_filename, self.serialization_format)

        temp_dir = os.path.join(self.temp_dir, "save_copy")
        os.mkdir(temp_dir)
        new_model_filename = os.path.join(temp_dir, "model.onnx")
        onnx.save_model(model, new_model_filename, self.serialization_format)

        new_model = onnx.load_model(new_model_filename, self.serialization_format)
        initializer_tensor = new_model.graph.initializer[0]
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)

        attribute_tensor = new_model.graph.node[0].attribute[0].t
        np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)


class TestModelFileAccess:
    @pytest.mark.skipif(os.name != "nt", reason="Windows drive-relative paths")
    @pytest.mark.parametrize(
        "path_kind", ["relative", "drive_relative", "rooted", "absolute"]
    )
    def test_windows_model_path_forms(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path_kind: str
    ) -> None:
        monkeypatch.chdir(tmp_path)
        destination = str(tmp_path / "model.onnx")
        drive, tail = os.path.splitdrive(destination)
        path = {
            "relative": "model.onnx",
            "drive_relative": drive + "model.onnx",
            "rooted": tail,
            "absolute": destination,
        }[path_kind]
        assert onnx._get_file_path(path) == destination
        model = helper.make_model(helper.make_graph([], "test", [], []))
        onnx.save_model(model, path)
        assert onnx.load_model(path) == model

    @pytest.mark.skipif(
        os.name == "nt", reason="POSIX symlink and '..' lookup semantics"
    )
    @pytest.mark.parametrize("relative", [True, False])
    @pytest.mark.parametrize("named_file", [True, False])
    def test_symlink_dotdot_selects_model_and_sidecar_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        relative: bool,
        named_file: bool,
    ) -> None:
        base = tmp_path / "base"
        other = tmp_path / "other"
        base.mkdir()
        (other / "subdir").mkdir(parents=True)
        (base / "link").symlink_to(other / "subdir", target_is_directory=True)
        sentinel = b"do not overwrite the lexically normalized path"
        (base / "model.onnx").write_bytes(sentinel)
        monkeypatch.chdir(tmp_path)
        path = os.path.join("base", "link", "..", "model.onnx")
        if not relative:
            path = os.path.join(str(tmp_path), path)
        values = np.arange(6, dtype=np.float32)
        model = helper.make_model(
            helper.make_graph([], "test", [], [], [from_array(values, name="weight")])
        )
        if named_file:
            with open(path, "wb") as stream:
                onnx.save_model(
                    model,
                    stream,
                    save_as_external_data=True,
                    location="weights.bin",
                    size_threshold=0,
                )
            with open(path, "rb") as stream:
                loaded = onnx.load_model(stream)
        else:
            onnx.save_model(
                model,
                path,
                save_as_external_data=True,
                location="weights.bin",
                size_threshold=0,
            )
            loaded = onnx.load_model(path)
        np.testing.assert_array_equal(to_array(loaded.graph.initializer[0]), values)
        assert (base / "model.onnx").read_bytes() == sentinel
        assert not (base / "weights.bin").exists()
        assert (other / "weights.bin").read_bytes() == values.tobytes()

    @pytest.mark.parametrize("existing", [True, False])
    def test_serialization_failure_preserves_model_destination(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
    ) -> None:
        destination = tmp_path / "model.onnx"
        sentinel = b"existing model must survive"
        if existing:
            destination.write_bytes(sentinel)
        model = helper.make_model(helper.make_graph([], "test", [], []))

        def fail_serialization(*_args, **_kwargs):
            raise RuntimeError("Cannot serialize model")

        monkeypatch.setattr(
            onnx._get_serializer("protobuf"), "serialize_proto", fail_serialization
        )
        with pytest.raises(RuntimeError, match="Cannot serialize model"):
            onnx.save_model(model, destination)
        if existing:
            assert destination.read_bytes() == sentinel
        else:
            assert not destination.exists()

    @pytest.mark.skipif(os.name == "nt", reason="Requires POSIX file permissions")
    @pytest.mark.parametrize("named_file", [True, False])
    def test_save_write_only_model_with_external_data(
        self, tmp_path: Path, named_file: bool
    ) -> None:
        destination = tmp_path / "model.onnx"
        destination.write_bytes(b"stale data" * 1000)
        destination.chmod(0o200)
        try:
            if os.access(destination, os.R_OK):
                pytest.skip("Current user can bypass read permission checks")
            values = np.arange(6, dtype=np.float32)
            model = helper.make_model(
                helper.make_graph(
                    [], "test", [], [], [from_array(values, name="weight")]
                )
            )
            if named_file:
                with open(destination, "wb") as stream:
                    onnx.save_model(
                        model,
                        stream,
                        save_as_external_data=True,
                        location="weights.bin",
                        size_threshold=0,
                    )
            else:
                onnx.save_model(
                    model,
                    destination,
                    save_as_external_data=True,
                    location="weights.bin",
                    size_threshold=0,
                )
        finally:
            destination.chmod(0o600)
        loaded = onnx.load_model(destination)
        np.testing.assert_array_equal(to_array(loaded.graph.initializer[0]), values)

    @pytest.mark.skipif(os.name == "nt", reason="Requires POSIX FIFOs")
    @pytest.mark.parametrize(
        "operation",
        [
            "load_model",
            "load_tensor",
            "load_tensors",
            "load_container",
            "load_main",
            "save_main",
            "check_main",
        ],
    )
    def test_fifo_is_rejected_without_blocking(
        self, tmp_path: Path, operation: str
    ) -> None:
        values = np.arange(6, dtype=np.float32)
        tensor = from_array(values, name="weight")
        set_external_data(tensor, location="weights.bin")
        tensor.ClearField("raw_data")
        model = helper.make_model(helper.make_graph([], "test", [], [], [tensor]))
        (tmp_path / "model.onnx").write_bytes(model.SerializeToString())
        os.mkfifo(tmp_path / "weights.bin")
        os.mkfifo(tmp_path / "main.onnx")
        # Keep a regression bounded even if the native open blocks with the GIL held.
        script = """
import sys
import onnx
from onnx import external_data_helper, model_container

operation = sys.argv[1]
try:
    if operation == "load_model":
        onnx.load_model("model.onnx")
    elif operation == "load_container":
        model_container.ModelContainer().load("model.onnx")
    elif operation == "load_main":
        onnx.load_model("main.onnx")
    elif operation == "save_main":
        onnx.save_model(onnx.ModelProto(), "main.onnx")
    elif operation == "check_main":
        onnx.checker.check_model("main.onnx")
    else:
        model = onnx.load_model("model.onnx", load_external_data=False)
        if operation == "load_tensor":
            external_data_helper.load_external_data_for_tensor(model.graph.initializer[0], ".")
        else:
            external_data_helper.load_external_data_for_model(model, ".")
except onnx.checker.ValidationError:
    pass
else:
    raise AssertionError("FIFO was accepted")
"""
        result = subprocess.run(  # noqa: S603 - fixed script and parametrized test operations
            [sys.executable, "-c", script, operation],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


class TestLoadExternalDataSingleFile(TestLoadExternalDataBase):
    @pytest.fixture(scope="class", params=["protobuf", "textproto"], autouse=True)
    def override_serialization_format(self, request):
        # Override the class' `serialization_format`.
        # This is not idiomatic pytest code which would structure all
        # these dependencies as explicit fixtures rather than setting
        # state on `self`. The code is as it is because it was
        # inherited from an earlier unittest/parameterized setup.
        self.serialization_format = request.param

    def create_external_data_tensors(
        self, tensors_data: list[tuple[list[Any], Any]]
    ) -> list[TensorProto]:
        tensor_filename = "tensors.bin"
        tensors = []

        with open(os.path.join(self.temp_dir, tensor_filename), "ab") as data_file:
            for value, tensor_name in tensors_data:
                tensor = from_array(np.array(value))
                offset = data_file.tell()
                if offset % 4096 != 0:
                    data_file.write(b"\0" * (4096 - offset % 4096))
                    offset = offset + 4096 - offset % 4096

                data_file.write(tensor.raw_data)
                set_external_data(
                    tensor,
                    location=tensor_filename,
                    offset=offset,
                    length=data_file.tell() - offset,
                )
                tensor.name = tensor_name
                tensor.ClearField("raw_data")
                tensor.data_location = onnx.TensorProto.EXTERNAL
                tensors.append(tensor)

        return tensors

    def test_load_external_single_file_data(self) -> None:
        model = onnx.load_model(self.model_filename, self.serialization_format)

        initializer_tensor = model.graph.initializer[0]
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)

        attribute_tensor = model.graph.node[0].attribute[0].t
        np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)

    def test_save_external_single_file_data(self) -> None:
        model = onnx.load_model(self.model_filename, self.serialization_format)

        temp_dir = os.path.join(self.temp_dir, "save_copy")
        os.mkdir(temp_dir)
        new_model_filename = os.path.join(temp_dir, "model.onnx")
        onnx.save_model(model, new_model_filename, self.serialization_format)

        new_model = onnx.load_model(new_model_filename, self.serialization_format)
        initializer_tensor = new_model.graph.initializer[0]
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)

        attribute_tensor = new_model.graph.node[0].attribute[0].t
        np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)

    @pytest.mark.parametrize("use_absolute_path", (True, False))
    def test_save_external_invalid_single_file_data_and_check(
        self, use_absolute_path: bool
    ) -> None:
        model = onnx.load_model(self.model_filename, self.serialization_format)

        model_dir = os.path.join(self.temp_dir, "save_copy")
        os.mkdir(model_dir)

        traversal_external_data_dir = os.path.join(
            self.temp_dir, "invalid_external_data"
        )
        os.mkdir(traversal_external_data_dir)

        if use_absolute_path:
            traversal_external_data_location = os.path.join(
                traversal_external_data_dir, "tensors.bin"
            )
        else:
            traversal_external_data_location = "../invalid_external_data/tensors.bin"

        external_data_dir = os.path.join(self.temp_dir, "external_data")
        os.mkdir(external_data_dir)
        new_model_filepath = os.path.join(model_dir, "model.onnx")

        def convert_model_to_external_data_no_check(model: ModelProto, location: str):
            for tensor in model.graph.initializer:
                if tensor.HasField("raw_data"):
                    set_external_data(tensor, location)

        convert_model_to_external_data_no_check(
            model,
            location=traversal_external_data_location,
        )

        with pytest.raises(onnx.checker.ValidationError):
            onnx.save_model(model, new_model_filepath, self.serialization_format)
        assert not os.path.exists(new_model_filepath)


@pytest.mark.parametrize("serialization_format", ["protobuf", "textproto"])
class TestSaveAllTensorsAsExternalData:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, serialization_format: str):
        self.serialization_format = serialization_format
        self.temp_dir: str = str(tmp_path)
        self.initializer_value = np.arange(6).reshape(3, 2).astype(np.float32) + 512
        self.attribute_value = np.arange(6).reshape(2, 3).astype(np.float32) + 256
        self.model = self.create_test_model_proto()

    def get_temp_model_filename(self):
        return os.path.join(self.temp_dir, str(uuid.uuid4()) + ".onnx")

    def create_data_tensors(
        self, tensors_data: list[tuple[list[Any], Any]]
    ) -> list[TensorProto]:
        tensors = []
        for value, tensor_name in tensors_data:
            tensor = from_array(np.array(value))
            tensor.name = tensor_name
            tensors.append(tensor)

        return tensors

    def create_test_model_proto(self) -> ModelProto:
        tensors = self.create_data_tensors(
            [
                (self.attribute_value, "attribute_value"),
                (self.initializer_value, "input_value"),
            ]
        )

        constant_node = onnx.helper.make_node(
            "Constant", inputs=[], outputs=["values"], value=tensors[0]
        )

        inputs = [
            helper.make_tensor_value_info(
                "input_value", onnx.TensorProto.FLOAT, self.initializer_value.shape
            )
        ]

        graph = helper.make_graph(
            [constant_node],
            "test_graph",
            inputs=inputs,
            outputs=[],
            initializer=[tensors[1]],
        )
        return helper.make_model(graph)

    def test_check_model(self) -> None:
        if self.serialization_format != "protobuf":
            pytest.skip("check_model supports protobuf only when provided as a path")
        checker.check_model(self.model)

    def test_convert_model_to_external_data_with_size_threshold(self) -> None:
        model_file_path = self.get_temp_model_filename()

        convert_model_to_external_data(self.model, size_threshold=1024)
        onnx.save_model(self.model, model_file_path, self.serialization_format)

        model = onnx.load_model(model_file_path, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        assert not initializer_tensor.HasField("data_location")

    def test_convert_model_to_external_data_without_size_threshold(self) -> None:
        model_file_path = self.get_temp_model_filename()
        convert_model_to_external_data(self.model, size_threshold=0)
        onnx.save_model(self.model, model_file_path, self.serialization_format)

        model = onnx.load_model(model_file_path, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        assert initializer_tensor.HasField("data_location")
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)

    def test_convert_model_to_external_data_from_one_file_with_location(self) -> None:
        model_file_path = self.get_temp_model_filename()
        external_data_file = str(uuid.uuid4())

        convert_model_to_external_data(
            self.model,
            size_threshold=0,
            all_tensors_to_one_file=True,
            location=external_data_file,
        )
        onnx.save_model(self.model, model_file_path, self.serialization_format)

        assert os.path.isfile(os.path.join(self.temp_dir, external_data_file))

        model = onnx.load_model(model_file_path, self.serialization_format)

        # test convert model from external data
        convert_model_from_external_data(model)
        model_file_path = self.get_temp_model_filename()
        onnx.save_model(model, model_file_path, self.serialization_format)
        model = onnx.load_model(model_file_path, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        assert not len(initializer_tensor.external_data)
        assert initializer_tensor.data_location == TensorProto.DEFAULT
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)

        attribute_tensor = model.graph.node[0].attribute[0].t
        assert not len(attribute_tensor.external_data)
        assert attribute_tensor.data_location == TensorProto.DEFAULT
        np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)

    def test_convert_model_to_external_data_from_one_file_without_location_uses_model_name(
        self,
    ) -> None:
        model_file_path = self.get_temp_model_filename()

        convert_model_to_external_data(
            self.model, size_threshold=0, all_tensors_to_one_file=True
        )
        onnx.save_model(self.model, model_file_path, self.serialization_format)

        assert os.path.isfile(model_file_path)
        assert os.path.isfile(os.path.join(self.temp_dir, model_file_path))

    def test_convert_model_to_external_data_one_file_per_tensor_without_attribute(
        self,
    ) -> None:
        model_file_path = self.get_temp_model_filename()

        convert_model_to_external_data(
            self.model,
            size_threshold=0,
            all_tensors_to_one_file=False,
            convert_attribute=False,
        )
        onnx.save_model(self.model, model_file_path, self.serialization_format)

        assert os.path.isfile(model_file_path)
        assert os.path.isfile(os.path.join(self.temp_dir, "input_value"))
        assert not os.path.isfile(os.path.join(self.temp_dir, "attribute_value"))

    def test_convert_model_to_external_data_one_file_per_tensor_with_attribute(
        self,
    ) -> None:
        model_file_path = self.get_temp_model_filename()

        convert_model_to_external_data(
            self.model,
            size_threshold=0,
            all_tensors_to_one_file=False,
            convert_attribute=True,
        )
        onnx.save_model(self.model, model_file_path, self.serialization_format)

        assert os.path.isfile(model_file_path)
        assert os.path.isfile(os.path.join(self.temp_dir, "input_value"))
        assert os.path.isfile(os.path.join(self.temp_dir, "attribute_value"))

    def test_convert_model_to_external_data_does_not_convert_attribute_values(
        self,
    ) -> None:
        model_file_path = self.get_temp_model_filename()

        convert_model_to_external_data(
            self.model,
            size_threshold=0,
            convert_attribute=False,
            all_tensors_to_one_file=False,
        )
        onnx.save_model(self.model, model_file_path, self.serialization_format)

        assert os.path.isfile(os.path.join(self.temp_dir, "input_value"))
        assert not os.path.isfile(os.path.join(self.temp_dir, "attribute_value"))

        model = onnx.load_model(model_file_path, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        assert initializer_tensor.HasField("data_location")

        attribute_tensor = model.graph.node[0].attribute[0].t
        assert not attribute_tensor.HasField("data_location")

    def test_convert_model_to_external_data_converts_attribute_values(self) -> None:
        model_file_path = self.get_temp_model_filename()

        convert_model_to_external_data(
            self.model, size_threshold=0, convert_attribute=True
        )
        onnx.save_model(self.model, model_file_path, self.serialization_format)

        model = onnx.load_model(model_file_path, self.serialization_format)

        initializer_tensor = model.graph.initializer[0]
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)
        assert initializer_tensor.HasField("data_location")

        attribute_tensor = model.graph.node[0].attribute[0].t
        np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)
        assert attribute_tensor.HasField("data_location")

    def test_save_model_does_not_convert_to_external_data_and_saves_the_model(
        self,
    ) -> None:
        model_file_path = self.get_temp_model_filename()
        onnx.save_model(
            self.model,
            model_file_path,
            self.serialization_format,
            save_as_external_data=False,
        )
        assert os.path.isfile(model_file_path)

        model = onnx.load_model(model_file_path, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        assert not initializer_tensor.HasField("data_location")

        attribute_tensor = model.graph.node[0].attribute[0].t
        assert not attribute_tensor.HasField("data_location")

    def test_save_model_does_convert_and_saves_the_model(self) -> None:
        model_file_path = self.get_temp_model_filename()
        onnx.save_model(
            self.model,
            model_file_path,
            self.serialization_format,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=None,
            size_threshold=0,
            convert_attribute=False,
        )

        model = onnx.load_model(model_file_path, self.serialization_format)

        initializer_tensor = model.graph.initializer[0]
        assert initializer_tensor.HasField("data_location")
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)

        attribute_tensor = model.graph.node[0].attribute[0].t
        assert not attribute_tensor.HasField("data_location")
        np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)

    def test_save_model_without_loading_external_data(self) -> None:
        model_file_path = self.get_temp_model_filename()
        onnx.save_model(
            self.model,
            model_file_path,
            self.serialization_format,
            save_as_external_data=True,
            location=None,
            size_threshold=0,
            convert_attribute=False,
        )
        # Save without load_external_data
        model = onnx.load_model(
            model_file_path, self.serialization_format, load_external_data=False
        )
        onnx.save_model(
            model,
            model_file_path,
            self.serialization_format,
            save_as_external_data=True,
            location=None,
            size_threshold=0,
            convert_attribute=False,
        )
        # Load the saved model again; Only works if the saved path is under the same directory
        model = onnx.load_model(model_file_path, self.serialization_format)

        initializer_tensor = model.graph.initializer[0]
        assert initializer_tensor.HasField("data_location")
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)

        attribute_tensor = model.graph.node[0].attribute[0].t
        assert not attribute_tensor.HasField("data_location")
        np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)

    def test_save_model_with_existing_raw_data_should_override(self) -> None:
        model_file_path = self.get_temp_model_filename()
        original_raw_data = self.model.graph.initializer[0].raw_data
        onnx.save_model(
            self.model,
            model_file_path,
            self.serialization_format,
            save_as_external_data=True,
            size_threshold=0,
        )
        assert os.path.isfile(model_file_path)

        model = onnx.load_model(
            model_file_path, self.serialization_format, load_external_data=False
        )
        initializer_tensor = model.graph.initializer[0]
        initializer_tensor.raw_data = b"dummpy_raw_data"
        # If raw_data and external tensor exist at the same time, override existing raw_data
        load_external_data_for_tensor(initializer_tensor, self.temp_dir)
        assert initializer_tensor.raw_data == original_raw_data


@pytest.mark.parametrize("serialization_format", ["protobuf", "textproto"])
class TestExternalDataToArray:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, serialization_format: str) -> None:
        self.serialization_format = serialization_format
        self.temp_dir = str(tmp_path)
        self._model_file_path: str = os.path.join(self.temp_dir, "model.onnx")
        self.large_data = np.random.rand(10, 60, 100).astype(np.float32)
        self.small_data = (200, 300)
        self.model = self.create_test_model()

    @property
    def model_file_path(self):
        return self._model_file_path

    def create_test_model(self) -> ModelProto:
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, self.large_data.shape)
        input_init = helper.make_tensor(
            name="X",
            data_type=TensorProto.FLOAT,
            dims=self.large_data.shape,
            vals=onnx.numpy_helper.tobytes_little_endian(self.large_data),
            raw=True,
        )

        shape_data = np.array(self.small_data, np.int64)
        shape_init = helper.make_tensor(
            name="Shape",
            data_type=TensorProto.INT64,
            dims=shape_data.shape,
            vals=onnx.numpy_helper.tobytes_little_endian(shape_data),
            raw=True,
        )
        C = helper.make_tensor_value_info("C", TensorProto.INT64, self.small_data)

        reshape = onnx.helper.make_node(
            "Reshape",
            inputs=["X", "Shape"],
            outputs=["Y"],
        )
        cast = onnx.helper.make_node(
            "Cast", inputs=["Y"], outputs=["C"], to=TensorProto.INT64
        )

        graph_def = helper.make_graph(
            [reshape, cast],
            "test-model",
            [X],
            [C],
            initializer=[input_init, shape_init],
        )
        return helper.make_model(graph_def, producer_name="onnx-example")

    def test_check_model(self) -> None:
        if self.serialization_format != "protobuf":
            pytest.skip("check_model supports protobuf only when provided as a path")

        checker.check_model(self.model)

    def test_reshape_inference_with_external_data_fail(self) -> None:
        onnx.save_model(
            self.model,
            self.model_file_path,
            self.serialization_format,
            save_as_external_data=True,
            all_tensors_to_one_file=False,
            size_threshold=0,
        )
        model_without_external_data = onnx.load(
            self.model_file_path, self.serialization_format, load_external_data=False
        )
        # Shape inference of Reshape uses ParseData
        # ParseData cannot handle external data and should throw the error as follows:
        # Cannot parse data from external tensors. Please load external data into raw data for tensor: Shape
        with pytest.raises(shape_inference.InferenceError):
            shape_inference.infer_shapes(
                model_without_external_data,
                strict_mode=True,
            )

    def test_to_array_with_external_data(self) -> None:
        onnx.save_model(
            self.model,
            self.model_file_path,
            self.serialization_format,
            save_as_external_data=True,
            all_tensors_to_one_file=False,
            size_threshold=0,
        )
        # raw_data of external tensor is not loaded
        model = onnx.load(
            self.model_file_path, self.serialization_format, load_external_data=False
        )
        # Specify self.temp_dir to load external tensor
        loaded_large_data = to_array(model.graph.initializer[0], self.temp_dir)
        np.testing.assert_allclose(loaded_large_data, self.large_data)

    def test_to_array_does_not_mutate_external_tensor(self) -> None:
        # to_array() must not leave the tensor in a state that mixes stale
        # external_data metadata with a populated raw_data field, otherwise
        # a subsequent save tries to rewrite (rather than append to) the
        # existing external data file.
        onnx.save_model(
            self.model,
            self.model_file_path,
            self.serialization_format,
            save_as_external_data=True,
            all_tensors_to_one_file=False,
            size_threshold=0,
        )
        model = onnx.load(
            self.model_file_path, self.serialization_format, load_external_data=False
        )
        initializer_tensor = model.graph.initializer[0]
        assert initializer_tensor.data_location == TensorProto.EXTERNAL
        assert not initializer_tensor.HasField("raw_data")

        to_array(initializer_tensor, self.temp_dir)

        assert initializer_tensor.data_location == TensorProto.EXTERNAL
        assert not initializer_tensor.HasField("raw_data")

        # Saving the model again, in the same directory as the original
        # external data file, must not fail.
        resaved_model_path = os.path.join(self.temp_dir, "resaved.onnx")
        onnx.save_model(model, resaved_model_path, self.serialization_format)

        reloaded_model = onnx.load(resaved_model_path, self.serialization_format)
        np.testing.assert_allclose(
            to_array(reloaded_model.graph.initializer[0], self.temp_dir),
            self.large_data,
        )

    def test_save_model_with_external_data_multiple_times(self) -> None:
        # Test onnx.save should respectively handle typical tensor and external tensor properly
        # 1st save: save two tensors which have raw_data
        # Only w_large will be stored as external tensors since it's larger than 1024
        onnx.save_model(
            self.model,
            self.model_file_path,
            self.serialization_format,
            save_as_external_data=True,
            all_tensors_to_one_file=False,
            location=None,
            size_threshold=1024,
            convert_attribute=True,
        )
        model_without_loading_external = onnx.load(
            self.model_file_path, self.serialization_format, load_external_data=False
        )
        large_input_tensor = model_without_loading_external.graph.initializer[0]
        assert large_input_tensor.HasField("data_location")
        np.testing.assert_allclose(
            to_array(large_input_tensor, self.temp_dir), self.large_data
        )

        small_shape_tensor = model_without_loading_external.graph.initializer[1]
        assert not small_shape_tensor.HasField("data_location")
        np.testing.assert_allclose(to_array(small_shape_tensor), self.small_data)

        # 2nd save: one tensor has raw_data (small); one external tensor (large)
        # Save them both as external tensors this time
        onnx.save_model(
            model_without_loading_external,
            self.model_file_path,
            self.serialization_format,
            save_as_external_data=True,
            all_tensors_to_one_file=False,
            location=None,
            size_threshold=0,
            convert_attribute=True,
        )

        model_without_loading_external = onnx.load(
            self.model_file_path, self.serialization_format, load_external_data=False
        )
        large_input_tensor = model_without_loading_external.graph.initializer[0]
        assert large_input_tensor.HasField("data_location")
        np.testing.assert_allclose(
            to_array(large_input_tensor, self.temp_dir), self.large_data
        )

        small_shape_tensor = model_without_loading_external.graph.initializer[1]
        assert small_shape_tensor.HasField("data_location")
        np.testing.assert_allclose(
            to_array(small_shape_tensor, self.temp_dir), self.small_data
        )


class TestNotAllowToLoadExternalDataOutsideModelDirectory(TestLoadExternalDataBase):
    """Essential test to check that onnx (validate) C++ code will not allow to load external_data outside the model
    directory.
    """

    def create_external_data_tensor(
        self, value: list[Any], tensor_name: str, location: str = ""
    ) -> TensorProto:
        tensor = from_array(np.array(value))
        tensor.name = tensor_name
        tensor_filename = location or f"{tensor_name}.bin"

        set_external_data(tensor, location=tensor_filename)

        tensor.ClearField("raw_data")
        tensor.data_location = onnx.TensorProto.EXTERNAL
        return tensor

    def test_check_model(self) -> None:
        """We only test the model validation as onnxruntime uses this to load the model."""
        self.model_filename = self.create_test_model("../../file.bin")
        with pytest.raises(onnx.checker.ValidationError):
            checker.check_model(self.model_filename)

    def test_check_model_relative(self) -> None:
        """More relative path test."""
        self.model_filename = self.create_test_model("../test/../file.bin")
        with pytest.raises(onnx.checker.ValidationError):
            checker.check_model(self.model_filename)

    def test_check_model_absolute(self) -> None:
        """ONNX checker disallows using absolute path as location in external tensor."""
        self.model_filename = self.create_test_model("//file.bin")
        with pytest.raises(onnx.checker.ValidationError):
            checker.check_model(self.model_filename)


@pytest.mark.skipif(os.name != "nt", reason="Skip Windows test")
class TestNotAllowToLoadExternalDataOutsideModelDirectoryOnWindows(
    TestNotAllowToLoadExternalDataOutsideModelDirectory
):
    """Essential test to check that onnx (validate) C++ code will not allow to load external_data outside the model
    directory.
    """

    def test_check_model(self) -> None:
        """We only test the model validation as onnxruntime uses this to load the model."""
        self.model_filename = self.create_test_model("..\\..\\file.bin")
        with pytest.raises(onnx.checker.ValidationError):
            checker.check_model(self.model_filename)

    def test_check_model_relative(self) -> None:
        """More relative path test."""
        self.model_filename = self.create_test_model("..\\test\\..\\file.bin")
        with pytest.raises(onnx.checker.ValidationError):
            checker.check_model(self.model_filename)

    def test_check_model_absolute(self) -> None:
        """ONNX checker disallows using absolute path as location in external tensor."""
        self.model_filename = self.create_test_model("C:/file.bin")
        with pytest.raises(onnx.checker.ValidationError):
            checker.check_model(self.model_filename)


class TestCheckExternalDataBounds:
    """check_model validates offset and length against the external data file the
    same way onnx.load does, and that the data fits the tensor's shape and type.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path) -> None:
        self.temp_dir = str(tmp_path)
        self.value = np.arange(6, dtype=np.float32)  # 24 bytes

    def create_test_model(self, external_data: dict[str, str], data: bytes) -> str:
        with open(os.path.join(self.temp_dir, "data.bin"), "wb") as data_file:
            data_file.write(data)
        tensor = from_array(self.value, "weight")
        tensor.ClearField("raw_data")
        tensor.data_location = TensorProto.EXTERNAL
        for key, value in {"location": "data.bin", **external_data}.items():
            entry = tensor.external_data.add()
            entry.key = key
            entry.value = value
        graph = helper.make_graph(
            [helper.make_node("Identity", ["weight"], ["y"])],
            "test_graph",
            inputs=[],
            outputs=[
                helper.make_tensor_value_info("y", TensorProto.FLOAT, self.value.shape)
            ],
            initializer=[tensor],
        )
        model_filename = os.path.join(self.temp_dir, "model.onnx")
        with open(model_filename, "wb") as model_file:
            model_file.write(helper.make_model(graph).SerializeToString())
        return model_filename

    @pytest.mark.parametrize("offset", [0, 8])
    @pytest.mark.parametrize("payload_size", [15, 16])
    def test_check_model_segment(self, offset: int, payload_size: int) -> None:
        model_filename = self.create_test_model(
            {"offset": str(offset)}, bytes(offset + payload_size)
        )
        model = onnx.load(model_filename, load_external_data=False)
        model.graph.initializer[0].segment.begin = 1
        model.graph.initializer[0].segment.end = 5
        with open(model_filename, "wb") as model_file:
            model_file.write(model.SerializeToString())

        if payload_size == 16:
            checker.check_model(model_filename)
        else:
            with pytest.raises(checker.ValidationError, match="too small"):
                checker.check_model(model_filename)

    def test_check_model_empty_segment(self) -> None:
        model_filename = self.create_test_model({}, b"")
        model = onnx.load(model_filename, load_external_data=False)
        segment = model.graph.initializer[0].segment
        segment.begin = segment.end = self.value.size
        with open(model_filename, "wb") as model_file:
            model_file.write(model.SerializeToString())
        checker.check_model(model_filename)

    @pytest.mark.parametrize(
        ("external_data", "prefix", "suffix"),
        [
            ({}, b"", b""),
            ({"offset": "8", "length": "24"}, b"\x00" * 8, b"\x00" * 8),
            ({"offset": "8"}, b"\x00" * 8, b""),
            # Data beyond what the shape needs is allowed, as for raw_data.
            ({}, b"", b"\x00" * 8),
        ],
        ids=["whole_file", "offset_and_length", "offset_to_end", "trailing_data"],
    )
    def test_check_model_valid(
        self, external_data: dict[str, str], prefix: bytes, suffix: bytes
    ) -> None:
        model_filename = self.create_test_model(
            external_data, prefix + self.value.tobytes() + suffix
        )
        checker.check_model(model_filename)
        onnx.load(model_filename)

    def test_check_model_in_memory_does_not_resolve_against_cwd(self) -> None:
        model_filename = self.create_test_model({}, self.value.tobytes())
        model = onnx.load(model_filename, load_external_data=False)
        with pytest.raises(
            checker.ValidationError, match="not checked from a file path"
        ):
            checker.check_model(model)

    def test_check_model_in_memory_hash_dotdot_location_still_rejected(self) -> None:
        # '#unused/../data.bin' normalizes to 'data.bin', so it refers to a
        # file even though the raw location starts with '#'.
        model_filename = self.create_test_model(
            {"location": "#unused/../data.bin"}, self.value.tobytes()
        )
        model = onnx.load(model_filename, load_external_data=False)
        with pytest.raises(
            checker.ValidationError, match="not checked from a file path"
        ):
            checker.check_model(model)

    def test_check_model_bare_filename_still_resolves_against_cwd(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model_filename = self.create_test_model({}, self.value.tobytes())
        monkeypatch.chdir(self.temp_dir)
        checker.check_model(os.path.basename(model_filename))

    @pytest.mark.parametrize(
        ("external_data", "data_size", "match"),
        [
            ({"offset": "abc"}, 24, "must be a non-negative integer"),
            ({"offset": "-1"}, 24, "must be a non-negative integer"),
            ({"offset": ""}, 24, "must be a non-negative integer"),
            ({"length": "1.5"}, 24, "must be a non-negative integer"),
            ({"length": "99999999999999999999"}, 24, "must be a non-negative integer"),
            ({"offset": "100"}, 24, "exceeds file size"),
            ({"offset": "8", "length": "24"}, 24, "exceeds available data"),
            ({}, 12, "too small for the declared shape"),
            ({"length": "12"}, 24, "too small for the declared shape"),
            ({"offset": "16"}, 24, "too small for the declared shape"),
        ],
        ids=[
            "offset_not_a_number",
            "offset_negative",
            "offset_empty",
            "length_not_an_integer",
            "length_overflows_int64",
            "offset_past_end",
            "length_past_end",
            "file_too_small",
            "length_too_small",
            "offset_leaves_too_little",
        ],
    )
    def test_check_model_invalid(
        self, external_data: dict[str, str], data_size: int, match: str
    ) -> None:
        model_filename = self.create_test_model(
            external_data, self.value.tobytes()[:data_size]
        )
        with pytest.raises(checker.ValidationError, match=match):
            checker.check_model(model_filename)

    def test_check_model_uses_last_value_of_each_key(self) -> None:
        # onnx.load uses the last "length"; it must be the one that is checked.
        model_filename = self.create_test_model({"length": "24"}, self.value.tobytes())
        model = onnx.load(model_filename, load_external_data=False)
        entry = model.graph.initializer[0].external_data.add()
        entry.key, entry.value = "length", "12"
        with open(model_filename, "wb") as model_file:
            model_file.write(model.SerializeToString())
        with pytest.raises(checker.ValidationError, match="too small"):
            checker.check_model(model_filename)

    @pytest.mark.parametrize("key", ["offset", "length"])
    def test_check_model_ignores_invalid_superseded_value(self, key: str) -> None:
        model_filename = self.create_test_model({key: "invalid"}, self.value.tobytes())
        model = onnx.load(model_filename, load_external_data=False)
        entry = model.graph.initializer[0].external_data.add()
        entry.key, entry.value = key, "0" if key == "offset" else "24"
        with open(model_filename, "wb") as model_file:
            model_file.write(model.SerializeToString())

        checker.check_model(model_filename)
        onnx.load(model_filename)

    @pytest.mark.parametrize("key", ["offset", "length"])
    def test_check_model_rejects_omitted_value(self, key: str) -> None:
        model_filename = self.create_test_model({}, self.value.tobytes())
        model = onnx.load(model_filename, load_external_data=False)
        entry = model.graph.initializer[0].external_data.add()
        entry.key = key
        with open(model_filename, "wb") as model_file:
            model_file.write(model.SerializeToString())

        with pytest.raises(checker.ValidationError, match="non-negative integer"):
            checker.check_model(model_filename)


class TestCheckExternalStringTensor:
    """STRING tensors cannot be stored externally: onnx.load would put their bytes
    into raw_data, which cannot hold strings.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path) -> None:
        self.temp_dir = str(tmp_path)

    @staticmethod
    def make_external_string_tensor(name: str) -> TensorProto:
        tensor = TensorProto(name=name, data_type=TensorProto.STRING, dims=[2])
        tensor.data_location = TensorProto.EXTERNAL
        tensor.external_data.add(key="location", value="strings.bin")
        return tensor

    def save(self, model: ModelProto, *, with_data_file: bool) -> str:
        if with_data_file:
            with open(os.path.join(self.temp_dir, "strings.bin"), "wb") as data_file:
                data_file.write(b"catdog")
        model_filename = os.path.join(self.temp_dir, "model.onnx")
        with open(model_filename, "wb") as model_file:
            model_file.write(model.SerializeToString())
        return model_filename

    @staticmethod
    def make_model(where: str, tensor: TensorProto) -> ModelProto:
        output = helper.make_tensor_value_info("y", TensorProto.STRING, [2])
        if where == "initializer":
            nodes = [helper.make_node("Identity", [tensor.name], ["y"])]
            return helper.make_model(
                helper.make_graph(nodes, "g", [], [output], [tensor])
            )
        constant = helper.make_node("Constant", [], ["y"], value=tensor)
        if where == "attribute":
            return helper.make_model(helper.make_graph([constant], "g", [], [output]))
        # where == "subgraph"
        branch = helper.make_graph([constant], "branch", [], [output])
        if_node = helper.make_node(
            "If", ["cond"], ["y"], then_branch=branch, else_branch=branch
        )
        cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
        return helper.make_model(helper.make_graph([if_node], "g", [cond], [output]))

    @pytest.mark.parametrize("with_data_file", [True, False])
    @pytest.mark.parametrize("where", ["initializer", "attribute", "subgraph"])
    def test_check_model_rejects(self, where: str, with_data_file: bool) -> None:
        model = self.make_model(where, self.make_external_string_tensor("labels"))
        model_filename = self.save(model, with_data_file=with_data_file)
        with pytest.raises(
            checker.ValidationError,
            match=r"STRING data .* should not be stored externally",
        ):
            checker.check_model(model_filename)

    def test_check_tensor_rejects(self) -> None:
        with pytest.raises(
            checker.ValidationError, match="should not be stored externally"
        ):
            checker.check_tensor(self.make_external_string_tensor("labels"))


class TestSaveAllTensorsAsExternalDataWithPath(TestSaveAllTensorsAsExternalData):
    def get_temp_model_filename(self) -> pathlib.Path:
        return pathlib.Path(super().get_temp_model_filename())


class TestExternalDataToArrayWithPath(TestExternalDataToArray):
    @property
    def model_file_path(self) -> pathlib.Path:
        return pathlib.Path(self._model_file_path)


class TestFunctionsAndSubGraphs:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path) -> None:
        temp_dir = str(tmp_path)
        self._model_file_path: str = os.path.join(temp_dir, "model.onnx")
        array = np.arange(4096).astype(np.float32)
        self._tensor = from_array(array, "tensor")

    def _check_is_internal(self, tensor: TensorProto) -> None:
        assert tensor.data_location == TensorProto.DEFAULT

    def _check_is_external(self, tensor: TensorProto) -> None:
        assert tensor.data_location == TensorProto.EXTERNAL

    def _check(self, model: ModelProto, nodes: Sequence[NodeProto]) -> None:
        """Check that the tensors in the model are externalized.

        The tensors in the specified sequence of Constant nodes are set to self._tensor,
        an internal tensor. The model is then converted to external data format.
        The tensors are then checked to ensure that they are externalized.

        Arguments:
            model: The model to check.
            nodes: A sequence of Constant nodes.

        """
        for node in nodes:
            assert node.op_type == "Constant"
            tensor = node.attribute[0].t
            tensor.CopyFrom(self._tensor)
            self._check_is_internal(tensor)

        convert_model_to_external_data(model, size_threshold=0, convert_attribute=True)

        for node in nodes:
            tensor = node.attribute[0].t
            self._check_is_external(tensor)

    def test_function(self) -> None:
        model_text = """
           <ir_version: 7,  opset_import: ["": 15, "local": 1]>
           agraph (float[N] X) => (float[N] Y)
            {
              Y = local.add(X)
            }

            <opset_import: ["" : 15],  domain: "local">
            add (float[N] X) => (float[N] Y) {
              C = Constant <value = float[1] {1.0}> ()
              Y = Add (X, C)
           }
        """
        model = parser.parse_model(model_text)
        self._check(model, [model.functions[0].node[0]])

    def test_subgraph(self) -> None:
        model_text = """
           <ir_version: 7,  opset_import: ["": 15, "local": 1]>
           agraph (bool flag, float[N] X) => (float[N] Y)
            {
              Y = if (flag) <
                then_branch = g1 () => (float[N] Y_then) {
                    B = Constant <value = float[1] {0.0}> ()
                    Y_then = Add (X, C)
                },
                else_branch = g2 () => (float[N] Y_else) {
                    C = Constant <value = float[1] {1.0}> ()
                    Y_else = Add (X, C)
                }
              >
            }
        """
        model = parser.parse_model(model_text)
        if_node = model.graph.node[0]
        constant_nodes = [attr.g.node[0] for attr in if_node.attribute]
        self._check(model, constant_nodes)


def _make_external_data_test_model() -> tuple[ModelProto, np.ndarray]:
    """Create a simple model with a large initializer suitable for external data tests."""
    model = parser.parse_model(
        """
        <ir_version: 7, opset_import: ["": 17]>
        agraph (float[100, 100] input) => (float[100, 100] output) {
            output = Identity(input)
        }
        """
    )
    array = np.ones((100, 100), dtype=np.float32)
    model.graph.initializer.append(from_array(array, name="weight"))
    return model, array


def test_save_model_pins_main_model_and_external_data_to_same_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing the output directory before the model write must not split the output."""
    base_path = tmp_path / "output"
    original_path = tmp_path / "output-original"
    base_path.mkdir()
    model_path = base_path / "model.onnx"
    model, _ = _make_external_data_test_model()
    original_context = onnx._ExternalDataContext
    replaced = False

    class ReplacingContext:
        def __init__(self, base_dir: str) -> None:
            self._context = original_context(base_dir)

        def open(self, location: str, tensor_name: str, read_only: bool) -> int:
            return self._context.open(location, tensor_name, read_only)

        def open_existing_model_file_for_write(self, location: str) -> int:
            nonlocal replaced
            try:
                return self._context.open_existing_model_file_for_write(location)
            finally:
                if not replaced:
                    base_path.rename(original_path)
                    base_path.mkdir()
                    replaced = True

        def create_model_file(self, location: str) -> int:
            return self._context.create_model_file(location)

    monkeypatch.setattr(onnx, "_ExternalDataContext", ReplacingContext)
    try:
        onnx.save_model(
            model,
            model_path,
            save_as_external_data=True,
            location="data.bin",
        )

        assert (original_path / "model.onnx").is_file()
        assert (original_path / "data.bin").is_file()
        assert not (base_path / "model.onnx").exists()
        assert not (base_path / "data.bin").exists()
    finally:
        if replaced:
            shutil.rmtree(base_path)
            original_path.rename(base_path)


def test_main_model_filename_may_contain_two_dots(tmp_path: Path) -> None:
    """The external-data '..' substring restriction must not affect model names."""
    model, _ = _make_external_data_test_model()
    model_path = tmp_path / "model..onnx"

    onnx.save_model(model, model_path)

    loaded = onnx.load_model(model_path)
    assert loaded.graph.name == model.graph.name


@pytest.mark.parametrize("mode", ["wb", "w+b"])
def test_named_model_file_object_saves_external_data(tmp_path: Path, mode: str) -> None:
    """A verified named output handle and its sidecars share one authority."""
    model, array = _make_external_data_test_model()
    model_path = tmp_path / "model.onnx"
    with open(model_path, mode) as model_file:
        onnx.save_model(
            model,
            model_file,
            save_as_external_data=True,
            location="data.bin",
        )

    loaded = onnx.load_model(model_path)
    np.testing.assert_allclose(to_array(loaded.graph.initializer[0]), array)


@pytest.mark.skipif(
    os.name == "nt", reason="Symlinks require elevated privileges on Windows"
)
class TestSaveExternalDataSymlinkProtection(TestLoadExternalDataBase):
    """Test that save_external_data rejects symlinks to prevent arbitrary file overwrites."""

    def test_save_rejects_symlink_target(self) -> None:
        """Saving external data must refuse to follow symlinks."""
        sensitive_file = os.path.join(self.temp_dir, "sensitive.txt")
        with open(sensitive_file, "w") as f:
            f.write("SENSITIVE DATA")

        model, array = _make_external_data_test_model()
        model_path = os.path.join(self.temp_dir, "model.onnx")
        ext_data = "data.bin"
        onnx.save_model(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=ext_data,
            size_threshold=1024,
        )

        # Replace external data file with a symlink to the sensitive file
        ext_data_path = os.path.join(self.temp_dir, ext_data)
        os.remove(ext_data_path)
        os.symlink(sensitive_file, ext_data_path)

        loaded_model = onnx.load(model_path, load_external_data=False)
        loaded_model.graph.initializer[0].raw_data = array.tobytes()

        with pytest.raises(checker.ValidationError):
            onnx.save_model(
                loaded_model,
                model_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=ext_data,
                size_threshold=1024,
            )

        # Sensitive file must not be modified
        with open(sensitive_file) as f:
            assert f.read() == "SENSITIVE DATA"


@pytest.mark.skipif(
    os.name == "nt", reason="Symlinks require elevated privileges on Windows"
)
class TestLoadExternalDataSymlinkProtection(TestLoadExternalDataBase):
    """Test that loading external data rejects symlinks to prevent arbitrary file reads."""

    def test_load_rejects_symlink_external_data(self) -> None:
        """Loading a model whose external data is a symlink must raise ValidationError."""
        model, _ = _make_external_data_test_model()
        model_path = os.path.join(self.temp_dir, "model.onnx")
        ext_data = "data.bin"
        onnx.save_model(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=ext_data,
            size_threshold=1024,
        )

        # Create a target file and replace external data with a symlink to it
        target_file = os.path.join(self.temp_dir, "target.txt")
        with open(target_file, "w") as f:
            f.write("SENSITIVE DATA")

        ext_data_path = os.path.join(self.temp_dir, ext_data)
        os.remove(ext_data_path)
        os.symlink(target_file, ext_data_path)

        # Loading with onnx.load (which loads external data) must fail
        with pytest.raises(checker.ValidationError):
            onnx.load(model_path)

    def test_load_external_data_for_model_rejects_symlink(self) -> None:
        """load_external_data_for_model must reject symlinked external data."""
        model, _ = _make_external_data_test_model()
        model_path = os.path.join(self.temp_dir, "model.onnx")
        ext_data = "data.bin"
        onnx.save_model(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=ext_data,
            size_threshold=1024,
        )

        # Replace external data with a symlink
        target_file = os.path.join(self.temp_dir, "target.txt")
        with open(target_file, "w") as f:
            f.write("SENSITIVE DATA")

        ext_data_path = os.path.join(self.temp_dir, ext_data)
        os.remove(ext_data_path)
        os.symlink(target_file, ext_data_path)

        # Load model without external data, then try to load external data explicitly
        loaded_model = onnx.load(model_path, load_external_data=False)
        with pytest.raises(checker.ValidationError):
            load_external_data_for_model(loaded_model, self.temp_dir)

    def test_load_rejects_parent_directory_symlink(self) -> None:
        """A symlink in the parent directory must be caught by realpath containment."""
        # Create a "sensitive" directory outside the model directory with a data file
        sensitive_dir = os.path.join(self.temp_dir, "sensitive")
        os.makedirs(sensitive_dir)
        secret_file = os.path.join(sensitive_dir, "secret.bin")
        with open(secret_file, "wb") as f:
            f.write(b"SENSITIVE DATA" * 100)

        # Create a model directory with a real subdir for saving
        model_dir = os.path.join(self.temp_dir, "model_dir")
        os.makedirs(model_dir)
        subdir_path = os.path.join(model_dir, "subdir")
        os.makedirs(subdir_path)

        # Create model with external data location "subdir/secret.bin"
        model, _ = _make_external_data_test_model()
        model_path = os.path.join(model_dir, "model.onnx")
        onnx.save_model(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="subdir/secret.bin",
            size_threshold=1024,
        )

        # Replace the real subdir with a symlink to the sensitive directory
        shutil.rmtree(subdir_path)
        os.symlink(sensitive_dir, subdir_path)

        # Loading must fail because realpath resolves outside model_dir.
        loaded_model = onnx.load(model_path, load_external_data=False)
        with pytest.raises(checker.ValidationError):
            load_external_data_for_model(loaded_model, model_dir)

    def test_empty_base_dir_rejects_parent_directory_symlink(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty base directory must mean the current directory for containment."""
        sensitive_dir = os.path.join(self.temp_dir, "sensitive")
        os.makedirs(sensitive_dir)
        with open(os.path.join(sensitive_dir, "secret.bin"), "wb") as f:
            f.write(b"SENSITIVE DATA" * 100)

        model_dir = os.path.join(self.temp_dir, "model_dir")
        os.makedirs(model_dir)
        subdir_path = os.path.join(model_dir, "subdir")
        os.makedirs(subdir_path)

        model, _ = _make_external_data_test_model()
        model_path = os.path.join(model_dir, "model.onnx")
        onnx.save_model(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="subdir/secret.bin",
            size_threshold=1024,
        )
        shutil.rmtree(subdir_path)
        os.symlink(sensitive_dir, subdir_path)

        monkeypatch.chdir(model_dir)
        loaded_model = onnx.load("model.onnx", load_external_data=False)
        with pytest.raises(checker.ValidationError):
            load_external_data_for_model(loaded_model, "")
        with pytest.raises(checker.ValidationError):
            checker.check_model("model.onnx")


@pytest.mark.skipif(os.name == "nt", reason="Hardlinks behave differently on Windows")
class TestLoadExternalDataHardlinkProtection(TestLoadExternalDataBase):
    """Test that loading external data rejects files with multiple hardlinks."""

    def test_load_rejects_hardlinked_external_data(self) -> None:
        """Loading a model whose external data has multiple hardlinks must raise ValidationError."""
        model, _ = _make_external_data_test_model()
        model_path = os.path.join(self.temp_dir, "model.onnx")
        ext_data = "data.bin"
        onnx.save_model(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=ext_data,
            size_threshold=1024,
        )

        # Create a hardlink to the external data file
        ext_data_path = os.path.join(self.temp_dir, ext_data)
        hardlink_path = os.path.join(self.temp_dir, "hardlink_data.bin")
        os.link(ext_data_path, hardlink_path)

        # Loading must fail because the external data file has multiple hardlinks.
        # Either the C++ checker or Python code catches this as ValidationError.
        with pytest.raises(checker.ValidationError):
            onnx.load(model_path)


class TestSaveExternalDataAbsolutePathValidation(TestLoadExternalDataBase):
    """Test that save_external_data rejects absolute paths."""

    def test_save_rejects_absolute_path(self) -> None:
        """Absolute paths must be rejected as external data locations."""
        array = np.ones((100,), dtype=np.float32)
        tensor = from_array(array, name="weight")
        set_external_data(tensor, location="/etc/passwd")
        with pytest.raises(checker.ValidationError) as exc_info:
            save_external_data(tensor, self.temp_dir)
        assert exc_info.value.code is None


def _tensor_with_offset(offset: int, location: str) -> TensorProto:
    """Create an in-memory tensor object that references an external 'location'."""
    tensor = from_array(np.ones((4,), dtype=np.float32), name="weight")
    set_external_data(tensor, location=location, offset=offset)
    return tensor


class TestSaveExternalDataOffsetBounds:
    """Test that save_external_data only writes at the end of the external file.

    A legitimate offset lands at the current end of the file, optionally bumped
    forward by an alignment gap. An offset before the end would overwrite existing
    bytes; one far past it would pad with an unbounded amount of zeros.
    """

    # On a fresh directory the external file does not exist yet, so its size is 0
    # and the padding written equals the offset itself.
    @pytest.mark.parametrize(
        "offset", [0, _MAX_EXTERNAL_DATA_PADDING], ids=["no_padding", "max_padding"]
    )
    def test_padding_up_to_the_cap_is_accepted(
        self, offset: int, tmp_path: Path
    ) -> None:
        """Padding a fresh file up to (and including) the cap is legitimate."""
        location = "data.bin"
        tensor = _tensor_with_offset(offset, location)
        save_external_data(tensor, str(tmp_path))
        assert (tmp_path / location).stat().st_size == offset + len(tensor.raw_data)

    @pytest.mark.parametrize(
        ("existing_data", "offset"),
        [
            (b"", _MAX_EXTERNAL_DATA_PADDING + 1),
            (b"some bytes longer than the offset", 4),
        ],
        ids=["padding_over_cap", "offset_inside_existing_file"],
    )
    def test_out_of_bounds_offset_rejected_without_writing(
        self, existing_data: bytes, offset: int, tmp_path: Path
    ) -> None:
        """Offsets past the cap or before the file's end are rejected, file untouched.

        Past the cap would pad with an unbounded amount of zeros; before the end
        would overwrite existing bytes. In both cases nothing must be written.
        """
        location = "data.bin"
        (tmp_path / location).write_bytes(existing_data)
        tensor = _tensor_with_offset(offset, location)
        with pytest.raises(checker.ValidationError, match=r"offset.*must be between"):
            save_external_data(tensor, str(tmp_path))
        assert (tmp_path / location).read_bytes() == existing_data


class TestWriteExternalDataTensorsOffsetOrder:
    """write_external_data_tensors() must write tensors in offset order.

    save_external_data() validates that a tensor's pre-assigned offset lands at the
    current end of its external file. If tensors are processed in graph order while
    their pre-assigned offsets describe a different file layout, that validation
    depends on graph order and rejects an otherwise valid layout.
    """

    def test_write_order_follows_offset_not_graph_order(self, tmp_path: Path) -> None:
        """Two initializers with a layout ('second' then 'first') opposite of graph order."""
        location = "data.bin"
        first = from_array(np.zeros((4,), dtype=np.float32), name="first")
        second = from_array(np.ones((4,), dtype=np.float32), name="second")
        first_bytes = first.raw_data
        second_bytes = second.raw_data

        # 'first' precedes 'second' in the graph's initializer list, but the
        # pre-assigned offsets place 'second' first in the external file.
        set_external_data(second, location, offset=0, length=len(second_bytes))
        set_external_data(
            first, location, offset=len(second_bytes), length=len(first_bytes)
        )

        graph = helper.make_graph(
            nodes=[],
            name="test",
            inputs=[],
            outputs=[],
            initializer=[first, second],
        )
        model = helper.make_model(graph)

        write_external_data_tensors(model, str(tmp_path))

        assert (tmp_path / location).read_bytes() == second_bytes + first_bytes


class TestExternalDataInfoSecurity:
    """Tests for ExternalDataInfo hardening against attribute injection and bounds.

    Covers all attack vectors from the security advisory: unknown key injection,
    dunder attribute injection, negative offset/length bypass, and validates
    that legitimate keys still work correctly.
    """

    @staticmethod
    def _make_tensor_with_external_data(
        entries: dict[str, str],
        tensor_name: str = "test_tensor",
    ) -> TensorProto:
        """Create a TensorProto with given external_data key-value entries."""
        tensor = TensorProto()
        tensor.name = tensor_name
        tensor.data_type = TensorProto.FLOAT
        tensor.dims.extend([4])
        tensor.data_location = TensorProto.EXTERNAL
        for key, value in entries.items():
            entry = tensor.external_data.add()
            entry.key = key
            entry.value = value
        return tensor

    def test_valid_external_data_accepted(self) -> None:
        """All valid external_data keys must be accepted and correctly parsed."""
        tensor = self._make_tensor_with_external_data(
            {
                "location": "weights.bin",
                "offset": "16",
                "length": "1024",
                "checksum": "sha256:abc123",
            }
        )
        info = ExternalDataInfo(tensor)
        assert info.location == "weights.bin"
        assert info.offset == 16
        assert isinstance(info.offset, int)
        assert info.length == 1024
        assert isinstance(info.length, int)
        assert info.checksum == "sha256:abc123"

    def test_unknown_key_rejected(self) -> None:
        """Unknown external_data keys must not be set as object attributes (CWE-915)."""
        tensor = self._make_tensor_with_external_data(
            {"location": "weights.bin", "malicious_attr": "evil_value"}
        )
        with warnings.catch_warnings(record=True, action="always") as caught:
            info = ExternalDataInfo(tensor)
        # Unknown attribute must NOT be set on the object
        assert not hasattr(info, "malicious_attr"), (
            "Unknown key 'malicious_attr' should not become an attribute"
        )
        # Valid key must still work
        assert info.location == "weights.bin"
        # A warning must have been emitted for the unknown key
        assert any("malicious_attr" in str(w.message) for w in caught), (
            "Expected warning about unknown key 'malicious_attr'"
        )

    def test_dunder_key_rejected(self) -> None:
        """Dunder keys like '__class__' must not be injected via external_data (CWE-915).

        Without the whitelist, setattr(self, '__class__', ...) would corrupt
        the object type, enabling type confusion attacks.
        """
        tensor = self._make_tensor_with_external_data({"location": "weights.bin"})
        # Add __class__ key via protobuf add() to mimic direct protobuf injection
        dunder_entry = tensor.external_data.add()
        dunder_entry.key = "__class__"
        dunder_entry.value = "builtins.dict"

        original_class = ExternalDataInfo
        with warnings.catch_warnings(record=True, action="always") as caught:
            info = ExternalDataInfo(tensor)
        # Object type must not have been corrupted
        assert isinstance(info, original_class)
        assert type(info).__name__ == "ExternalDataInfo"
        assert info.location == "weights.bin"
        # A warning must have been emitted for the dunder key
        assert any("__class__" in str(w.message) for w in caught), (
            "Expected warning about dunder key '__class__'"
        )

    def test_negative_offset_rejected(self) -> None:
        """Negative offset must raise ValueError to prevent seek(-1) attacks."""
        tensor = self._make_tensor_with_external_data(
            {"location": "weights.bin", "offset": "-1"}
        )
        with pytest.raises(ValueError, match="non-negative"):
            ExternalDataInfo(tensor)

    def test_negative_length_rejected(self) -> None:
        """Negative length must raise ValueError to prevent underflow attacks."""
        tensor = self._make_tensor_with_external_data(
            {"location": "weights.bin", "length": "-100"}
        )
        with pytest.raises(ValueError, match="non-negative"):
            ExternalDataInfo(tensor)

    def test_zero_offset_and_length_accepted(self) -> None:
        """Zero values for offset/length should be accepted (edge case for bounds check)."""
        tensor = self._make_tensor_with_external_data(
            {"location": "weights.bin", "offset": "0", "length": "0"}
        )
        # Should not raise — zero is a valid non-negative value
        info = ExternalDataInfo(tensor)
        assert info.location == "weights.bin"
        assert info.offset == 0
        assert info.length == 0

    def test_multiple_unknown_keys_all_rejected(self) -> None:
        """Multiple unknown keys in a single tensor must all be rejected."""
        tensor = self._make_tensor_with_external_data(
            {
                "location": "weights.bin",
                "evil_one": "a",
                "evil_two": "b",
                "__dict__": "c",
            }
        )
        with warnings.catch_warnings(record=True, action="always") as caught:
            info = ExternalDataInfo(tensor)
        assert not hasattr(info, "evil_one")
        assert not hasattr(info, "evil_two")
        assert info.location == "weights.bin"
        unknown_key_warnings = [
            str(w.message)
            for w in caught
            if "unknown external data key" in str(w.message).lower()
        ]
        assert len(unknown_key_warnings) == 1, (
            "Expected 1 aggregated warning for unknown keys"
        )
        # All unknown keys should be mentioned in the single warning
        assert "evil_one" in unknown_key_warnings[0]
        assert "evil_two" in unknown_key_warnings[0]
        assert "__dict__" in unknown_key_warnings[0]

    def test_allowed_keys_constant_is_frozen(self) -> None:
        """The whitelist must be a frozenset to prevent runtime mutation."""
        assert isinstance(_ALLOWED_EXTERNAL_DATA_KEYS, frozenset)
        assert (
            frozenset({"location", "offset", "length", "checksum", "basepath"})
            == _ALLOWED_EXTERNAL_DATA_KEYS
        )

    def test_non_numeric_offset_raises(self) -> None:
        """Non-numeric offset string must raise ValueError from int() conversion."""
        tensor = self._make_tensor_with_external_data(
            {"location": "weights.bin", "offset": "abc"}
        )
        with pytest.raises(ValueError):
            ExternalDataInfo(tensor)

    def test_non_numeric_length_raises(self) -> None:
        """Non-numeric length string must raise ValueError from int() conversion."""
        tensor = self._make_tensor_with_external_data(
            {"location": "weights.bin", "length": "not_a_number"}
        )
        with pytest.raises(ValueError):
            ExternalDataInfo(tensor)


class TestLoadExternalDataFileSizeValidation(TestLoadExternalDataBase):
    """Tests for defense-in-depth file-size validation in load_external_data_for_tensor."""

    def test_offset_exceeds_file_size_raises(self) -> None:
        """Offset beyond file size must raise ValueError."""
        array = np.ones((4,), dtype=np.float32)
        tensor = from_array(array, name="weight")
        set_external_data(tensor, location="data.bin")

        data_path = os.path.join(self.temp_dir, "data.bin")
        with open(data_path, "wb") as f:
            f.write(tensor.raw_data)

        file_size = os.path.getsize(data_path)
        # Set offset beyond file size
        set_external_data(tensor, location="data.bin", offset=file_size + 100)
        tensor.ClearField("raw_data")

        with pytest.raises(ValueError, match=r"offset.*exceeds file size"):
            load_external_data_for_tensor(tensor, self.temp_dir)

    def test_length_exceeds_available_data_raises(self) -> None:
        """Length that overflows available data must raise ValueError."""
        array = np.ones((4,), dtype=np.float32)
        tensor = from_array(array, name="weight")
        set_external_data(tensor, location="data.bin")

        data_path = os.path.join(self.temp_dir, "data.bin")
        with open(data_path, "wb") as f:
            f.write(tensor.raw_data)

        file_size = os.path.getsize(data_path)
        # Set length much larger than file
        set_external_data(tensor, location="data.bin", length=file_size * 1000)
        tensor.ClearField("raw_data")

        with pytest.raises(ValueError, match=r"length.*exceeds available data"):
            load_external_data_for_tensor(tensor, self.temp_dir)

    def test_valid_offset_and_length_load_correctly(self) -> None:
        """Valid offset+length within file size should load correctly."""
        array = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        tensor = from_array(array, name="weight")
        raw = tensor.raw_data

        data_path = os.path.join(self.temp_dir, "data.bin")
        with open(data_path, "wb") as f:
            f.write(raw)

        set_external_data(tensor, location="data.bin", offset=0, length=len(raw))
        tensor.ClearField("raw_data")

        load_external_data_for_tensor(tensor, self.temp_dir)
        assert tensor.raw_data == raw
