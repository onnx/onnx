# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
import os
import pathlib
import tempfile
import unittest
import uuid
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import parameterized

import onnx
from onnx import (
    ModelProto,
    NodeProto,
    TensorProto,
    checker,
    helper,
    parser,
    shape_inference,
)
from onnx.external_data_helper import (
    convert_model_from_external_data,
    convert_model_to_external_data,
    ExternalDataInfo,
    load_external_data_for_model,
    load_external_data_for_tensor,
    save_external_data,
    set_external_data
)
from onnx.numpy_helper import from_array, to_array

if TYPE_CHECKING:
    from collections.abc import Sequence


class TestLoadExternalDataBase(unittest.TestCase):
    """Base class for testing external data related behaviors.

    Subclasses should be parameterized with a serialization format.
    """

    serialization_format: str = "protobuf"

    def setUp(self) -> None:
        self._temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir: str = self._temp_dir_obj.name
        self.initializer_value = np.arange(6).reshape(3, 2).astype(np.float32) + 512
        self.attribute_value = np.arange(6).reshape(2, 3).astype(np.float32) + 256
        self.model_filename = self.create_test_model()

    def tearDown(self) -> None:
        self._temp_dir_obj.cleanup()

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
            self.skipTest(
                "check_model supports protobuf only as binary when provided as a path"
            )
        checker.check_model(self.model_filename)


@parameterized.parameterized_class(
    [
        {"serialization_format": "protobuf"},
        {"serialization_format": "textproto"},
    ]
)
class TestLoadExternalData(TestLoadExternalDataBase):
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


@parameterized.parameterized_class(
    [
        {"serialization_format": "protobuf"},
        {"serialization_format": "textproto"},
    ]
)
class TestLoadExternalDataSingleFile(TestLoadExternalDataBase):
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
    
    def _save_external_data_without_validation(
        self,
        tensor: TensorProto, 
        base_path: str
    ) -> None:
        """
        Test helper: Writes tensor data to external file WITHOUT path validation.
        Used in test_save_external_invalid_single_file_data_and_check
        """
        info = ExternalDataInfo(tensor)
        external_data_file_path = os.path.join(base_path, info.location)

        # Retrieve the tensor's data from raw_data
        if not tensor.HasField("raw_data"):
            raise ValueError("raw_data field doesn't exist.")

        # Create directory if needed
        external_data_dir = os.path.dirname(external_data_file_path)
        if external_data_dir and not os.path.exists(external_data_dir):
            os.makedirs(external_data_dir, exist_ok=True)

        # Create file if it doesn't exist
        if not os.path.isfile(external_data_file_path):
            with open(external_data_file_path, "ab"):
                pass

        # Open file for reading and writing at random locations ('r+b')
        with open(external_data_file_path, "r+b") as data_file:
            data_file.seek(0, 2)
            if info.offset is not None:
                # Pad file to required offset if needed
                file_size = data_file.tell()
                if info.offset > file_size:
                    data_file.write(b"\0" * (info.offset - file_size))

                data_file.seek(info.offset)
            offset = data_file.tell()
            data_file.write(tensor.raw_data)
            set_external_data(tensor, info.location, offset, data_file.tell() - offset)

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

    @parameterized.parameterized.expand(itertools.product((True, False), (True, False)))
    def test_save_external_invalid_single_file_data_and_check(
        self, use_absolute_path: bool, use_model_path: bool
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
        
        with patch("onnx.external_data_helper.save_external_data", side_effect=self._save_external_data_without_validation):
            onnx.save_model(model, new_model_filepath, self.serialization_format)
            
        if use_model_path:
            with self.assertRaises(onnx.checker.ValidationError):
                _ = onnx.load_model(new_model_filepath, self.serialization_format)
        else:
            onnx_model = onnx.load_model(
                new_model_filepath, self.serialization_format, load_external_data=False
            )
            with self.assertRaises(onnx.checker.ValidationError):
                load_external_data_for_model(onnx_model, external_data_dir)


@parameterized.parameterized_class(
    [
        {"serialization_format": "protobuf"},
        {"serialization_format": "textproto"},
    ]
)
class TestSaveAllTensorsAsExternalData(unittest.TestCase):
    serialization_format: str = "protobuf"

    def setUp(self) -> None:
        self._temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir: str = self._temp_dir_obj.name
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

    @unittest.skipIf(
        serialization_format != "protobuf",
        "check_model supports protobuf only when provided as a path",
    )
    def test_check_model(self) -> None:
        checker.check_model(self.model)

    def test_convert_model_to_external_data_with_size_threshold(self) -> None:
        model_file_path = self.get_temp_model_filename()

        convert_model_to_external_data(self.model, size_threshold=1024)
        onnx.save_model(self.model, model_file_path, self.serialization_format)

        model = onnx.load_model(model_file_path, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        self.assertFalse(initializer_tensor.HasField("data_location"))

    def test_convert_model_to_external_data_without_size_threshold(self) -> None:
        model_file_path = self.get_temp_model_filename()
        convert_model_to_external_data(self.model, size_threshold=0)
        onnx.save_model(self.model, model_file_path, self.serialization_format)

        model = onnx.load_model(model_file_path, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(initializer_tensor.HasField("data_location"))
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

        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, external_data_file)))

        model = onnx.load_model(model_file_path, self.serialization_format)

        # test convert model from external data
        convert_model_from_external_data(model)
        model_file_path = self.get_temp_model_filename()
        onnx.save_model(model, model_file_path, self.serialization_format)
        model = onnx.load_model(model_file_path, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        self.assertFalse(len(initializer_tensor.external_data))
        self.assertEqual(initializer_tensor.data_location, TensorProto.DEFAULT)
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertFalse(len(attribute_tensor.external_data))
        self.assertEqual(attribute_tensor.data_location, TensorProto.DEFAULT)
        np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)

    def test_convert_model_to_external_data_from_one_file_without_location_uses_model_name(
        self,
    ) -> None:
        model_file_path = self.get_temp_model_filename()

        convert_model_to_external_data(
            self.model, size_threshold=0, all_tensors_to_one_file=True
        )
        onnx.save_model(self.model, model_file_path, self.serialization_format)

        self.assertTrue(os.path.isfile(model_file_path))
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, model_file_path)))

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

        self.assertTrue(os.path.isfile(model_file_path))
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "input_value")))
        self.assertFalse(os.path.isfile(os.path.join(self.temp_dir, "attribute_value")))

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

        self.assertTrue(os.path.isfile(model_file_path))
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "input_value")))
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "attribute_value")))

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

        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "input_value")))
        self.assertFalse(os.path.isfile(os.path.join(self.temp_dir, "attribute_value")))

        model = onnx.load_model(model_file_path, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(initializer_tensor.HasField("data_location"))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertFalse(attribute_tensor.HasField("data_location"))

    def test_convert_model_to_external_data_converts_attribute_values(self) -> None:
        model_file_path = self.get_temp_model_filename()

        convert_model_to_external_data(
            self.model, size_threshold=0, convert_attribute=True
        )
        onnx.save_model(self.model, model_file_path, self.serialization_format)

        model = onnx.load_model(model_file_path, self.serialization_format)

        initializer_tensor = model.graph.initializer[0]
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)
        self.assertTrue(initializer_tensor.HasField("data_location"))

        attribute_tensor = model.graph.node[0].attribute[0].t
        np.testing.assert_allclose(to_array(attribute_tensor), self.attribute_value)
        self.assertTrue(attribute_tensor.HasField("data_location"))

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
        self.assertTrue(os.path.isfile(model_file_path))

        model = onnx.load_model(model_file_path, self.serialization_format)
        initializer_tensor = model.graph.initializer[0]
        self.assertFalse(initializer_tensor.HasField("data_location"))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertFalse(attribute_tensor.HasField("data_location"))

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
        self.assertTrue(initializer_tensor.HasField("data_location"))
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertFalse(attribute_tensor.HasField("data_location"))
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
        self.assertTrue(initializer_tensor.HasField("data_location"))
        np.testing.assert_allclose(to_array(initializer_tensor), self.initializer_value)

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertFalse(attribute_tensor.HasField("data_location"))
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
        self.assertTrue(os.path.isfile(model_file_path))

        model = onnx.load_model(
            model_file_path, self.serialization_format, load_external_data=False
        )
        initializer_tensor = model.graph.initializer[0]
        initializer_tensor.raw_data = b"dummpy_raw_data"
        # If raw_data and external tensor exist at the same time, override existing raw_data
        load_external_data_for_tensor(initializer_tensor, self.temp_dir)
        self.assertEqual(initializer_tensor.raw_data, original_raw_data)


class TestSaveExternalDataPathValidation(unittest.TestCase):
    
    def setUp(self) -> None:
        self._temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir: str = self._temp_dir_obj.name

    def tearDown(self) -> None:
        self._temp_dir_obj.cleanup()
    
    def create_tensor_with_location(self, location: str) -> TensorProto:
        """Helper to create a TensorProto with external data location."""
        tensor = TensorProto()
        tensor.raw_data = b"test_data_content"
        ext_data = tensor.external_data.add()
        ext_data.key = "location"
        ext_data.value = location
        return tensor
    
    def test_valid_relative_path(self):
        """Test that valid relative paths are accepted."""
        tensor = self.create_tensor_with_location("data/tensor.bin")
        
        os.makedirs(os.path.join(self.temp_dir, "data"), exist_ok=True)
        save_external_data(tensor, self.temp_dir)
        
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "data", "tensor.bin")))
    
    def test_valid_simple_filename(self):
        """Test that simple filenames are accepted."""
        tensor = self.create_tensor_with_location("tensor.bin")
        
        save_external_data(tensor, self.temp_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "tensor.bin")))
    
    @unittest.skipIf(os.name != 'nt', "Windows paths only relevant on Windows")
    def test_reject_absolute_path_windows(self):
        """Test that absolute Windows paths are rejected."""
        # Use a safe but still absolute Windows path
        tensor = self.create_tensor_with_location("C:\\absolute\\path\\tensor.bin")
        
        with self.assertRaisesRegex(ValueError, "Unsafe path"):
            save_external_data(tensor, self.temp_dir)

    @unittest.skipIf(os.name == 'nt', "Unix-style absolute paths behave differently on Windows")
    def test_reject_absolute_path_unix(self):
        """Test that absolute Unix paths are rejected."""
        # Use a safe but still absolute Unix path
        tensor = self.create_tensor_with_location("/tmp/absolute/path/tensor.bin")
        
        with self.assertRaisesRegex(ValueError, "Unsafe path"):
            save_external_data(tensor, self.temp_dir)

            
    def test_reject_parent_directory_traversal(self):
        """Test that parent directory traversal (..) is rejected."""
        tensor = self.create_tensor_with_location("../../../etc/passwd")
        
        with self.assertRaisesRegex(ValueError, "Unsafe path"):
            save_external_data(tensor, self.temp_dir)
    
    def test_reject_single_parent_directory(self):
        """Test that even a single .. is rejected."""
        tensor = self.create_tensor_with_location("../tensor.bin")
        
        with self.assertRaisesRegex(ValueError, "Unsafe path"):
            save_external_data(tensor, self.temp_dir)
    
    def test_reject_hidden_traversal_in_middle(self):
        """Test that .. in the middle of a path is rejected."""
        tensor = self.create_tensor_with_location("data/../../../etc/passwd")
        
        with self.assertRaisesRegex(ValueError, "Unsafe path"):
            save_external_data(tensor, self.temp_dir)
    
    def test_valid_path_with_dots(self):
        """Test that paths with single dots (current directory) work."""
        tensor = self.create_tensor_with_location("./tensor.bin")
        
        save_external_data(tensor, self.temp_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "tensor.bin")))
    
    def test_valid_nested_path(self):
        """Test that deeply nested valid paths work."""
        tensor = self.create_tensor_with_location("models/v1/weights/tensor.bin")
        
        os.makedirs(os.path.join(self.temp_dir, "models", "v1", "weights"), exist_ok=True)
        save_external_data(tensor, self.temp_dir)
        
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "models", "v1", "weights", "tensor.bin")))


@parameterized.parameterized_class(
    [
        {"serialization_format": "protobuf"},
        {"serialization_format": "textproto"},
    ]
)
class TestExternalDataToArray(unittest.TestCase):
    serialization_format: str = "protobuf"

    def setUp(self) -> None:
        self._temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir: str = self._temp_dir_obj.name
        self._model_file_path: str = os.path.join(self.temp_dir, "model.onnx")
        self.large_data = np.random.rand(10, 60, 100).astype(np.float32)
        self.small_data = (200, 300)
        self.model = self.create_test_model()

    @property
    def model_file_path(self):
        return self._model_file_path

    def tearDown(self) -> None:
        self._temp_dir_obj.cleanup()

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

    @unittest.skipIf(
        serialization_format != "protobuf",
        "check_model supports protobuf only when provided as a path",
    )
    def test_check_model(self) -> None:
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
        self.assertRaises(
            shape_inference.InferenceError,
            shape_inference.infer_shapes,
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
        self.assertTrue(large_input_tensor.HasField("data_location"))
        np.testing.assert_allclose(
            to_array(large_input_tensor, self.temp_dir), self.large_data
        )

        small_shape_tensor = model_without_loading_external.graph.initializer[1]
        self.assertTrue(not small_shape_tensor.HasField("data_location"))
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
        self.assertTrue(large_input_tensor.HasField("data_location"))
        np.testing.assert_allclose(
            to_array(large_input_tensor, self.temp_dir), self.large_data
        )

        small_shape_tensor = model_without_loading_external.graph.initializer[1]
        self.assertTrue(small_shape_tensor.HasField("data_location"))
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
        with self.assertRaises(onnx.checker.ValidationError):
            checker.check_model(self.model_filename)

    def test_check_model_relative(self) -> None:
        """More relative path test."""
        self.model_filename = self.create_test_model("../test/../file.bin")
        with self.assertRaises(onnx.checker.ValidationError):
            checker.check_model(self.model_filename)

    def test_check_model_absolute(self) -> None:
        """ONNX checker disallows using absolute path as location in external tensor."""
        self.model_filename = self.create_test_model("//file.bin")
        with self.assertRaises(onnx.checker.ValidationError):
            checker.check_model(self.model_filename)


@unittest.skipIf(os.name != "nt", reason="Skip Windows test")
class TestNotAllowToLoadExternalDataOutsideModelDirectoryOnWindows(
    TestNotAllowToLoadExternalDataOutsideModelDirectory
):
    """Essential test to check that onnx (validate) C++ code will not allow to load external_data outside the model
    directory.
    """

    def test_check_model(self) -> None:
        """We only test the model validation as onnxruntime uses this to load the model."""
        self.model_filename = self.create_test_model("..\\..\\file.bin")
        with self.assertRaises(onnx.checker.ValidationError):
            checker.check_model(self.model_filename)

    def test_check_model_relative(self) -> None:
        """More relative path test."""
        self.model_filename = self.create_test_model("..\\test\\..\\file.bin")
        with self.assertRaises(onnx.checker.ValidationError):
            checker.check_model(self.model_filename)

    def test_check_model_absolute(self) -> None:
        """ONNX checker disallows using absolute path as location in external tensor."""
        self.model_filename = self.create_test_model("C:/file.bin")
        with self.assertRaises(onnx.checker.ValidationError):
            checker.check_model(self.model_filename)


class TestSaveAllTensorsAsExternalDataWithPath(TestSaveAllTensorsAsExternalData):
    def get_temp_model_filename(self) -> pathlib.Path:
        return pathlib.Path(super().get_temp_model_filename())


class TestExternalDataToArrayWithPath(TestExternalDataToArray):
    @property
    def model_file_path(self) -> pathlib.Path:
        return pathlib.Path(self._model_file_path)


class TestFunctionsAndSubGraphs(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = self._temp_dir_obj.name
        self._model_file_path: str = os.path.join(temp_dir, "model.onnx")
        array = np.arange(4096).astype(np.float32)
        self._tensor = from_array(array, "tensor")

    def tearDown(self) -> None:
        self._temp_dir_obj.cleanup()

    def _check_is_internal(self, tensor: TensorProto) -> None:
        self.assertEqual(tensor.data_location, TensorProto.DEFAULT)

    def _check_is_external(self, tensor: TensorProto) -> None:
        self.assertEqual(tensor.data_location, TensorProto.EXTERNAL)

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
            self.assertEqual(node.op_type, "Constant")
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


if __name__ == "__main__":
    unittest.main()
