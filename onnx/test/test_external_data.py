# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
import os
import pathlib
import shutil
import tempfile
import unittest
import uuid
import warnings
from typing import TYPE_CHECKING, Any

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
    _ALLOWED_EXTERNAL_DATA_KEYS,
    ExternalDataInfo,
    convert_model_from_external_data,
    convert_model_to_external_data,
    load_external_data_for_model,
    load_external_data_for_tensor,
    save_external_data,
    set_external_data,
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

    @parameterized.parameterized.expand(
        itertools.product(
            (True, False),
        )
    )
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

        with self.assertRaises(onnx.checker.ValidationError):
            onnx.save_model(model, new_model_filepath, self.serialization_format)


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


@unittest.skipIf(
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

        with self.assertRaises(checker.ValidationError):
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
            self.assertEqual(f.read(), "SENSITIVE DATA")


@unittest.skipIf(
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
        with self.assertRaises(checker.ValidationError):
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
        with self.assertRaises(checker.ValidationError):
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
        with self.assertRaises(checker.ValidationError):
            load_external_data_for_model(loaded_model, model_dir)


@unittest.skipIf(os.name == "nt", reason="Hardlinks behave differently on Windows")
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
        with self.assertRaises(checker.ValidationError):
            onnx.load(model_path)


class TestSaveExternalDataAbsolutePathValidation(TestLoadExternalDataBase):
    """Test that save_external_data rejects absolute paths."""

    def test_save_rejects_absolute_path(self) -> None:
        """Absolute paths must be rejected as external data locations."""
        array = np.ones((100,), dtype=np.float32)
        tensor = from_array(array, name="weight")
        set_external_data(tensor, location="/etc/passwd")
        with self.assertRaises(checker.ValidationError):
            save_external_data(tensor, self.temp_dir)


class TestExternalDataInfoSecurity(unittest.TestCase):
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
        self.assertEqual(info.location, "weights.bin")
        self.assertEqual(info.offset, 16)
        self.assertIsInstance(info.offset, int)
        self.assertEqual(info.length, 1024)
        self.assertIsInstance(info.length, int)
        self.assertEqual(info.checksum, "sha256:abc123")

    def test_unknown_key_rejected(self) -> None:
        """Unknown external_data keys must not be set as object attributes (CWE-915)."""
        tensor = self._make_tensor_with_external_data(
            {"location": "weights.bin", "malicious_attr": "evil_value"}
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            info = ExternalDataInfo(tensor)
        # Unknown attribute must NOT be set on the object
        self.assertFalse(
            hasattr(info, "malicious_attr"),
            "Unknown key 'malicious_attr' should not become an attribute",
        )
        # Valid key must still work
        self.assertEqual(info.location, "weights.bin")
        # A warning must have been emitted for the unknown key
        self.assertTrue(
            any("malicious_attr" in str(w.message) for w in caught),
            "Expected warning about unknown key 'malicious_attr'",
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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            info = ExternalDataInfo(tensor)
        # Object type must not have been corrupted
        self.assertIsInstance(info, original_class)
        self.assertEqual(type(info).__name__, "ExternalDataInfo")
        self.assertEqual(info.location, "weights.bin")
        # A warning must have been emitted for the dunder key
        self.assertTrue(
            any("__class__" in str(w.message) for w in caught),
            "Expected warning about dunder key '__class__'",
        )

    def test_negative_offset_rejected(self) -> None:
        """Negative offset must raise ValueError to prevent seek(-1) attacks."""
        tensor = self._make_tensor_with_external_data(
            {"location": "weights.bin", "offset": "-1"}
        )
        with self.assertRaises(ValueError) as ctx:
            ExternalDataInfo(tensor)
        self.assertIn("non-negative", str(ctx.exception).lower())

    def test_negative_length_rejected(self) -> None:
        """Negative length must raise ValueError to prevent underflow attacks."""
        tensor = self._make_tensor_with_external_data(
            {"location": "weights.bin", "length": "-100"}
        )
        with self.assertRaises(ValueError) as ctx:
            ExternalDataInfo(tensor)
        self.assertIn("non-negative", str(ctx.exception).lower())

    def test_zero_offset_and_length_accepted(self) -> None:
        """Zero values for offset/length should be accepted (edge case for bounds check)."""
        tensor = self._make_tensor_with_external_data(
            {"location": "weights.bin", "offset": "0", "length": "0"}
        )
        # Should not raise — zero is a valid non-negative value
        info = ExternalDataInfo(tensor)
        self.assertEqual(info.location, "weights.bin")
        self.assertEqual(info.offset, 0)
        self.assertEqual(info.length, 0)

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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            info = ExternalDataInfo(tensor)
        self.assertFalse(hasattr(info, "evil_one"))
        self.assertFalse(hasattr(info, "evil_two"))
        self.assertEqual(info.location, "weights.bin")
        unknown_key_warnings = [
            str(w.message)
            for w in caught
            if "unknown external data key" in str(w.message).lower()
        ]
        self.assertEqual(
            len(unknown_key_warnings),
            1,
            "Expected 1 aggregated warning for unknown keys",
        )
        # All unknown keys should be mentioned in the single warning
        self.assertIn("evil_one", unknown_key_warnings[0])
        self.assertIn("evil_two", unknown_key_warnings[0])
        self.assertIn("__dict__", unknown_key_warnings[0])

    def test_allowed_keys_constant_is_frozen(self) -> None:
        """The whitelist must be a frozenset to prevent runtime mutation."""
        self.assertIsInstance(_ALLOWED_EXTERNAL_DATA_KEYS, frozenset)
        self.assertEqual(
            _ALLOWED_EXTERNAL_DATA_KEYS,
            frozenset({"location", "offset", "length", "checksum", "basepath"}),
        )

    def test_non_numeric_offset_raises(self) -> None:
        """Non-numeric offset string must raise ValueError from int() conversion."""
        tensor = self._make_tensor_with_external_data(
            {"location": "weights.bin", "offset": "abc"}
        )
        with self.assertRaises(ValueError):
            ExternalDataInfo(tensor)

    def test_non_numeric_length_raises(self) -> None:
        """Non-numeric length string must raise ValueError from int() conversion."""
        tensor = self._make_tensor_with_external_data(
            {"location": "weights.bin", "length": "not_a_number"}
        )
        with self.assertRaises(ValueError):
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

        with self.assertRaisesRegex(ValueError, "offset.*exceeds file size"):
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

        with self.assertRaisesRegex(ValueError, "length.*exceeds available data"):
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
        self.assertEqual(tensor.raw_data, raw)


if __name__ == "__main__":
    unittest.main()
