# SPDX-License-Identifier: Apache-2.0
import tempfile
import unittest
import uuid

import numpy as np  # type: ignore
import shutil

import os
import os.path as Path

import onnx
from onnx import checker, helper, shape_inference
from onnx import ModelProto, TensorProto
from onnx.external_data_helper import set_external_data
from onnx.external_data_helper import convert_model_to_external_data
from onnx.external_data_helper import convert_model_from_external_data
from onnx.external_data_helper import load_external_data_for_model, load_external_data_for_tensor
from onnx.numpy_helper import to_array, from_array
from typing import Any, Tuple, List
import pytest  # type: ignore
import sys


class TestLoadExternalDataBase(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir: str = tempfile.mkdtemp()
        self.initializer_value = np.arange(6).reshape(3, 2).astype(np.float32) + 512
        self.attribute_value = np.arange(6).reshape(2, 3).astype(np.float32) + 256
        self.model_filename = self.create_test_model()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def get_temp_model_filename(self) -> str:
        return os.path.join(self.temp_dir, str(uuid.uuid4()) + '.onnx')

    def create_external_data_tensor(self, value: List[Any], tensor_name: str) -> TensorProto:
        tensor = from_array(np.array(value))
        tensor.name = tensor_name
        tensor_filename = f"{tensor_name}.bin"
        set_external_data(tensor, location=tensor_filename)

        with open(os.path.join(self.temp_dir, tensor_filename), 'wb') as data_file:
            data_file.write(tensor.raw_data)
        tensor.ClearField('raw_data')
        tensor.data_location = onnx.TensorProto.EXTERNAL
        return tensor

    def create_test_model(self) -> str:

        constant_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['values'],
            value=self.create_external_data_tensor(self.attribute_value, "attribute_value")
        )

        initializers = [self.create_external_data_tensor(self.initializer_value, "input_value")]
        inputs = [helper.make_tensor_value_info("input_value",
                                                onnx.TensorProto.FLOAT,
                                                self.initializer_value.shape)]

        graph = helper.make_graph([constant_node], "test_graph",
                                  inputs=inputs, outputs=[],
                                  initializer=initializers)
        model = helper.make_model(graph)

        model_filename = os.path.join(self.temp_dir, "model.onnx")
        with open(model_filename, "wb") as model_file:
            model_file.write(model.SerializeToString())

        return model_filename

    def test_check_model(self) -> None:
        checker.check_model(self.model_filename)


class TestLoadExternalData(TestLoadExternalDataBase):

    def test_load_external_data(self) -> None:
        model = onnx.load_model(self.model_filename)
        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))

    def test_load_external_data_for_model(self) -> None:
        model = onnx.load_model(self.model_filename, load_external_data=False)
        load_external_data_for_model(model, self.temp_dir)
        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))

    def test_save_external_data(self) -> None:
        model = onnx.load_model(self.model_filename)

        temp_dir = os.path.join(self.temp_dir, "save_copy")
        os.mkdir(temp_dir)
        new_model_filename = os.path.join(temp_dir, 'model.onnx')
        onnx.save_model(model, new_model_filename)

        new_model = onnx.load_model(new_model_filename)
        initializer_tensor = new_model.graph.initializer[0]
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

        attribute_tensor = new_model.graph.node[0].attribute[0].t
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))


class TestLoadExternalDataSingleFile(TestLoadExternalDataBase):

    def create_external_data_tensors(self, tensors_data: List[Tuple[List[Any], Any]]) -> List[TensorProto]:
        tensor_filename = "tensors.bin"
        tensors = []

        with open(os.path.join(self.temp_dir, tensor_filename), 'ab') as data_file:
            for (value, tensor_name) in tensors_data:
                tensor = from_array(np.array(value))
                offset = data_file.tell()
                if offset % 4096 != 0:
                    data_file.write(b"\0" * (4096 - offset % 4096))
                    offset = offset + 4096 - offset % 4096

                data_file.write(tensor.raw_data)
                set_external_data(tensor, location=tensor_filename, offset=offset, length=data_file.tell() - offset)
                tensor.name = tensor_name
                tensor.ClearField("raw_data")
                tensor.data_location = onnx.TensorProto.EXTERNAL
                tensors.append(tensor)

        return tensors

    def test_load_external_single_file_data(self) -> None:
        model = onnx.load_model(self.model_filename)

        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))

    def test_save_external_single_file_data(self) -> None:
        model = onnx.load_model(self.model_filename)

        temp_dir = os.path.join(self.temp_dir, "save_copy")
        os.mkdir(temp_dir)
        new_model_filename = os.path.join(temp_dir, 'model.onnx')
        onnx.save_model(model, new_model_filename)

        new_model = onnx.load_model(new_model_filename)
        initializer_tensor = new_model.graph.initializer[0]
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

        attribute_tensor = new_model.graph.node[0].attribute[0].t
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))


class TestSaveAllTensorsAsExternalData(TestLoadExternalDataBase):

    def setUp(self) -> None:
        self.temp_dir: str = tempfile.mkdtemp()
        self.initializer_value = np.arange(6).reshape(3, 2).astype(np.float32) + 512
        self.attribute_value = np.arange(6).reshape(2, 3).astype(np.float32) + 256
        self.model = self.create_test_model_proto()

    def create_data_tensors(self, tensors_data: List[Tuple[List[Any], Any]]) -> List[TensorProto]:
        tensors = []
        for (value, tensor_name) in tensors_data:
            tensor = from_array(np.array(value))
            tensor.name = tensor_name
            tensors.append(tensor)

        return tensors

    def create_test_model_proto(self) -> ModelProto:
        tensors = self.create_data_tensors([
            (self.attribute_value, "attribute_value"),
            (self.initializer_value, "input_value"),
        ])

        constant_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['values'],
            value=tensors[0]
        )

        inputs = [helper.make_tensor_value_info("input_value",
                                                onnx.TensorProto.FLOAT,
                                                self.initializer_value.shape)]

        graph = helper.make_graph([constant_node], "test_graph",
                                  inputs=inputs, outputs=[],
                                  initializer=[tensors[1]])
        return helper.make_model(graph)

    def test_check_model(self) -> None:
        checker.check_model(self.model)

    def test_convert_model_to_external_data_with_size_threshold(self) -> None:
        model_file_path = self.get_temp_model_filename()

        convert_model_to_external_data(self.model, size_threshold=1024)
        onnx.save_model(self.model, model_file_path)

        model = onnx.load_model(model_file_path)
        initializer_tensor = model.graph.initializer[0]
        self.assertFalse(initializer_tensor.HasField("data_location"))

    def test_convert_model_to_external_data_without_size_threshold(self) -> None:
        model_file_path = self.get_temp_model_filename()
        convert_model_to_external_data(self.model, size_threshold=0)
        onnx.save_model(self.model, model_file_path)

        model = onnx.load_model(model_file_path)
        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(initializer_tensor.HasField("data_location"))
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

    def test_convert_model_to_external_data_from_one_file_with_location(self) -> None:
        model_file_path = self.get_temp_model_filename()
        external_data_file = str(uuid.uuid4())

        convert_model_to_external_data(self.model, size_threshold=0, all_tensors_to_one_file=True, location=external_data_file)
        onnx.save_model(self.model, model_file_path)

        self.assertTrue(Path.isfile(os.path.join(self.temp_dir, external_data_file)))

        model = onnx.load_model(model_file_path)

        # test convert model from external data
        convert_model_from_external_data(model)
        model_file_path = self.get_temp_model_filename()
        onnx.save_model(model, model_file_path)
        model = onnx.load_model(model_file_path)
        initializer_tensor = model.graph.initializer[0]
        self.assertFalse(len(initializer_tensor.external_data))
        self.assertEqual(initializer_tensor.data_location, TensorProto.DEFAULT)
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertFalse(len(attribute_tensor.external_data))
        self.assertEqual(attribute_tensor.data_location, TensorProto.DEFAULT)
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))

    def test_convert_model_to_external_data_from_one_file_without_location_uses_model_name(self) -> None:
        model_file_path = self.get_temp_model_filename()

        convert_model_to_external_data(self.model, size_threshold=0, all_tensors_to_one_file=True)
        onnx.save_model(self.model, model_file_path)

        self.assertTrue(Path.isfile(model_file_path))
        self.assertTrue(Path.isfile(os.path.join(self.temp_dir, model_file_path)))

    def test_convert_model_to_external_data_one_file_per_tensor_without_attribute(self) -> None:
        model_file_path = self.get_temp_model_filename()

        convert_model_to_external_data(self.model, size_threshold=0, all_tensors_to_one_file=False, convert_attribute=False)
        onnx.save_model(self.model, model_file_path)

        self.assertTrue(Path.isfile(model_file_path))
        self.assertTrue(Path.isfile(os.path.join(self.temp_dir, "input_value")))
        self.assertFalse(Path.isfile(os.path.join(self.temp_dir, "attribute_value")))

    def test_convert_model_to_external_data_one_file_per_tensor_with_attribute(self) -> None:
        model_file_path = self.get_temp_model_filename()

        convert_model_to_external_data(self.model, size_threshold=0, all_tensors_to_one_file=False, convert_attribute=True)
        onnx.save_model(self.model, model_file_path)

        self.assertTrue(Path.isfile(model_file_path))
        self.assertTrue(Path.isfile(os.path.join(self.temp_dir, "input_value")))
        self.assertTrue(Path.isfile(os.path.join(self.temp_dir, "attribute_value")))

    def test_convert_model_to_external_data_does_not_convert_attribute_values(self) -> None:
        model_file_path = self.get_temp_model_filename()

        convert_model_to_external_data(self.model, size_threshold=0, convert_attribute=False, all_tensors_to_one_file=False)
        onnx.save_model(self.model, model_file_path)

        self.assertTrue(Path.isfile(os.path.join(self.temp_dir, "input_value")))
        self.assertFalse(Path.isfile(os.path.join(self.temp_dir, "attribute_value")))

        model = onnx.load_model(model_file_path)
        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(initializer_tensor.HasField("data_location"))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertFalse(attribute_tensor.HasField("data_location"))

    def test_convert_model_to_external_data_converts_attribute_values(self) -> None:
        model_file_path = self.get_temp_model_filename()

        convert_model_to_external_data(self.model, size_threshold=0, convert_attribute=True)
        onnx.save_model(self.model, model_file_path)

        model = onnx.load_model(model_file_path)

        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))
        self.assertTrue(initializer_tensor.HasField("data_location"))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))
        self.assertTrue(attribute_tensor.HasField("data_location"))

    def test_save_model_does_not_convert_to_external_data_and_saves_the_model(self) -> None:
        model_file_path = self.get_temp_model_filename()
        onnx.save_model(self.model, model_file_path, save_as_external_data=False)
        self.assertTrue(Path.isfile(model_file_path))

        model = onnx.load_model(model_file_path)
        initializer_tensor = model.graph.initializer[0]
        self.assertFalse(initializer_tensor.HasField("data_location"))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertFalse(attribute_tensor.HasField("data_location"))

    def test_save_model_does_convert_and_saves_the_model(self) -> None:
        model_file_path = self.get_temp_model_filename()
        onnx.save_model(self.model,
                        model_file_path,
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        location=None,
                        size_threshold=0,
                        convert_attribute=False)

        model = onnx.load_model(model_file_path)

        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(initializer_tensor.HasField("data_location"))
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertFalse(attribute_tensor.HasField("data_location"))
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))

    def test_save_model_without_loading_external_data(self) -> None:
        model_file_path = self.get_temp_model_filename()
        onnx.save_model(self.model,
                        model_file_path,
                        save_as_external_data=True,
                        location=None,
                        size_threshold=0,
                        convert_attribute=False)
        # Save without load_external_data
        model = onnx.load_model(model_file_path, load_external_data=False)
        onnx.save_model(model,
                        model_file_path,
                        save_as_external_data=True,
                        location=None,
                        size_threshold=0,
                        convert_attribute=False)
        # Load the saved model again; Only works if the saved path is under the same directory
        model = onnx.load_model(model_file_path)

        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(initializer_tensor.HasField("data_location"))
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertFalse(attribute_tensor.HasField("data_location"))
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))

    def test_save_model_with_existing_raw_data_should_override(self) -> None:
        model_file_path = self.get_temp_model_filename()
        original_raw_data = self.model.graph.initializer[0].raw_data
        onnx.save_model(self.model, model_file_path, save_as_external_data=True, size_threshold=0)
        self.assertTrue(Path.isfile(model_file_path))

        model = onnx.load_model(model_file_path, load_external_data=False)
        initializer_tensor = model.graph.initializer[0]
        initializer_tensor.raw_data = b'dummpy_raw_data'
        # If raw_data and external tensor exist at the same time, override existing raw_data
        load_external_data_for_tensor(initializer_tensor, self.temp_dir)
        self.assertEqual(initializer_tensor.raw_data, original_raw_data)


class TestExternalDataToArray(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir: str = tempfile.mkdtemp()
        self.model_file_path: str = os.path.join(self.temp_dir, 'model.onnx')
        self.large_data = np.random.rand(10, 60, 100).astype(np.float32)
        self.small_data = (200, 300)
        self.model = self.create_test_model()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def get_temp_model_filename(self) -> str:
        return os.path.join(self.temp_dir, str(uuid.uuid4()) + '.onnx')

    def create_test_model(self) -> ModelProto:
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, self.large_data.shape)
        input_init = helper.make_tensor(name='X', data_type=TensorProto.FLOAT,
            dims=self.large_data.shape, vals=self.large_data.tobytes(), raw=True)

        shape_data = np.array(self.small_data, np.int64)
        shape_init = helper.make_tensor(name='Shape', data_type=TensorProto.INT64,
            dims=shape_data.shape, vals=shape_data.tobytes(), raw=True)
        C = helper.make_tensor_value_info('C', TensorProto.INT64, self.small_data)

        reshape = onnx.helper.make_node(
            'Reshape',
            inputs=['X', 'Shape'],
            outputs=['Y'],
        )
        cast = onnx.helper.make_node(
            'Cast',
            inputs=['Y'],
            outputs=['C'],
            to=getattr(TensorProto, 'INT64')
        )

        graph_def = helper.make_graph(
            [reshape, cast],
            'test-model',
            [X],
            [C],
            initializer=[input_init, shape_init],
        )
        model = helper.make_model(graph_def, producer_name='onnx-example')
        return model

    def test_check_model(self) -> None:
        checker.check_model(self.model)

    def test_reshape_inference_with_external_data_fail(self) -> None:
        onnx.save_model(self.model, self.model_file_path, save_as_external_data=True, all_tensors_to_one_file=False, size_threshold=0)
        model_without_external_data = onnx.load(self.model_file_path, load_external_data=False)
        # Shape inference of Reshape uses ParseData
        # ParseData cannot handle external data and should throw the error as follows:
        # Cannot parse data from external tensors. Please load external data into raw data for tensor: Shape
        self.assertRaises(shape_inference.InferenceError, shape_inference.infer_shapes,
            model_without_external_data, strict_mode=True)

    def test_to_array_with_external_data(self) -> None:
        onnx.save_model(self.model,
                        self.model_file_path,
                        save_as_external_data=True,
                        all_tensors_to_one_file=False,
                        size_threshold=0)
        # raw_data of external tensor is not loaded
        model = onnx.load(self.model_file_path, load_external_data=False)
        # Specify self.temp_dir to load external tensor
        loaded_large_data = to_array(model.graph.initializer[0], self.temp_dir)
        self.assertTrue(np.allclose(loaded_large_data, self.large_data))

    def test_save_model_with_external_data_multiple_times(self) -> None:
        # Test onnx.save should respectively handle typical tensor and external tensor properly
        # 1st save: save two tensors which have raw_data
        # Only w_large will be stored as external tensors since it's larger than 1024
        onnx.save_model(self.model,
                        self.model_file_path,
                        save_as_external_data=True,
                        all_tensors_to_one_file=False,
                        location=None,
                        size_threshold=1024,
                        convert_attribute=True)
        model_without_loading_external = onnx.load(self.model_file_path, load_external_data=False)
        large_input_tensor = model_without_loading_external.graph.initializer[0]
        self.assertTrue(large_input_tensor.HasField("data_location"))
        self.assertTrue(np.allclose(to_array(large_input_tensor, self.temp_dir), self.large_data))

        small_shape_tensor = model_without_loading_external.graph.initializer[1]
        self.assertTrue(not small_shape_tensor.HasField("data_location"))
        self.assertTrue(np.allclose(to_array(small_shape_tensor), self.small_data))

        # 2nd save: one tensor has raw_data (small); one external tensor (large)
        # Save them both as external tensors this time
        onnx.save_model(model_without_loading_external,
                        self.model_file_path,
                        save_as_external_data=True,
                        all_tensors_to_one_file=False,
                        location=None,
                        size_threshold=0,
                        convert_attribute=True)

        model_without_loading_external = onnx.load(self.model_file_path, load_external_data=False)
        large_input_tensor = model_without_loading_external.graph.initializer[0]
        self.assertTrue(large_input_tensor.HasField("data_location"))
        self.assertTrue(np.allclose(to_array(large_input_tensor, self.temp_dir), self.large_data))

        small_shape_tensor = model_without_loading_external.graph.initializer[1]
        self.assertTrue(small_shape_tensor.HasField("data_location"))
        self.assertTrue(np.allclose(to_array(small_shape_tensor, self.temp_dir), self.small_data))


if __name__ == '__main__':
    unittest.main()
