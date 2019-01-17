import tempfile
import unittest
import uuid

import numpy as np  # type: ignore
import shutil

import os
import os.path as Path

import onnx
from onnx import checker, helper
from onnx import ModelProto, TensorProto
from onnx.external_data_helper import set_external_data
from onnx.external_data_helper import convert_model_to_external_data
from onnx.external_data_helper import convert_model_from_external_data
from onnx.external_data_helper import load_external_data_for_model
from onnx.numpy_helper import to_array, from_array
from typing import Any, Tuple, Text, List


class TestLoadExternalData(unittest.TestCase):

    def setUp(self):  # type: () -> None
        self.temp_dir = tempfile.mkdtemp()  # type: Text
        self.initializer_value = np.arange(6).reshape(3, 2).astype(np.float32) + 512
        self.attribute_value = np.arange(6).reshape(2, 3).astype(np.float32) + 256
        self.model_filename = self.create_test_model()

    def tearDown(self):  # type: () -> None
        shutil.rmtree(self.temp_dir)

    def get_temp_model_filename(self):  # type: () -> Text
        return os.path.join(self.temp_dir, str(uuid.uuid4()) + '.onnx')

    def create_external_data_tensor(self, value, tensor_name):  # type: (List[Any], Text) -> TensorProto
        tensor = from_array(np.array(value))
        tensor.name = tensor_name
        tensor_filename = "{}.bin".format(tensor_name)
        set_external_data(tensor, location=tensor_filename)

        with open(os.path.join(self.temp_dir, tensor_filename), 'wb') as data_file:
            data_file.write(tensor.raw_data)
        tensor.ClearField('raw_data')
        tensor.data_location = onnx.TensorProto.EXTERNAL
        return tensor

    def create_test_model(self):  # type: () -> Text

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

    def test_check_model(self):  # type: () -> None
        checker.check_model(self.model_filename)

    def test_load_external_data(self):  # type: () -> None
        model = onnx.load_model(self.model_filename)
        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))

    def test_load_external_data_for_model(self):  # type: () -> None
        model = onnx.load_model(self.model_filename, load_external_data=False)
        load_external_data_for_model(model, self.temp_dir)
        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))

    def test_save_external_data(self):  # type: () -> None
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


class TestLoadExternalDataSingleFile(unittest.TestCase):

    def setUp(self):  # type: () -> None
        self.temp_dir = tempfile.mkdtemp()  # type: Text
        self.initializer_value = np.arange(6).reshape(3, 2).astype(np.float32) + 512
        self.attribute_value = np.arange(6).reshape(2, 3).astype(np.float32) + 256
        self.model_filename = self.create_test_model()

    def tearDown(self):  # type: () -> None
        shutil.rmtree(self.temp_dir)

    def get_temp_model_filename(self):  # type: () -> Text
        return os.path.join(self.temp_dir, str(uuid.uuid4()) + '.onnx')

    def create_external_data_tensors(self, tensors_data):  # type: (List[Tuple[List[Any],Any]]) -> List[TensorProto]
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

    def create_test_model(self):  # type: () -> Text
        tensors = self.create_external_data_tensors([
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
        model = helper.make_model(graph)

        model_filename = os.path.join(self.temp_dir, 'model.onnx')
        with open(model_filename, "wb") as model_file:
            model_file.write(model.SerializeToString())
        return model_filename

    def test_check_model(self):  # type: () -> None
        checker.check_model(self.model_filename)

    def test_load_external_single_file_data(self):  # type: () -> None
        model = onnx.load_model(self.model_filename)

        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))

    def test_save_external_single_file_data(self):  # type: () -> None
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


class TestSaveAllTensorsAsExternalData(unittest.TestCase):

    def setUp(self):  # type: () -> None
        self.temp_dir = tempfile.mkdtemp()  # type: Text
        self.initializer_value = np.arange(6).reshape(3, 2).astype(np.float32) + 512
        self.attribute_value = np.arange(6).reshape(2, 3).astype(np.float32) + 256
        self.model = self.create_test_model()

    def tearDown(self):  # type: () -> None
        shutil.rmtree(self.temp_dir)

    def get_temp_model_filename(self):  # type: () -> Text
        return os.path.join(self.temp_dir, str(uuid.uuid4()) + '.onnx')

    def create_data_tensors(self, tensors_data):  # type: (List[Tuple[List[Any],Any]]) -> List[TensorProto]
        tensors = []
        for (value, tensor_name) in tensors_data:
            tensor = from_array(np.array(value))
            tensor.name = tensor_name
            tensors.append(tensor)

        return tensors

    def create_test_model(self):  # type: () -> ModelProto
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

    def test_check_model(self):  # type: () -> None
        checker.check_model(self.model)

    def test_convert_model_to_from_one_file(self):  # type: () -> None
        model_file_path = self.get_temp_model_filename()
        external_data_file = str(uuid.uuid4())
        convert_model_to_external_data(self.model, location=external_data_file)
        onnx.save_model(self.model, model_file_path)
        self.assertTrue(Path.isfile(model_file_path))
        self.assertTrue(Path.isfile(os.path.join(self.temp_dir, external_data_file)))
        model = onnx.load_model(model_file_path)
        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))

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
        self.assertFalse(len(initializer_tensor.external_data))
        self.assertEqual(attribute_tensor.data_location, TensorProto.DEFAULT)
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))

    def test_convert_model_to_external_data_one_file_per_tensor(self):  # type: () -> None
        model_file_path = self.get_temp_model_filename()
        convert_model_to_external_data(self.model, all_tensors_to_one_file=False)
        onnx.save_model(self.model, model_file_path)
        self.assertTrue(Path.isfile(model_file_path))
        self.assertTrue(Path.isfile(os.path.join(self.temp_dir, "input_value")))
        self.assertTrue(Path.isfile(os.path.join(self.temp_dir, "attribute_value")))
        model = onnx.load_model(model_file_path)
        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))


if __name__ == '__main__':
    unittest.main()
