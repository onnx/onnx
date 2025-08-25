# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import unittest

import numpy as np

import onnx
import onnx.helper as xoh
import onnx.numpy_helper as xonh
import onnx.onnx2.helper as xoh2
from onnx import onnx2


class TestOnnx2Helper(unittest.TestCase):
    def get_dump_file(self, name, folder=None, clean=False) -> str:
        """Returns a filename to dump a model."""
        if folder is None:
            folder = "dump_test"
        if folder and not os.path.exists(folder):
            os.mkdir(folder)
        res = os.path.join(folder, name)
        if clean and os.path.exists(res):
            os.remove(res)
        return res

    def assertEqualModelProto(self, model1, model2):  # noqa: N802
        self.assertEqual(type(model1), type(model2))
        search = 'domain: ""'
        s1 = model1.SerializeToString()
        s2 = model2.SerializeToString()
        if len(s1) < 100000:
            spl1 = str(model1).split(search)
            spl2 = str(model2).split(search)
            if len(spl1) != len(spl2) or s1 != s2:
                n1 = self.get_dump_file("model1.onnx.txt")
                with open(n1, "w") as f:
                    f.write(str(model1))
                n2 = self.get_dump_file("model2.onnx.txt")
                with open(n2, "w") as f:
                    f.write(str(model2))
            self.assertEqual(len(spl1), len(spl2))
        self.assertEqual(s1, s2)

    @classmethod
    def make_model_gemm(cls, oh, tp):
        itype = tp.FLOAT
        return oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Gemm", ["X", "Y"], ["XY"]),
                    oh.make_node("Gemm", ["X", "Z"], ["XZ"]),
                    oh.make_node("Concat", ["XY", "XZ"], ["XYZ"], axis=1),
                ],
                "nd",
                [
                    oh.make_tensor_value_info("X", itype, [None, None]),
                    oh.make_tensor_value_info("Y", itype, [None, None]),
                    oh.make_tensor_value_info("Z", itype, [None, None]),
                ],
                [oh.make_tensor_value_info("XYZ", itype, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

    def test_model_gemm_onnx_to_onnx2(self):
        name = self.get_dump_file("test_model_gemm_onnx_to_onnx2.onnx")
        model = self.make_model_gemm(xoh, onnx.TensorProto)
        onnx.save(model, name)
        model2 = onnx2.load(name)
        self.assertEqual(len(model.graph.node), len(model2.graph.node))
        name2 = self.get_dump_file("test_model_gemm_onnx_to_onnx2_2.onnx")
        onnx2.save(model2, name2)
        model3 = onnx.load(name2)
        self.assertEqualModelProto(model, model3)

    def test_model_gemm_onnx2_to_onnx(self):
        name2 = self.get_dump_file("test_model_gemm_onnx2_to_onnx_2.onnx")
        model2 = self.make_model_gemm(xoh2, onnx2.TensorProto)
        onnx2.save(model2, name2)
        model = onnx.load(name2)
        self.assertEqual(len(model.graph.node), len(model2.graph.node))
        name = self.get_dump_file("test_model_gemm_onnx2_to_onnx.onnx")
        onnx.save(model, name)
        model3 = onnx.load(name)
        self.assertEqualModelProto(model, model3)

    def _get_model_with_initializers(self, oh, onh):
        TFLOAT = oh.TensorProto.FLOAT
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Add", ["Y1", "Y2"], ["Y"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, [32, 128])],
                [oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 32, 64])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(
                        np.random.rand(3, 5, 128, 64).astype(np.float32), name="Y1"
                    ),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(
                        np.random.rand(3, 5, 128, 64).astype(np.float32), name="Y2"
                    ),
                    onh.from_array(
                        np.array([1, 32, 128], dtype=np.int64), name="shape1"
                    ),
                    onh.from_array(
                        np.array([15, 128, 64], dtype=np.int64), name="shape2"
                    ),
                    onh.from_array(
                        np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"
                    ),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        return model

    def test_parallelized_loading(self):
        # saving with onnx
        name = self.get_dump_file("test_parallelized_loading.onnx")
        model = self._get_model_with_initializers(xoh, onnx.numpy_helper)
        onnx.save(model, name)
        # loading with onnx2
        model2 = onnx2.load(name, parallel=True, num_threads=2)
        self.assertEqual(len(model.graph.node), len(model2.graph.node))
        name2 = self.get_dump_file("test_parallelized_loading.onnx")
        onnx2.save(model2, name2)
        model3 = onnx.load(name2)
        self.assertEqualModelProto(model, model3)

    def test_writing_external_weights_write(self):
        nameo = self.get_dump_file("test_writing_external_weights.original.onnx")
        name = self.get_dump_file("test_writing_external_weights.onnx")
        weights = self.get_dump_file("test_writing_external_weights.data")
        model = self._get_model_with_initializers(xoh, onnx.numpy_helper)
        proto = onnx2.ModelProto()
        s = model.SerializeToString()
        with open(nameo, "wb") as f:
            f.write(s)
        proto.ParseFromString(s)
        proto.SerializeToFile(name, external_data_file=weights)
        reload = onnx.load(name)
        self.assertEqual(len(reload.graph.initializer), len(model.graph.initializer))

    def test_writing_external_weights_read(self):
        nameo = self.get_dump_file("test_writing_external_weights.original.onnx")
        name = self.get_dump_file("test_writing_external_weights.onnx")
        weights = self.get_dump_file("test_writing_external_weights.data")
        model = self._get_model_with_initializers(xoh, onnx.numpy_helper)
        proto = onnx2.ModelProto()
        s = model.SerializeToString()
        with open(nameo, "wb") as f:
            f.write(s)
        proto.ParseFromString(s)
        proto.SerializeToFile(name, external_data_file=weights)
        reload = onnx.load(name)
        self.assertEqual(len(reload.graph.initializer), len(model.graph.initializer))
        proto2 = onnx2.ModelProto()
        proto2.ParseFromFile(name, external_data_file=weights)
        self.assertEqual(len(proto2.graph.initializer), len(model.graph.initializer))

    def test_writing_external_weights_read_from_onnx(self):
        model = self._get_model_with_initializers(xoh, onnx.numpy_helper)
        expected = [xonh.to_array(i) for i in model.graph.initializer]
        name = self.get_dump_file("test_writing_external_weights_read_from_onnx.onnx")
        weights = self.get_dump_file(
            "test_writing_external_weights_read_from_onnx.data", clean=True
        )
        onnx.save(
            model, name, save_as_external_data=True, location=os.path.split(weights)[-1]
        )
        location = [init.data_location for init in model.graph.initializer]
        self.assertEqual(location, [0, 1, 0, 1, 0, 0, 0])
        proto2 = onnx2.ModelProto()
        proto2.ParseFromFile(name, external_data_file=weights)
        self.assertEqual(len(proto2.graph.initializer), len(model.graph.initializer))

        def tweak(i):
            t = onnx.TensorProto()
            t.ParseFromString(i.SerializeToString())
            return t

        got = [xonh.to_array(tweak(i)) for i in proto2.graph.initializer]
        self.assertEqual(len(expected), len(got))
        for a, b in zip(expected, got):
            np.testing.assert_allclose(a, b)

    def test_loading_external_weights(self):
        name = self.get_dump_file("test_loading_external_weights.onnx")
        weights = self.get_dump_file("test_loading_external_weights.data")
        model = self._get_model_with_initializers(xoh, onnx.numpy_helper)
        onnx.save(
            model, name, location=os.path.split(weights)[-1], save_as_external_data=True
        )
        proto = onnx2.load(name, location=weights)
        self.assertEqual(len(proto.graph.initializer), len(model.graph.initializer))
        proto_name = self.get_dump_file("test_loading_external_weights.2.onnx")
        onnx2.save(proto, proto_name)
        restored = onnx.load(proto_name)
        self.assertEqual(len(restored.graph.initializer), len(model.graph.initializer))


if __name__ == "__main__":
    unittest.main(verbosity=2)
