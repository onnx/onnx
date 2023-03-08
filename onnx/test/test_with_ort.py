# SPDX-License-Identifier: Apache-2.0
# This file is for testing ONNX with ONNXRuntime during ONNX Release
# Create a general scenario to use ONNXRuntime with ONNX
# pylint: disable=C0415
import unittest


class TestONNXRuntime(unittest.TestCase):
    def test_with_ort_example(self) -> None:
        try:
            import onnxruntime  # pylint: disable=W0611

            del onnxruntime
        except ImportError:
            raise unittest.SkipTest("onnxruntime not installed") from None

        from numpy import float32, random
        from onnxruntime import InferenceSession
        from onnxruntime.datasets import get_example

        from onnx import checker, load, save, shape_inference

        # get certain example model from ORT
        example1 = get_example("sigmoid.onnx")

        # test ONNX functions
        model = load(example1)
        checker.check_model(model)
        checker.check_model(model, True)
        inferred_model = shape_inference.infer_shapes(model, True)
        temp_filename = "temp.onnx"
        save(inferred_model, temp_filename)

        # test ONNX Runtime functions
        sess = InferenceSession(temp_filename)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        x = random.random((3, 4, 5))
        x = x.astype(float32)

        sess.run([output_name], {input_name: x})


if __name__ == "__main__":
    unittest.main()
