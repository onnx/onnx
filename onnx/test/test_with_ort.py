# SPDX-License-Identifier: Apache-2.0
# This file is for testing ONNX with ONNXRuntime during ONNX Release
# Create a general scenario to use ONNXRuntime with ONNX

def example_test_with_ort() -> None:
    import onnx
    import numpy  # type: ignore
    import onnxruntime as rt  # type: ignore
    from onnxruntime.datasets import get_example  # type: ignore
    import numpy.random  # type: ignore

    # get certain example model from ORT
    example1 = get_example("sigmoid.onnx")

    # test ONNX functions
    model = onnx.load(example1)
    onnx.checker.check_model(model)
    onnx.checker.check_model(model, True)
    inferred_model = onnx.shape_inference.infer_shapes(model, True)
    temp_filename = "temp.onnx"
    onnx.save(inferred_model, temp_filename)

    # test ONNXRuntime functions
    sess = rt.InferenceSession(temp_filename)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    x = numpy.random.random((3, 4, 5))
    x = x.astype(numpy.float32)

    sess.run([output_name], {input_name: x})


if __name__ == "__main__":
    example_test_with_ort()
