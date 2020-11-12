import onnx
import numpy
import onnxruntime as rt
from onnxruntime.datasets import get_example
import numpy.random
 
example1 = get_example("sigmoid.onnx")
model = onnx.load(example1)
onnx.checker.check_model(model)
onnx.checker.check_model(model, True)
inferred_model = onnx.shape_inference.infer_shapes(model, True)
# maybe also test with onnx.version_converter
temp_filename = "temp.onnx"
onnx.save(inferred_model, temp_filename)
print(onnx.__version__)
 
print(rt.__version__)
sess = rt.InferenceSession(temp_filename)
 
input_name = sess.get_inputs()[0].name
print("input name", input_name)
input_shape = sess.get_inputs()[0].shape
print("input shape", input_shape)
input_type = sess.get_inputs()[0].type
print("input type", input_type)
 
output_name = sess.get_outputs()[0].name
print("output name", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)
output_type = sess.get_outputs()[0].type
print("output type", output_type)
 
x = numpy.random.random((3,4,5))
x = x.astype(numpy.float32)
res = sess.run([output_name], {input_name: x})
print(res)