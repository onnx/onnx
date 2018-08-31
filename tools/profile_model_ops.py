# A tool to profile the ops in ONNX models
# Run by providing path to an ONNX model as the only CLI arg
# A set of all unique ops present in the model will be printed

import onnx
import sys

if len(sys.argv) == 2:
    model = onnx.load(sys.argv[1])
    print(set([n.op_type for n in model.graph.node]))
