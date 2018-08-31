# A tool to profile the ops in ONNX models
# Run by providing path to an ONNX model as the only CLI arg
# A set of all unique ops present in the model will be printed

import onnx
import sys
import argparse


def uniqueOps(model):
    print(set([n.op_type for n in model.graph.node]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate stats on ONNX models')
    parser.add_argument('path', type=str, help='a path to an ONNX model')
    args = parser.parse_args()
    model = onnx.load(args.path)
    uniqueOps(model)
