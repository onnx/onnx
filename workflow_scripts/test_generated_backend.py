import onnx
import os
import pytest

def main():
    count = 0
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(script_dir, '../onnx/backend/test/data/node')
    count = 0
    fail_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.onnx'):
                test_dir_name = os.path.basename(os.path.normpath(root))
                try:
                    onnx_model_path = os.path.join(root, file)
                    model = onnx.load(onnx_model_path)
                    onnx.checker.check_model(model)
                    inferred_model = onnx.shape_inference.infer_shapes(model)
                    onnx.checker.check_model(inferred_model)
                except Exception as e:
                    fail_count += 1
                    print("{} fail: {}".format(test_dir_name, e))
                count += 1
    print("{} failed in {} backend models.".format(fail_count, count))

if __name__ == '__main__':
    main()
