# SPDX-License-Identifier: Apache-2.0
import config
import onnx
import os
import sys


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(script_dir, '../onnx/backend/test/data/node')
    count = failed_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.onnx'):
                test_dir_name = os.path.basename(os.path.normpath(root))
                onnx_model_path = os.path.join(root, file)
                try:
                    model = onnx.load(onnx_model_path)
                    # check model by ONNX checker
                    inferred_model = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
                    onnx.checker.check_model(inferred_model)

                except Exception as e:
                    failed_count += 1
                    print("{} failed: {}".format(test_dir_name, e))
                count += 1
    print('-----------------------------')
    if failed_count == 0:
        print("{} backend models passed.".format(count))
    else:
        print("{} failed in {} backend models.".format(failed_count, count))
        sys.exit(1)


if __name__ == '__main__':
    main()
