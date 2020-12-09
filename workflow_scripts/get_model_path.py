import config
import os

parent_dir = []
for file in os.listdir():
    if os.path.isdir(file):
        parent_dir.append(file)
model_paths = ""
for directory in parent_dir:
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.onnx'):
                onnx_model_path = os.path.join(root, file)
                # if the model_path exists in the skip list, simply skip it
                if onnx_model_path.replace('\\', '/') in config.SKIP_CHECKER_MODELS:
                    continue
                print(onnx_model_path)
