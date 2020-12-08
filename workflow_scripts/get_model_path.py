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
                print(onnx_model_path)
