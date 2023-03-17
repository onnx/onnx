from os import environ

from onnx.helper import VERSION_TABLE

# TODO (https://github.com/microsoft/onnxruntime/issues/14932): Get max supported version from onnxruntime directly
# For now, bump the version here whenever there is a new onnxruntime release
# These env variables are used in onnxruntime tests in this repo
environ["ORT_MAX_ONNX_OPSET_SUPPORTED_VERSION"] = "18"
environ["ORT_MAX_ML_OPSET_SUPPORTED_VERSION"] = "3"
for version_info in VERSION_TABLE[::-1]:
    if version_info[2] == int(environ["ORT_MAX_ONNX_OPSET_SUPPORTED_VERSION"]):
        environ["ORT_MAX_IR_SUPPORTED_VERSION"] = str(version_info[1])
        break
