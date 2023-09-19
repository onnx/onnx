# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
import argparse
import gc
import os
import sys
import time
from typing import List

import config

import onnx
from onnx import hub, version_converter

MIN_SHAPE_INFERENCE_OPSET = 4


def skip_model(error_message: str, skip_list: List[str], model_name: str):
    print(error_message)
    skip_list.append(model_name)


def main():
    parser = argparse.ArgumentParser(description="Test settings")
    # default: test all models in the repo
    # if test_dir is specified, only test files under that specified path
    parser.add_argument(
        "--test_dir",
        required=False,
        default="",
        type=str,
        help="Directory path for testing. e.g., text, vision",
    )
    model_list = hub.list_models()
    print(f"=== Running ONNX Checker on {len(model_list)} models ===")

    # run checker on each model
    failed_models = []
    failed_messages = []
    skip_models: List[str] = []
    for m in model_list:
        start = time.time()
        model_name = m.model
        model_path = m.model_path
        print(f"-----------------Testing: {model_name}-----------------")
        try:
            model = hub.load(model_name)
            # 1) Test onnx checker and shape inference
            if model.opset_import[0].version < MIN_SHAPE_INFERENCE_OPSET:
                # Ancient opset version does not have defined shape inference function
                onnx.checker.check_model(model)
                print(f"[PASS]: {model_name} is checked by onnx checker. ")
            else:
                # stricter onnx.checker with onnx.shape_inference
                onnx.checker.check_model(model, True)
                print(
                    f"[PASS]: {model_name} is checked by onnx checker with shape_inference. "
                )

                # 2) Test onnx version converter with upgrade functionality
                original_version = model.opset_import[0].version
                latest_opset_version = onnx.helper.VERSION_TABLE[-1][2]
                if original_version < latest_opset_version:
                    if model_path in config.SKIP_VERSION_CONVERTER_MODELS:
                        skip_model(
                            f"[SKIP]: model {model_name} is in the skip list for version converter. ",
                            skip_models,
                            model_name,
                        )
                    elif model_path.endswith("-int8.onnx"):
                        skip_model(
                            f"[SKIP]: model {model_name} is a quantized model using non-official ONNX domain. ",
                            skip_models,
                            model_name,
                        )
                    else:
                        converted = version_converter.convert_version(
                            model, original_version + 1
                        )
                        onnx.checker.check_model(converted, True)
                        print(
                            f"[PASS]: {model_name} can be version converted by original_version+1. "
                        )
                elif original_version == latest_opset_version:
                    skip_model(
                        f"[SKIP]: {model_name} is already the latest opset version. ",
                        skip_models,
                        model_name,
                    )
                else:
                    raise RuntimeError(
                        f"{model_name} has unsupported opset_version {original_version}. "
                    )

            # remove the model to save space in CIs
            if os.path.exists(model_name):
                os.remove(model_name)

        except Exception as e:
            print(f"[FAIL]: {e}")
            failed_models.append(model_name)
            failed_messages.append((model_name, e))
        end = time.time()
        print(f"--------------Time used: {end - start} secs-------------")
        # enable gc collection to prevent MemoryError by loading too many large models
        gc.collect()

    if len(failed_models) == 0:
        print(
            f"{len(model_list)} models have been checked. {len(skip_models)} models were skipped."
        )
    else:
        print(
            f"In all {len(model_list)} models, {len(failed_models)} models failed, {len(skip_models)} models were skipped"
        )
        for model_name, error in failed_messages:
            print(f"{model_name} failed because: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
