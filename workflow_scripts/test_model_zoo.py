# SPDX-License-Identifier: Apache-2.0
import argparse
import config
import gc
import onnx
import os
from pathlib import Path
import subprocess
import sys
import time

cwd_path = Path.cwd()


def run_lfs_install():
    result = subprocess.run(['git', 'lfs', 'install'], cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print('Git LFS install completed with return code= {}'.format(result.returncode))


def pull_lfs_file(file_name):
    result = subprocess.run(['git', 'lfs', 'pull', '--include', file_name, '--exclude', '\'\''], cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print('LFS pull completed with return code= {}'.format(result.returncode))
    print(result)


def run_lfs_prune():
    result = subprocess.run(['git', 'lfs', 'prune'], cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print('LFS prune completed with return code= {}'.format(result.returncode))


def main():
    parser = argparse.ArgumentParser(description='Test settings')
    # default: test all models in the repo
    # if test_dir is specified, only test files under that specified path
    parser.add_argument('--test_dir', required=False, default='', type=str,
                        help='Directory path for testing. e.g., text, vision')
    args = parser.parse_args()
    parent_dir = []
    # if not set, go throught each directory
    if not args.test_dir:
        for file in os.listdir():
            if os.path.isdir(file):
                parent_dir.append(file)
    else:
        parent_dir.append(args.test_dir)
    model_list = []
    for directory in parent_dir:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.onnx'):
                    onnx_model_path = os.path.join(root, file)
                    model_list.append(onnx_model_path)
                    print(onnx_model_path)
    # run lfs install before starting the tests
    run_lfs_install()

    print('=== Running ONNX Checker on {} models ==='.format(len(model_list)))
    # run checker on each model
    failed_models = []
    failed_messages = []
    skip_models = []
    for model_path in model_list:
        start = time.time()
        model_name = model_path.split('/')[-1]
        # if the model_path exists in the skip list, simply skip it
        if model_path.replace('\\', '/') in config.SKIP_CHECKER_MODELS:
            print('Skip model: {}'.format(model_path))
            skip_models.append(model_path)
            continue
        print('-----------------Testing: {}-----------------'.format(model_name))
        try:
            pull_lfs_file(model_path)
            model = onnx.load(model_path)
            # stricter onnx.checker with onnx.shape_inference
            onnx.checker.check_model(model, True)
            # remove the model to save space in CIs
            if os.path.exists(model_path):
                os.remove(model_path)
            # clean git lfs cache
            run_lfs_prune()
            print('[PASS]: {} is checked by onnx. '.format(model_name))

        except Exception as e:
            print('[FAIL]: {}'.format(e))
            failed_models.append(model_path)
            failed_messages.append((model_name, e))
        end = time.time()
        print('--------------Time used: {} secs-------------'.format(end - start))
        # enable gc collection to prevent MemoryError by loading too many large models
        gc.collect()

    if len(failed_models) == 0:
        print('{} models have been checked. {} models were skipped.'.format(len(model_list), len(skip_models)))
    else:
        print('In all {} models, {} models failed, {} models were skipped'.format(len(model_list), len(failed_models), len(skip_models)))
        for model, error in failed_messages:
            print('{} failed because: {}'.format(model, error))
        sys.exit(1)


if __name__ == '__main__':
    main()
