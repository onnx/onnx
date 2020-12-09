import argparse
import config
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
    parser.add_argument('--test_model', required=True, default='', type=str,
                        help='ONNX model path for testing.')
    args = parser.parse_args()
    model_path = args.test_model

    # run lfs install before starting the tests
    run_lfs_install()

    print('=== Running ONNX Checker on models ===')
    start = time.time()
    model_name = model_path.split('/')[-1]
    print('-----------------Testing: {}-----------------'.format(model_name))
    failed = False
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
        failed = True
    end = time.time()
    print('--------------Time used: {} secs-------------'.format(end - start))
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
