#!/bin/bash

set -ex

export MAX_JOBS=8

# setup sccache wrappers
if hash sccache 2>/dev/null; then
    SCCACHE_BIN_DIR="/tmp/sccache"
    mkdir -p "$SCCACHE_BIN_DIR"
    for compiler in cc c++ gcc g++ x86_64-linux-gnu-gcc; do
        (
            echo "#!/bin/sh"
            echo "exec $(which sccache) $(which $compiler) \"\$@\""
        ) > "$SCCACHE_BIN_DIR/$compiler"
        chmod +x "$SCCACHE_BIN_DIR/$compiler"
    done
    export PATH="$SCCACHE_BIN_DIR:$PATH"
fi

# setup virtualenv
VENV_DIR=/tmp/venv
PYTHON="$(which python)"
if [[ "${CIRCLE_JOB}" =~ py((2|3)\\.?[0-9]?\\.?[0-9]?) ]]; then
    PYTHON=$(which "python${BASH_REMATCH[1]}")
fi
$PYTHON -m virtualenv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install -U pip setuptools

# setup onnx as the submodule of pytorch
PYTORCH_DIR=/tmp/pytorch
ONNX_DIR="$PYTORCH_DIR/third_party/onnx"
git clone --recursive --quiet https://github.com/pytorch/pytorch.git "$PYTORCH_DIR"
rm -rf "$ONNX_DIR"
cp -r "$PWD" "$ONNX_DIR"

# install ninja to speedup the build
pip install ninja

# install everything
cd $PYTORCH_DIR
./scripts/onnx/install-develop.sh

# report sccache hit/miss stats
if hash sccache 2>/dev/null; then
    sccache --show-stats
fi
