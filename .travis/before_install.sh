#!/bin/bash

# Don't source setup.sh here, because the virtualenv might not be set up yet

export NUMCORES=`grep -c ^processor /proc/cpuinfo`
if [ ! -n "$NUMCORES" ]; then
  export NUMCORES=`sysctl -n hw.ncpu`
fi
echo Using $NUMCORES cores

# Install dependencies
if [ "$TRAVIS_OS_NAME" == "linux" ]; then
  sudo apt-get update
  APT_INSTALL_CMD='sudo apt-get install -y --no-install-recommends'
  $APT_INSTALL_CMD dos2unix

  # Install protobuf
  pb_dir="~/.cache/pb"
  mkdir -p "$pb_dir"
  wget -qO- "https://github.com/google/protobuf/releases/download/v${PB_VERSION}/protobuf-${PB_VERSION}.tar.gz" | tar -xz -C "$pb_dir" --strip-components 1
  ccache -z
  cd "$pb_dir" && ./configure && make -j${NUMCORES} && make check && sudo make install && sudo ldconfig
  ccache -s

  # Setup Python.
  export PYTHON_DIR="/usr/bin"
elif [ "$TRAVIS_OS_NAME" == "osx" ]; then
  brew install ccache protobuf

  # Setup Python.
  export PYTHON_DIR="/usr/local/bin"
  brew install ${PYTHON_VERSION}
else
  echo Unknown OS: $TRAVIS_OS_NAME
  exit 1
fi

# TODO consider using "python3.6 -m venv"
# which is recommended by python3.6 and may make some difference
pip install virtualenv
virtualenv -p "${PYTHON_DIR}/${PYTHON_VERSION}" "${HOME}/virtualenv"
source "${HOME}/virtualenv/bin/activate"
python --version

pip install pytest-cov nbval

if [[ $USE_NINJA == true ]]; then
    pip install ninja
fi

# Update all existing python packages
pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U
