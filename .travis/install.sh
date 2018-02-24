#!/bin/bash
set -e
set -x

scripts_dir=$(dirname "${BASH_SOURCE[0]}")
source "$scripts_dir/common"

# install protobuf
pb_dir="$build_cache_dir/pb"
mkdir -p $pb_dir
wget -qO- "https://github.com/google/protobuf/releases/download/v$PB_VERSION/protobuf-$PB_VERSION.tar.gz" | tar -xvz -C "$pb_dir" --strip-components 1
ccache -z
cd "$pb_dir" && ./configure && make && make check && sudo make install && sudo ldconfig
ccache -s

APT_INSTALL_CMD='sudo apt-get install -y --no-install-recommends'

if [ "$TRAVIS_OS_NAME" = 'linux' ]; then
    ####################
    # apt dependencies #
    ####################
    sudo apt-get update
    $APT_INSTALL_CMD \
        autoconf \
        automake \
        build-essential \
        python \
        python-dev \
        python-pip \
        python-wheel \

elif [ "$TRAVIS_OS_NAME" = 'osx' ]; then
    #####################
    # brew dependencies #
    #####################
    brew update
    brew install python
    pip uninstall -y numpy  # use brew version (opencv dependency)
    brew tap homebrew/science  # for OpenCV

else
    echo "OS \"$TRAVIS_OS_NAME\" is unknown"
    exit 1
fi

# make sure this doesn't error
echo ARST

protoc --version

####################
# pip dependencies #
####################
sudo pip install \
    future \
    numpy \
    protobuf \
    pytest

if [[ $USE_NINJA == true ]]; then
    pip install ninja
fi
