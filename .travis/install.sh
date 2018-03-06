#!/bin/bash

scripts_dir=$(dirname $(readlink -e "${BASH_SOURCE[0]}"))
source "$scripts_dir/common"

# install protobuf
pb_dir="$build_cache_dir/pb"
mkdir -p $pb_dir
wget -qO- "https://github.com/google/protobuf/releases/download/v$PB_VERSION/protobuf-$PB_VERSION.tar.gz" | tar -xvz -C "$pb_dir" --strip-components 1
ccache -z
cd "$pb_dir" && ./configure && make && make check && sudo make install && sudo ldconfig
ccache -s


if [[ $USE_NINJA == true ]]; then
    pip install ninja
fi
