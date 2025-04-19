# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

param(
    [Parameter()]
    [String]$cmake_arch = "x64",

    [Parameter()]
    [String]$build_type = "Release"
)

echo "Build ABSL-cpp from source on Windows."
Invoke-WebRequest -Uri https://github.com/abseil/abseil-cpp/releases/download/20230802.2/abseil-cpp-20230802.2.tar.gz -OutFile abseil-cpp.tar.gz -Verbose
tar -xvf abseil-cpp.tar.gz

echo "Build protobuf from source on Windows."
Invoke-WebRequest -Uri https://github.com/protocolbuffers/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz -OutFile protobuf.tar.gz -Verbose
tar -xvf protobuf.tar.gz
cd protobuf-25.1
mkdir protobuf_install
$protobuf_install_dir = Get-Location
mkdir build
cd build
$protobuf_root_dir = Get-Location

cmake -G "Visual Studio 17 2022" -A $cmake_arch -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -DABSL_MSVC_STATIC_RUNTIME=OFF -DBUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_EXAMPLES=OFF -DABSL_ROOT_DIR="$protobuf_root_dir/../../abseil-cpp-20230802.2" -DCMAKE_CXX_STANDARD=17 -DABSL_PROPAGATE_CXX_STD=on -DCMAKE_INSTALL_PREFIX="$protobuf_install_dir" ..
cmake --build . --config $build_type --target install
echo "Protobuf installation complete."
echo "Set paths"
$env:Path = "$protobuf_install_dir/lib;$protobuf_install_dir/include;$protobuf_install_dir/bin;$env:Path"
protoc
cd ../../
