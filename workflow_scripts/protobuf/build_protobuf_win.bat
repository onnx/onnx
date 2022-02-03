if [%1] ==[] (
    set arch=x64
) else (
    set arch=%1
)
set original_path=%cd%

echo "Build protobuf from source on Windows."
git clone https://github.com/protocolbuffers/protobuf.git
mkdir protobuf_install
set protobuf_root_dir=%cd%\protobuf_install
cd protobuf
git checkout v3.16.0
git submodule update --init --recursive
mkdir build_source && cd build_source
cmake ..\cmake -G "Visual Studio 16 2019" -A %arch% -DCMAKE_INSTALL_PREFIX=%protobuf_root_dir% -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_EXAMPLES=OFF .
msbuild protobuf.sln /m /p:Configuration=Release
msbuild INSTALL.vcxproj /p:Configuration=Release
echo "Protobuf installation complete."
echo "Set paths"
set PATH=%protobuf_root_dir%\bin;%protobuf_root_dir%\lib;%protobuf_root_dir%\include;%PATH%
protoc
echo "Protobuf installation complete."
cd %original_path%
