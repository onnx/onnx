export NUM_PROCESSOR=`grep -c ^processor /proc/cpuinfo`

git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout 3.11.x
git submodule update --init --recursive
./autogen.sh --disable-shared and enable PIC.

CFLAGS="-fPIC -g -O2" CXXFLAGS="-fPIC -g -O2" ./configure --disable-shared
make -j${NUM_PROCESSOR}
make check
make install
ldconfig
