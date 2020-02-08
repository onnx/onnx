:: Run this script from ONNX root directory under Anaconda.
set CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
set ONNX_ML=1

cd onnx
python gen_proto.py -l
python gen_proto.py -l --ml
cd ..

python setup.py develop

python onnx\backend\test\cmd_tools.py generate-data

python onnx\backend\test\stat_coverage.py

python onnx\defs\gen_doc.py
set ONNX_ML=0
python onnx\defs\gen_doc.py
set ONNX_ML=1