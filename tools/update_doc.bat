set ONNX_ML=1
python setup.py develop

python onnx\backend\test\cmd_tools.py generate-data

python onnx\backend\test\stat_coverage.py

python onnx\defs\gen_doc.py
set ONNX_ML=0
python onnx\defs\gen_doc.py
set ONNX_ML=1
