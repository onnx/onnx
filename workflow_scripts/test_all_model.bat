python ../onnx/workflow_scripts/get_model_path.py > models.txt
@echo off
for /F %%f in (models.txt) do ( 
    python ../onnx/workflow_scripts/test_single_model.py --test_model %%f
)