# CUDA 自定义算子 torch 化适配地平线平台指导工程

本工程用于指导客户把现网 CUDA 自定义算子模型，改造成可在地平线工具链 (`hb_compile`) 编译部署的 ONNX。

## 平台使用提示（非常重要）

- **阶段一：torch 化与导出 ONNX**（含 CUDA 算子示例、torch 化示例、一致性校验）
  - 可在客户自己的模型开发环境执行（Python + PyTorch + CUDA）。
- **阶段二：结果验证（hb_compile）**
  - 需要在地平线官方要求的 Docker 环境中执行 `hb_compile`。

---

## 1. 工程目录（根目录即 `horizon_cuda_torchify`）

```text
horizon_cuda_torchify/
├── cuda_op/
│   ├── __init__.py
│   ├── build.sh
│   ├── custom_add.cpp
│   ├── custom_add_kernel.cu
│   └── setup.py
├── data/
│   └── generate_input.py
├── export/
│   ├── export_cuda_onnx.py
│   └── export_torch_onnx.py
├── horizon_test/
│   ├── compile_cuda_onnx.sh
│   ├── compile_torch_onnx.sh
│   ├── config_cuda.yaml
│   └── config_torch.yaml
├── infer/
│   ├── compare_outputs.py
│   └── infer_onnxruntime.py
├── model/
│   └── model_pair_impl.py
└── README.md
```

---

## 2. 阶段一：torch 化（开发环境执行）

### 2.1 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch onnx onnxruntime numpy setuptools
```

### 2.2 编译 CUDA 自定义算子

```bash
cd cuda_op
./build.sh
cd ..
```

### 2.3 导出 CUDA 版 ONNX（torch 化前）

```bash
python3 export/export_cuda_onnx.py
```

输出：`export/model_cuda.onnx`

### 2.4 导出 torch 版 ONNX（torch 化后）

```bash
python3 export/export_torch_onnx.py
```

输出：`export/model_torch.onnx`

### 2.5 生成推理与标定输入（双输入 x/y）

```bash
python3 data/generate_input.py --seed 2026 --num-calib 5
```

数据组织：

- 推理输入：
  - `data/inference/x.npy`
  - `data/inference/y.npy`
- 标定输入（按输入名分目录）：
  - `data/calibration/x/0.npy ...`
  - `data/calibration/y/0.npy ...`

### 2.6 一致性校验（ONNXRuntime）

```bash
python3 infer/infer_onnxruntime.py
python3 infer/compare_outputs.py
```

目标：

- `MSE` 接近 0
- `Cosine Similarity` 接近 1

---

## 3. 阶段二：结果验证（地平线 Docker 环境执行）

> 请在地平线工具链指定 Docker 环境中执行以下命令。

### 3.1 CUDA 自定义算子 ONNX（预期失败）

```bash
cd horizon_test
./compile_cuda_onnx.sh
```

### 3.2 torch 化 ONNX（预期成功）

```bash
./compile_torch_onnx.sh
cd ..
```

说明：

- `hb_compile` 命令仅使用 `--config` 参数；
- 模型路径通过 config 中 `model_parameters.onnx_model` 指定；
- 两个 config 分别为：
  - `horizon_test/config_cuda.yaml`
  - `horizon_test/config_torch.yaml`

---

## 4. 统一调用形式说明（便于客户对比）

`model/model_pair_impl.py` 中将 torch 化前后模型放在同一文件，并保持相同调用形式：

- `ModelWithCuda.forward -> custom_op_call(x, y)`
- `ModelTorch.forward -> custom_op_call(x, y)`

唯一差异在 `custom_op_call` 的实现：

- CUDA 版调用 `CustomAddFunction.apply`（导出为自定义域算子）；
- torch 版在注释标记处替换为标准算子 `x + y`。

---

## 5. 客户结论

客户模型若包含 CUDA 自定义算子，需先完成 **torch 化替换为标准 ONNX 算子**，再进入地平线 `hb_compile` 流程；这是可编译、可部署的前提。
