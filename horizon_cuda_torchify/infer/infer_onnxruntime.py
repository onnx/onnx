#!/usr/bin/env python3
import pathlib

import numpy as np
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parents[1]
INFER_INPUT_DIR = ROOT / "data" / "inference"
CUDA_ONNX = ROOT / "export" / "model_cuda.onnx"
TORCH_ONNX = ROOT / "export" / "model_torch.onnx"
OUT_DIR = ROOT / "infer"


def run_onnx(model_path: pathlib.Path, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    out = sess.run(["output"], {"x": x, "y": y})[0]
    return out


def load_input(name: str) -> np.ndarray:
    input_path = INFER_INPUT_DIR / f"{name}.npy"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. Run `python3 data/generate_input.py` first."
        )
    return np.load(input_path).astype(np.float32)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    x = load_input("x")
    y = load_input("y")

    out_cuda = run_onnx(CUDA_ONNX, x, y)
    out_torch = run_onnx(TORCH_ONNX, x, y)

    np.save(OUT_DIR / "output_cuda.npy", out_cuda)
    np.save(OUT_DIR / "output_torch.npy", out_torch)

    print("CUDA ONNX output saved to infer/output_cuda.npy")
    print("Torch ONNX output saved to infer/output_torch.npy")


if __name__ == "__main__":
    main()
