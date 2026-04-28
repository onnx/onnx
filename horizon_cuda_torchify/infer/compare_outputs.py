#!/usr/bin/env python3
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "infer"


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    numerator = float(np.dot(a_flat, b_flat))
    denominator = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    return numerator / denominator


def main() -> None:
    out_cuda = np.load(OUT_DIR / "output_cuda.npy")
    out_torch = np.load(OUT_DIR / "output_torch.npy")

    mse_val = mse(out_cuda, out_torch)
    cos_val = cosine_similarity(out_cuda, out_torch)

    print(f"MSE: {mse_val:.10f}")
    print(f"Cosine Similarity: {cos_val:.10f}")


if __name__ == "__main__":
    main()
