#!/usr/bin/env python3
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model_pair_impl import ModelTorch


def main() -> None:
    out_dir = ROOT / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_torch.onnx"

    model = ModelTorch().eval()
    x = torch.randn(1, 3, 8, 8, dtype=torch.float32)
    y = torch.randn(1, 3, 8, 8, dtype=torch.float32)

    torch.onnx.export(
        model,
        (x, y),
        str(out_path),
        input_names=["x", "y"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
    )

    print(f"Exported torch-only ONNX to: {out_path}")


if __name__ == "__main__":
    main()
