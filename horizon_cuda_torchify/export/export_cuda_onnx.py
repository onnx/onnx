#!/usr/bin/env python3
import pathlib
import sys

import onnx
from onnx import TensorProto, helper
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model_pair_impl import ModelWithCuda


def add_function_body(model_path: pathlib.Path) -> None:
    model = onnx.load(str(model_path))
    function_node = helper.make_node("Add", ["x", "y"], ["z"])
    function = helper.make_function(
        domain="custom_domain",
        fname="CustomAdd",
        inputs=["x", "y"],
        outputs=["z"],
        nodes=[function_node],
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.functions.extend([function])
    model.opset_import.extend([helper.make_opsetid("custom_domain", 1)])
    onnx.save(model, str(model_path))


def main() -> None:
    out_dir = ROOT / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_cuda.onnx"

    model = ModelWithCuda().eval().cuda()
    x = torch.randn(1, 3, 8, 8, device="cuda", dtype=torch.float32)
    y = torch.randn(1, 3, 8, 8, device="cuda", dtype=torch.float32)

    torch.onnx.export(
        model,
        (x, y),
        str(out_path),
        input_names=["x", "y"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=False,
    )

    add_function_body(out_path)
    print(f"Exported CUDA custom-op ONNX to: {out_path}")


if __name__ == "__main__":
    main()
