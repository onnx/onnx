#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np


INPUT_NAMES = ["x", "y"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate inference inputs and Horizon calibration data"
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument(
        "--num-calib",
        type=int,
        default=1,
        help="Number of calibration samples per input",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    root = pathlib.Path(__file__).resolve().parent

    infer_dir = root / "inference"
    calib_dir = root / "calibration"
    infer_dir.mkdir(parents=True, exist_ok=True)
    calib_dir.mkdir(parents=True, exist_ok=True)

    for input_name in INPUT_NAMES:
        infer_sample = rng.standard_normal((1, 3, 8, 8), dtype=np.float32)
        np.save(infer_dir / f"{input_name}.npy", infer_sample)

        input_calib_dir = calib_dir / input_name
        input_calib_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(args.num_calib):
            calib_sample = rng.standard_normal((1, 3, 8, 8), dtype=np.float32)
            np.save(input_calib_dir / f"{idx}.npy", calib_sample)

    print(f"Inference inputs saved under: {infer_dir}")
    print(f"Calibration inputs saved under: {calib_dir}")


if __name__ == "__main__":
    main()
