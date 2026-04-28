import pathlib
import sys

import torch

_THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import custom_add_ext  # noqa: E402


def custom_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return custom_add_ext.custom_add(x, y)
