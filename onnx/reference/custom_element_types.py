# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

bfloat16 = np.dtype((np.uint16, {"bfloat16": (np.uint16, 0)}))
float8e4m3fn = np.dtype((np.uint8, {"e4m3fn": (np.uint8, 0)}))
float8e4m3fnuz = np.dtype((np.uint8, {"e4m3fnuz": (np.uint8, 0)}))
float8e5m2 = np.dtype((np.uint8, {"e5m2": (np.uint8, 0)}))
float8e5m2fnuz = np.dtype((np.uint8, {"e5m2fnuz": (np.uint8, 0)}))
uint4 = np.dtype((np.uint8, {"uint4": (np.uint8, 0)}))
int4 = np.dtype((np.int8, {"int4": (np.int8, 0)}))
