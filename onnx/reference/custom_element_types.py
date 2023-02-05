# SPDX-License-Identifier: Apache-2.0

import numpy as np

bfloat16 = np.dtype((np.uint16, {"bfloat16": (np.uint16, 0)}))
floate4m3 = np.dtype((np.uint8, {"e4m3": (np.uint8, 0)}))
floate5m2 = np.dtype((np.uint8, {"e5m2": (np.uint8, 0)}))
