from typing import Any
from nptyping import NDArray

import numpy as np

class Intrinsics:
    def __init__(self,
            fx: np.float32, fy: np.float32,
            cx: int, cy: int
        ) -> None:
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

class Extrinsics:
    def __init__(self
        ) -> None:
        pass

class Camera:
    def __init__(self,
            intr: Intrinsics, extr: Extrinsics
        ) -> None:
        self.intr = intr
        self.extr = extr
    
    def project(self,
            u: int, v: int, d: np.float32
        ) -> NDArray[3, np.float32]:
        xz = (self.intr.cx - u) / self.intr.fx
        yz = (self.intr.cy - v) / self.intr.fy
        z = d / np.sqrt(1.0 + xz**2 + yz**2)
        return np.array([xz * z, yz * z, z], dtype=np.float32)
