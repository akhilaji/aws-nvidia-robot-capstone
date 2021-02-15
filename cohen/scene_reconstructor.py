import numpy as np
from typing import Any
from nptyping import NDArray

class SceneReconstructor:
    def __init__(
            self: SceneReconstructor,
            fx: np.float32, fy: np.float32,
            cx: int, cy: int
        ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def add_frame(
            self: SceneReconstructor,
            rgb_img: NDArray[(Any, Any, 3), np.uint8]
        ):
        pass

    def finalize(
            self: SceneReconstructor
        ):
        pass