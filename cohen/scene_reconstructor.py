import numpy as np
from typing import Any
from nptyping import NDArray

class SceneReconstructor:
    def __init__(
            self: SceneReconstructor
        ):
        pass

    def add_frame(
            self: SceneReconstructor,
            rgb_img: NDArray[(Any, Any, 3), np.uint8]
        ):
        pass

    def finalize(
            self: SceneReconstructor
        ):
        pass