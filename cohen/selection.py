"""

"""

from typing import Any, Callable, Tuple

import cv2
import numpy as np
from nptyping import NDArray

def bounding_box(
        rgb_img: NDArray[(Any, Any, ...), Any],
        x1: int, y1: int,
        x2: int, y2: int
    ) -> NDArray[(Any, Any, ...), Any]:
    return rgb_img[y1:y2, x1:x2]

def characteristic_point_from_crop(
        bb_rgb_img: NDArray[(Any, Any, 3), np.uint8],
        depth_map: NDArray[(Any, Any), np.float32],
        pt_proj: Callable[[int, int, float], Tuple[float, float, float]]
    ) -> Tuple[float, float, float]:
    """

    """
    pass

def characteristic_point_from_uncropped(
        rgb_img: NDArray[(Any, Any, 3), np.uint8],
        depth_map: NDArray[(Any, Any), np.float32],
        bb_p1: Tuple[int, int], bb_p2: Tuple[int, int],
        pt_proj: Callable[[int, int, float], Tuple[float, float, float]]
    ) -> Tuple[float, float, float]:
    
    pass