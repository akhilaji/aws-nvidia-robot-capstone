import cv2
import numpy as np
from typing import Tuple
from typing import Callable

def characteristic_point(rgb: np.array, depth_map: np.array,
    bb_p1: Tuple[int, int], bb_p2: Tuple[int, int],
    pt_proj: Callable[[int, int, float], Tuple[float, float, float]]):
    pass