"""

"""

from typing import Any, Callable

import numpy as np
from nptyping import NDArray

def characteristic_point_canny(
        canny_edges: NDArray[(Any, Any), np.uint8],
        depth_map: NDArray[(Any, Any), np.float32],
        x1: int, y1: int, x2:int, y2: int,
        pt_projector: Callable[[int, int, np.float32], NDArray[3, np.float32]]
    ) -> NDArray[3, np.float32]:
    pt_sum = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pt_count = 0
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            if canny_edges[y][x]:
                pt_sum += pt_projector(x, y, depth_map[y][x])
                pt_count += 1
    print(pt_sum)
    print(pt_count)
    return pt_sum / pt_count