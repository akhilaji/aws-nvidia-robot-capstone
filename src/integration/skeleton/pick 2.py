from skeleton import detect

import numpy as np

from nptyping import NDArray
from typing import Any, List

class PointPicker:
    """

    """

    def __call__(self,
            detections: List[detect.ObjectDetection],
            frame: NDArray[(Any, Any, 3), np.uint8],
            depth: NDArray[(Any, Any), float],
            intr
        ) -> None:
        """

        """
        pass
