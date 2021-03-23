from skeleton import depth
from skeleton import detect
from skeleton import pick
from skeleton import track

import numpy as np

from nptyping import NDArray
from typing import Any, List


class DetectionPipeline:
    """

    """

    def __init__(self,
            object_detector: detect.ObjectDetector,
            depth_estimator: depth.DepthEstimator,
            point_picker: pick.PointPicker,
            object_tracker: track.ObjectTracker,
        ):
        self.object_detector = object_detector
        self.depth_estimator = depth_estimator
        self.point_picker = point_picker
        self.object_tracker = object_tracker

    def __call__(self,
            frame: NDArray[(Any, Any, 3), np.uint8],
            intr
        ) -> List[detect.ObjectDetection]:
        detections = self.object_detector(frame)
        depth_map = self.depth_estimator(frame)
        self.object_tracker(detections)
        self.point_picker(detections, frame, depth_map, intr)
        return detections
