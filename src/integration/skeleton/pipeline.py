from skeleton.depth import DepthEstimator
from skeleton.detect import ObjectDetector, ObjectDetection
from skeleton.pick import PointPicker
from skeleton.track import ObjectTracker

import numpy as np

from nptyping import NDArray
from typing import Any, List


class DetectionPipeline:
    """

    """

    def __init__(self,
            object_detector: ObjectDetector,
            depth_estimator: DepthEstimator,
            point_picker: PointPicker,
            object_tracker: ObjectTracker,
        ):
        self.object_detector = object_detector
        self.depth_estimator = depth_estimator
        self.point_picker = point_picker
        self.object_tracker = object_tracker

    def __call__(self,
            frame: NDArray[(Any, Any, 3), np.uint8],
            intr
        ) -> List[ObjectDetection]:
        detections = self.object_detector(frame)
        depth_map = self.depth_estimator(frame)
        self.object_tracker(detections)
        self.point_picker(detections, frame, depth_map, intr)
        return detections
