from skeleton.calibration import Camera
from skeleton.detect import ObjectDetection

from cv2 import cv2
import numpy as np

from nptyping import NDArray
from typing import Any, List

class PointPicker:
    """

    """

    def __init__(self, camera: Camera):
        self.camera = camera

    def __call__(self,
            detections: List[ObjectDetection],
            frame: NDArray[(Any, Any, 3), np.uint8],
            depth: NDArray[(Any, Any), float],
        ) -> None:
        """

        """
        pass

class AverageCannyEdgePointPicker(PointPicker):

    def __init__(self,
            camera: Camera,
            upper_thresh: float,
            lower_thresh: float,
            aperture_size: int = 3,
            l2_gradient: bool = False,
        ):
        self.camera = camera
        self.canny = lambda frame: cv2.Canny(
            frame,
            upper_thresh,
            lower_thresh,
            aperture_size,
            l2_gradient,
        )

    def __call__(self,
            detections: List[ObjectDetection],
            frame: NDArray[(Any, Any, 3), np.uint8],
            depth: NDArray[(Any, Any), np.float32],
        ) -> None:
        canny_frame = self.canny(frame)
        for det in detections:
            x, y, w, h = det.bbox
            pt_sum, pt_count = np.array([0, 0, 0], np.float32), 0
            for yp in range(y, y + h):
                for xp in range(x, x + w):
                    if canny_frame[yp][xp]:
                        pt_sum += self.camera.project(xp, yp, depth[yp][xp])
                        pt_count += 1
            det.pt = pt_sum / pt_count if pt_count else pt_sum

class AverageContourPointPicker(PointPicker):

    def __init__(self,
            camera: Camera,
            thresh_lower: int = 127,
            thresh_upper: int = 255,
            thresh_type: int = cv2.THRESH_BINARY,
            contours_mode: int = cv2.RETR_TREE,
            contours_method: int = cv2.CHAIN_APPROX_SIMPLE,
        ):
        self.camera = camera
        self.threshold_f = lambda frame: cv2.threshold(
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
            thresh_lower,
            thresh_upper,
            thresh_type,
        )
        self.contours_f = lambda thresh: cv2.findContours(
            thresh,
            contours_mode,
            contours_method,
        )
    
    def __call__(self,
            detections: List[ObjectDetection],
            frame: NDArray[(Any, Any, 3), np.uint8],
            depth: NDArray[(Any, Any), np.float32],
        ) -> None:
        thresh_frame = self.threshold_f(frame)
        for det in detections:
            x, y, w, h = det.bbox
            thresh_crop = thresh_frame[y:(y + h), x:(x + w)]
            all_contours = self.contours_f(thresh_crop)
            max_area_contours = max(all_contours, key=lambda c: cv2.contourArea(c))
            det.pt = sum([self.camera.project(x, y, depth[x, y]) for x, y in max_area_contours]) / len(max_area_contours) if max_area_contours else np.array([0, 0, 0], np.float32)
