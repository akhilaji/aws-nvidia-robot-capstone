from skeleton import detect

from cv2 import cv2
import numpy as np

from typing import Any, Callable, Dict, Iterator, List, Mapping, NamedTuple, TypeVar, Set
from nptyping import NDArray

Centroid = NDArray[(2, 1), np.float32]

def centroid(bbox: detect.BoundingBox) -> Centroid:
    return np.array([[bbox.w / 2.0 + bbox.x], [bbox.h / 2.0 + bbox.y]], np.float32)

def centroid_distance(c1: Centroid, c2: Centroid) -> float:
    return (c2[1][0] - c1[1][0]) ** 2.0 + (c2[0][0] - c1[0][0]) ** 2.0

def kalman_filter_factory(p_noise_cov_scale: float = 0.03) -> cv2.KalmanFilter:
    k_filter = cv2.KalmanFilter(4, 2)
    k_filter.measurementMatrix  = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0]], np.float32)

    k_filter.transitionMatrix   = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)

    k_filter.processNoiseCov    = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32) * p_noise_cov_scale
    return k_filter

class ObjectTracker:
    """

    """

    def track(self, detections: List[detect.ObjectDetection]) -> None:
        """

        """
        pass

class CentroidTracker:
    class ObjectInstance(NamedTuple):
        c_filter: cv2.KalmanFilter
        detection: detect.ObjectDetection
        age: int

    def __init__(self,
            on_screen: Set[Any] = set(),
            obj_instance: Dict[Any, ObjectInstance] = dict(),
            id_itr: Iterator[Any] = iter(int, 1),
            pruning_age: int = 50,
            dist_thresh: float = 100.0,
            filter_factory: Callable[[], cv2.KalmanFilter] = kalman_filter_factory,
        ):
        self.on_screen = on_screen
        self.obj_instance = obj_instance
        self.id_itr = id_itr
        self.pruning_age = pruning_age
        self.dist_thresh = dist_thresh
        self.filter_factory = filter_factory

    def __call__(self, detections: List[detect.ObjectDetection]) -> None:
        on_screen_predictions = [(o_id, self.obj_instance[o_id].c_filter.predict()) for o_id in self.on_screen]
        enumerated_centroids = [(index, centroid(detection.bbox)) for index, detection in enumerate(detections)]
        while on_screen_predictions:
            min_pair, min_dist = (None, None), float('inf')
            for o_id, o_prediction in on_screen_predictions:
                for ec_index, ec_centroid in enumerated_centroids:
                    if detections[ec_index].id == None and \
                       detections[ec_index].obj_class == self.obj_instance[o_id].detection.obj_class and \
                       (alt_dist := centroid_distance(o_prediction, ec_centroid)) < min_dist:
                        min_dist = alt_dist
                        min_pair = (o_id, ec_index)
            o_id, ec_index = min_pair
            if o_id != None and ec_index != None and min_dist < self.dist_thresh:
                detections[ec_index].id = o_id
                del on_screen_predictions[o_id]
        self.on_screen.clear()
        for detection in detections:
            if detection.id == None:
                detection.id = next(self.id_itr)
                self.obj_instance[detection.id] = CentroidTracker.ObjectInstance(self.filter_factory(), detection, 0)
            self.obj_instance[detection.id].age = 0
            self.on_screen.add(detection.id)
        for o_id, o_instance in self.obj_instance.items():
            if o_instance.age > self.pruning_age:
                del self.obj_instance[o_id]
            else:
                o_instance.age += 1
