from skeleton.detect import BoundingBox, ID, ObjectDetection

import collections

from cv2 import cv2
import numpy as np

from typing import Any, Callable, Dict, Iterator, Iterable, List, Mapping, NamedTuple, Tuple, TypeVar, Set
from nptyping import NDArray

Centroid = NDArray[2, np.float32]

def centroid(bbox: BoundingBox) -> Centroid:
    return np.array([bbox.w / 2.0 + bbox.x, bbox.h / 2.0 + bbox.y], np.float32)

def centroid_distance(c1: Centroid, c2: Centroid) -> float:
    return (c2[1] - c1[1]) ** 2.0 + (c2[0] - c1[0]) ** 2.0

def pos_kalman_filter(p_noise_cov_scale: float = 0.03) -> cv2.KalmanFilter:
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

    def __call__(self, detections: List[ObjectDetection]) -> None:
        """

        """
        pass

class CentroidTracker:
    class ObjectInstance:
        def __init__(self,
                kfilter: cv2.KalmanFilter,
                detection: ObjectDetection = None,
                age: int = 0,
            ):
            self.kfilter = kfilter
            self.detection = detection
            self.age = age


        def correct(self, measurement: Centroid) -> None:
            self.kfilter.correct(np.reshape(measurement, (2, 1)))
        
        def predict(self) -> Centroid:
            prediction = self.kfilter.predict()
            return np.reshape(prediction, (2, 1))

    def __init__(self,
            on_screen: Set[ID] = set(),
            off_screen: Set[ID] = set(),
            id_itr: Iterator[ID] = iter(int, 1),
            dist_f: Callable[[Centroid, Centroid], float] = centroid_distance,
            kfilter_factory: Callable[[], cv2.KalmanFilter] = pos_kalman_filter,
            pruning_age: int = 50,
            dist_thresh: float = 100.0,
        ):
        self.on_screen = on_screen
        self.off_screen = off_screen
        self.id_itr = id_itr
        self.dist_f = dist_f
        self.obj_instance = collections.defaultdict(lambda: CentroidTracker.ObjectInstance(kfilter_factory()))
        self.pruning_age = pruning_age
        self.dist_thresh = dist_thresh
    
    def __call__(self, detections: List[ObjectDetection]) -> None:
        c_pred = {o_id: o_inst.predict() for o_id, o_inst in self.obj_instance.items()}        
        def closest_detection(o_ids: Iterable[ID]) -> Tuple[ID, ObjectDetection]:
            closest_o_id, closest_det = None, None
            min_dist = self.dist_thresh
            for o_id in o_ids:
                o_id_class = self.obj_instance[o_id].obj_class
                valid_match = lambda x: x.id == None and x.obj_class == o_id_class
                for det in filter(valid_match, detections):
                    alt_dist = self.dist_f(c_pred[o_id], centroid(det.bbox))
                    if alt_dist < min_dist:
                        closest_o_id, closest_det = o_id, det
                        min_dist = alt_dist
            return closest_o_id, closest_det
        def assign_id(o_id: ID, det: ObjectDetection) -> None:
            o_inst = self.obj_instance[o_id]
            o_inst.correct(centroid(det.bbox))
            o_inst.detection = det
            o_inst.age = 0
            det.id = o_id
        while self.on_screen:
            o_id, det = closest_detection(self.on_screen)
            if o_id == None:
                break
            else:
                assign_id(o_id, det)
                self.on_screen.remove(o_id)
        while self.off_screen:
            o_id, det = closest_detection(self.off_screen)
            if o_id == None:
                break
            else:
                assign_id(o_id, det)
                self.off_screen.remove(o_id)
        for o_id, o_inst in self.obj_instance.items():
            if o_inst.age > self.pruning_age:
                del self.obj_instance[o_id]
            else:
                o_inst.age += 1
        self.on_screen = {det.id for det in detections}
        self.off_screen = self.obj_instance.keys() - self.on_screen
