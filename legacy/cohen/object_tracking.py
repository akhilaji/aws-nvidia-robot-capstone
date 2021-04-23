from typing import Any, List, NamedTuple, Tuple

import cv2
import numpy as np
from nptyping import NDArray

import util
from object_detection import BoundingBox

class ObjectDetection(NamedTuple):
    rgb: NDArray[(Any, Any, 3), np.uint8]
    pos: NDArray[2, np.float32]
    bbox: BoundingBox
    obj_class: str

class ObjectTracker:
    def __init__(self,
            edge_filter: None=None
        ) -> None:
        self.edge_filter = edge_filter

        self.obj_id_counter = 0
        self.frame_id_counter = 0
        self.in_frame = dict()
        self.in_frame.setdefault(set())
        self.objects = dict()
        self.objects.setdefault(dict())

    def centroid(self, bbox: BoundingBox) -> NDArray[2, np.float32]:
        """
        Gets the centroid of a BoundingBox. Note: a centroid is simply the
        center point of a BoundingBox.

        Args:
            bbox (BoundingBox): The BoundingBox to get the center point of.
                BoundingBox is a NamedTuple of 4 ints: x, y, w, h.
        
        Returns:
            NDArray[2, np.float32]: The centroid or center point of the given
                BoundingBox as a numpy array of floats.
        """
        return np.array([bbox.x + bbox.w / 2.0, bbox.y + bbox.h / 2.0], np.float32)

    def closest_centroid(self,
            frame_id: int, curr_centroid: NDArray[2, np.float32]
        ) -> int:
        """
        Gets the object id of the object with centroid closest to the given
        centroid within a specified frame.

        Args:
            frame_id (int): The frame to look for centroids in.

            curr_centroid (NDArray[2, np.float32]): The centroid to find the
                closest of.
        
        Returns:
            int: The object id with the closest centroid in the specified frame.
        """
        closest_dist, closest_id = float('inf'), None
        for obj_id in self.in_frame[frame_id]:
            diff = curr_centroid - self.centroid(self.objects[obj_id][frame_id].bbox)
            dist = np.linalg.dot(diff, diff)
            if dist < closest_dist:
                closest_dist = dist
                closest_id = obj_id
        return closest_id

    def add_frame(self,
            rgb_img: NDArray[(Any, Any, 3), np.uint8],
            obj_detections: List[ObjectDetection]
        ) -> None:
        if self.frame_id_counter - 1 not in self.in_frame:
            for obj_detection in obj_detections:
                self.objects[self.obj_id_counter][self.frame_id_counter] = obj_detection
                self.in_frame[self.frame_id_counter].add(self.obj_id_counter)
                self.obj_id_counter += 1
        elif len(obj_detections) > len(self.in_frame[self.frame_id_counter - 1]):
            pass
        elif len(obj_detections) < len(self.in_frame[self.frame_id_counter - 1]):
            pass
        else:
            pass
        self.frame_id_counter += 1

    def predict_centroid(self,
            obj_id: int
        ) -> Tuple[float, float]:
        pass

def run_tracking(
        cap: cv2.VideoCapture,
        obj_detector, obj_threshold: callable,
        contour_finder: callable, contour_filter: callable,
        esc_key: int=27
    ):
    key = int(esc_key == 0)
    while key != esc_key:
        ret, frame = cap.read()
        detection = obj_detector.apply(frame)
        detection = obj_threshold(detection)
        contours = contour_finder(detection)
        contours = filter(contour_filter, contours)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.drawContours(frame, [cnt], -1, (255, 255, 0), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        cv2.imshow('frame', frame)
        cv2.imshow('detection', detection)
        key = cv2.waitKey(30) & 0xFF
    pass

#cap = cv2.VideoCapture('C:\\Users\\Jacob\\Documents\\Code\\School\\CSE486\\research\\videos\\FroggerHighway.mp4')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
obj_detector = cv2.createBackgroundSubtractorMOG2(varThreshold=50)
obj_threshold = lambda img : cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)[1]
contour_finder = lambda img : cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
contour_filter = lambda cntr : cv2.contourArea(cntr) > 400
run_tracking(cap, obj_detector, obj_threshold, contour_finder, contour_filter)

cv2.track

cap.release()
cv2.destroyAllWindows()