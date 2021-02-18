from nptyping import NDArray
from typing import Any, List, Tuple

import cv2
import numpy as np

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

    def add_frame(self,
            rgb_frame: NDArray[(Any, Any, 3), np.uint8],
            obj_detections: List[Tuple[str, Tuple[int, int, int, int], Tuple[float, float, float]]]
        ) -> None:
        for obj_class, bbox, pt in obj_detections:
            match = self.find_match(self.frame_id_counter, obj_class, bbox, pt)
            if match == self.obj_id_counter:
                self.obj_id_counter += 1
            self.objects[match][self.frame_id_counter] = (obj_class, bbox, pt)
            self.in_frame[self.frame_id_counter].add(match)
        self.frame_id_counter += 1

    def find_match(self,
            frame_id: int,
            obj_class: str,
            bbox: Tuple[int, int, int, int],
            pt: Tuple[float, float, float]
        ) -> int:
        return self.obj_id_counter

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