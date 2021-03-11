import detect

import cv2

def draw_detection(rgb_img, detection: detect.ObjectDetection, color, font):
    bbox, obj_class = detection
    cv2.rectangle(rgb_img, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), color)
    pass

print('Hello World!')
