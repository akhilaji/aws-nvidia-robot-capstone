"""

"""

import numbers

from cv2 import cv2
import numpy as np
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from absl import app, flags, logging
from absl.flags import FLAGS

from core import utils
from core.config import cfg
from core.yolov4 import filter_boxes

from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from typing import Any, Dict, Iterator, List, NamedTuple, Set, Tuple, TypeVar
from nptyping import NDArray

ID = TypeVar('ID')

class BoundingBox(NamedTuple):
    """
    NamedTuple class definition for a bounding box inside of an image.

    Attributes:
        x (int): the x-coordinate position of the bounding box
        y (int): the y-coordinate position of the bounding box
        w (int): the width of the bounding box
        h (int); the height of the bounding box
    """
    x: int
    y: int
    w: int
    h: int

class ObjectDetection:
    """
    Class data object definition for detected object instances.

    Attributes:
        bbox (BoundingBox): NamedTuple class definition for bounding box
            information. Contains attributes x, y, w, h in that order.
        obj_class (str): The assigned object classification of this detection.
        id (ID): The assigned id of this detection.
        prob (float): The probability of the object detection.
        pt (NDArray[3, float]):
    """
    def __init__(self,
            id: ID,
            bbox: BoundingBox,
            obj_class: int,
            prob: float,
            pt: NDArray[3, float]
        ):
        self.id = id
        self.bbox = bbox
        self.obj_class = obj_class
        self.prob = prob
        self.pt = pt
    
    def __str__(self) -> str:
        return 'id=%r\nbbox=%r\nobj_class=%r\nprob=%r\npt=%r' % (self.id, self.bbox, self.obj_class, self.prob, self.pt)
        # return 'id=%r' % (self.id)
    
    def __repr__(self) -> str:
        return self.__str__()

class ObjectDetector:
    """
    Abstract callable object to run object detection on an RGB image.
    Impelementation details to be specified by child.
    """

    def __call__(self, frame) -> List[ObjectDetection]:
        """
        Runs object detection on a numpy 3-channel RGB image and formats the
        results as a list of ObjectDection containing the BoundingBox information
        and the detected object class.
        """
        pass

class YOLOv4ObjectDetector(ObjectDetector):
    def __init__(self,
                 model,
                 session,
                 input_dim: Tuple[int, int],
                 max_output_size_per_class: int = 50,
                 max_total_size: int = 50,
                 iou_threshold: float = 0.45,
                 score_threshold: float = 0.25,
        ):

        self.model = model
        self.session = session
        self.input_dim = input_dim
        self.max_output_size_per_class = max_output_size_per_class
        self.max_total_size = max_total_size
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
    
    def __call__(self, frame: NDArray[(Any, Any, 3), np.float32]) -> List[ObjectDetection]:
        resized_frame = cv2.resize(frame, self.input_dim)
        batch_data = tf.constant(np.array([resized_frame / 255], np.float32))
        model_signature = self.model.signatures['serving_default']
        pred_bbox = model_signature(batch_data)

        for value in pred_bbox.values():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=self.max_output_size_per_class,
            max_total_size=self.max_total_size,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
        )

        input_w, input_h = self.input_dim
        x_ratio = input_w
        y_ratio = input_h

        return [
            ObjectDetection(
                id=None,
                bbox=convert_bbox(boxes[0][i], x_ratio, y_ratio),
                obj_class=int(classes[0][i].numpy()),
                prob=scores[0][i].numpy(),
                pt=None,
            )
            for i in range(valid_detections[0])
        ]

def convert_bbox(boxes: tf.Tensor, x_ratio: int, y_ratio: float) -> BoundingBox:
    coor = boxes.numpy()

    return BoundingBox(
        x=int(coor[1] * x_ratio),
        y=int(coor[0] * y_ratio),
        w=int(coor[3] * x_ratio),
        h=int(coor[2] * y_ratio),
    )

def construct_yolov4_object_detector(
        model_path: str,
        input_dim: Tuple[int, int],
        max_output_size_per_class: int = 50,
        max_total_size: int = 50,
        iou_threshold: float = 0.45,
        score_threshold: float = 0.25
    ) -> YOLOv4ObjectDetector:
    model = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    return YOLOv4ObjectDetector(
        model,
        session,
        input_dim,
        max_output_size_per_class=max_output_size_per_class,
        max_total_size=max_total_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
