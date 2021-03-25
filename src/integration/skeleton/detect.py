"""

"""

from cv2 import cv2
import numpy as np
import tensorflow as tf

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

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
        return 'id=%r,bbox=%r,obj_class=%r,prob=%r,pt=%r' % (self.id, self.bbox, self.obj_class, self.prob, self.pt)
    
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
        frame = cv2.resize(frame, self.input_dim)
        batch_data = tf.constant(np.array([frame], np.float32))
        serving_default = self.model.signatures['serving_default']
        pred_bbox = serving_default(batch_data)
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
        return [
            ObjectDetection(
                id=None,
                bbox=BoundingBox(*boxes[0][i]),
                obj_class=classes[0][i],
                prob=scores[0][i],
                pt=None,
            )
            for i in range(valid_detections[0])
        ]

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

"""
Below contains detect.py before cleaned up.
Perserved incase of malfunction of current cleaned up detect.py
"""

# import numpy as np
# import cv2
# import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# from absl import app, flags, logging
# from absl.flags import FLAGS
# import core.utils as utils
# from core.config import cfg
# from core.yolov4 import filter_boxes
# from tensorflow.python.saved_model import tag_constants
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# from typing import Any, List, NamedTuple, Tuple
# from nptyping import NDArray
# from typing import Any, Dict, Iterator, List, NamedTuple, Set, Tuple, TypeVar

# ID = TypeVar('ID')

# ID = TypeVar('ID')

# class BoundingBox(NamedTuple):
#     """
#     NamedTuple class definition for a bounding box inside of an image.
#     Attributes:
#         x (int): the x-coordinate position of the bounding box
#         y (int): the y-coordinate position of the bounding box
#         w (int): the width of the bounding box
#         h (int); the height of the bounding box
#     """
#     x: int
#     y: int
#     w: int
#     h: int

# class ObjectDetection:
#     """
#     Class data object definition for detected object instances.
#     Attributes:
#         bbox (BoundingBox): NamedTuple class definition for bounding box
#             information. Contains attributes x, y, w, h in that order.
#         obj_class (str): The assigned object classification of this detection.
#         id (ID): The assigned id of this detection.
#         prob (float): The probability of the object detection.
#         pt (NDArray[3, float]):
#     """
#     def __init__(self, id: ID, bbox: BoundingBox, obj_class: int, prob: float,
#                  pt: NDArray[3, float]):

#         self.id = id
#         self.bbox = bbox
#         self.obj_class = obj_class
#         self.prob = prob
#         self.pt = pt

# class ObjectDetector:
#     """
#     Abstract callable object to run object detection on an RGB image.
#     Impelementation details to be specified by child.
#     """
#     def __init__(self, model_path , saved_model_loaded, config, session, input_size, images):
#         self.model_path = model_path
#         self.saved_model_loaded = saved_model_loaded
#         self.config = config
#         self.config.gpu_options.allow_growth = True
#         self.session = session
#         STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
#         self.input_size = input_size
#         self.images = images

        
#     def __call__(self, frame) -> List[ObjectDetection]:
#         """
#         Runs object detection on a numpy 3-channel RGB image and formats the
#         results as a list of ObjectDection containing the BoundingBox information
#         and the detected object class.
#         """
#         print("object detection function")
#         #load frame
#         data = frame
#         # detection_properties
#         iou = 0.45
#         score = 0.25

#         original_image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

#         image_data = data
#         image_data = image_data / 255
#         images_data = []
#         for i in range(1):
#             images_data.append(image_data)
#         images_data = np.asarray(images_data).astype(np.float32)

#         infer = self.saved_model_loaded.signatures['serving_default']
#         batch_data = tf.constant(images_data)
#         pred_bbox = infer(batch_data)
#         for key, value in pred_bbox.items():
#             boxes = value[:, :, 0:4]
#             pred_conf = value[:, :, 4:]
#         boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
#             boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
#             scores=tf.reshape(
#                 pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
#             max_output_size_per_class=50,
#             max_total_size=50,
#             iou_threshold=iou,
#             score_threshold=score
#         )
#         # read in all class names from config
#         class_names = utils.read_class_names(cfg.YOLO.CLASSES)

#         # by default allow all classes in .names file
#         allowed_classes = list(class_names.values())

#         detections = []
#         for i in range(valid_detections[0]):
#             bbox = BoundingBox(boxes[0][i][0],
#                             boxes[0][i][1],
#                             boxes[0][i][2],
#                             boxes[0][i][3])

#             obj = ObjectDetection(id=-1,
#                                 bbox=bbox,
#                                 obj_class=classes[0][i],
#                                 prob=scores[0][i],
#                                 pt=[-1,-1,-1])

#             detections.append(obj)


#         return detections
    
# def load_object_detector()-> ObjectDetector:
#     flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
#     flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
#     flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')
#     flags.DEFINE_string('output', './detections/', 'path to output folder')
#     # load model
#     model_path = "./yolov4-608"
#     saved_model_loaded = tf.saved_model.load(
#             model_path, tags=[tag_constants.SERVING])
#     #tensorflow loading
#     config = ConfigProto()
#     config.gpu_options.allow_growth = True
#     session = InteractiveSession(config=config)
#     STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
#     input_size = 608
#     images = FLAGS.images
#     return ObjectDetector(model_path, saved_model_loaded, config, session, input_size, images)
