import cv2
import numpy as np
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.config import cfg
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def main(_argv):
    flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
    flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
    flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')
    flags.DEFINE_string('output', './detections/', 'path to output folder')
    flags.DEFINE_float('iou', 0.45, 'iou threshold')
    flags.DEFINE_float('score', 0.25, 'score threshold')
    flags.DEFINE_boolean('dont_show', False, 'dont show image output')

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = 608
    images = FLAGS.images

    #load model
    model_path = "./yolov4-608"
    saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
