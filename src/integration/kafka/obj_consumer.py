import cv2
import numpy as np
from kafka import KafkaConsumer

from matplotlib import image, patches, pyplot as plt
import tensorflow as tensorflow
from darknetpy.detector import Detector

physical_devices = tf.config.experimental.list_physical_devices('GPU')

detector = Detector(
    '/home/akhil/darknet/cfg/obj.data',
    '/home/akhil/darknet/cfg/yolov4-obj.cfg',
    '/home/akhil/darknet/yolov4-obj_best.weights'
)

obj_consumer = KafkaConsumer('vidInput',
                             group_id='detection-group',
                             bootstrap_servers=['localhost:9092'])

for msg in obj_consumer:
    # message values are in raw bytes
    # e.g. for unicode: `msg.value.decode('utf-8')`

    #print("%s:%d%d: key=%s value=%s" %(msg.topic, msg.partition,
    #                                   msg.offset, msg.key,
    #                                   msg.value))

    nparr = np.fromstring(msg.value, np.uint8)
    image_np = cv2.imdecode(nparr, 1)

    # save the image
    #img = cv2.imdecode(msg.value, 1)
    cv2.imwrite("objimage.jpeg", image_np)
    boxes = detector.detect('./objimage.jpeg')
    print(boxes)
