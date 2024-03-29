import cv2
import numpy as np
from kafka import KafkaConsumer
from kafka import KafkaProducer
from kafka.errors import KafkaError
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

#consumer declaration to read image data as bytes from vidInput topic
obj_consumer = KafkaConsumer('vidInput',
                             group_id='detection-group',
                             bootstrap_servers=['localhost:9092'])

#declare a producer to pipe detection and bounding box data to objOutput topic
topic = 'objOutput'
obj_result_producer = KafkaProducer(bootstrap_servers=['localhost:9092'])


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = 608
    images = FLAGS.images


    #load model
    model_path = "./yolov4-608"
    saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
 
    count = 1

    for msg in obj_consumer:
        if count > 100000:
            count = 1
        else:
            count += 1
        # message values are in raw bytes
        # e.g. for unicode: `msg.value.decode('utf-8')`

        #print("%s:%d%d: key=%s value=%s" %(msg.topic, msg.partition,
        #                                   msg.offset, msg.key,
        #                                   msg.value))
        timestamp = msg.timestamp
        print(timestamp)
        nparr = np.fromstring(msg.value, np.uint8)
        image_np = cv2.imdecode(nparr, 1)

        original_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        image_data = image_np
        image_data = image_data / 255
        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)




        infer = saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        #print(pred_bbox)

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to allow detections for only people)
        #allowed_classes = ['person']

        image = utils.draw_bbox(original_image, pred_bbox, allowed_classes = allowed_classes)

        #image = Image.fromarray(image.astype(np.uint8))
        #if not FLAGS.dont_show:
        #    image.show()
        #image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        #cv2.imwrite(FLAGS.output + 'detection' + str(count) + '.png', image)
        # save the image
        #img = cv2.imdecode(msg.value, 1)
        #cv2.imwrite("objimage.jpeg", image_np)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
