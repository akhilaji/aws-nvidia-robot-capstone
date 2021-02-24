import cv2
import numpy as np
from kafka import KafkaConsumer

dep_consumer = KafkaConsumer('vidInput',
                             group_id='depth-group',
                             bootstrap_servers=['localhost:9092'])

for msg in dep_consumer:
    # message values are in raw bytes
    # e.g. for unicode: `msg.value.decode('utf-8')`

    print("%s:%d%d: key=%s value=%s" %(msg.topic, msg.partition,
                                       msg.offset, msg.key,
                                       msg.value))

    # decode message data (bytes -> jpg)
    nparr = np.fromstring(msg.value, np.uint8)
    image_np = cv2.imdecode(nparr, 1)

    cv2.imwrite("image.jpg", image_np)
