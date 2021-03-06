import cv2
import numpy as np
from kafka import KafkaConsumer
import depth
dep_consumer = KafkaConsumer('vidInput',
                             group_id='depth-group',
                             bootstrap_servers=['localhost:9092'])

for msg in dep_consumer:
    """
    The *msg* has the following attributes:
        - topic: The name of the topic subscribed to.
    	- partition: The topic partition of the consumer.
    	- offset: The message offset.
    	- key: The raw message key.
    	- value: The raw message value.
    """
    
    # decode message data (bytes -> numpy.array)
    nparr = np.frombuffer(msg.value, np.uint8)
    nparr_img = cv2.imdecode(nparr, 1)

    # reshape the image array
    data = cv2.resize(nparr_img, (480,640))

    # perform depth detection
    depth_map = depth.midas_large(data)
    print(depth_map)
