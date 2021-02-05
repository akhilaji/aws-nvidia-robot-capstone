import time
import sys
import cv2

from kafka import KafkaProducer
from kafka.errors import KafkaError

producer = KafkaProducer(bootstrap_servers='localhost:9092')
topic = 'vidInput'


def emit_video(path_to_video):
    print('start')

    video = cv2.VideoCapture(path_to_video)
    video.set(cv2.CAP_PROP_AUTOFOCUS, 1) # enable autofocus
    video.set(3, 608) # horizontal resolution
    video.set(4, 608) # vertical resolution

    frame_num = 0
    while video.isOpened():
        success, frame = video.read()
        frame_num += 1
        if not success:
            break

        #resize to match object detection and depth detection spec
        #frame = cv2.resize(frame, dsize = (608,608))

        # png might be too large to emit
        data = cv2.imencode('.jpeg', frame)[1].tobytes()

        future = producer.send(topic, data)
        try:
            future.get(timeout=10)
        except KafkaError as e:
            print(e)
            break

        print('.', end='', flush=True)

emit_video(0)

# zero is for open webcam or usb webcam
# can play a video just add video file in emit_video function
# rtsp camera stream add rtsp feed in emit_video function