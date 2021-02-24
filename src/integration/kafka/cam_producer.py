import cv2
import sys

from kafka import KafkaProducer
from kafka.errors import KafkaError

topic = 'vidInput'
cam_producer = KafkaProducer(bootstrap_servers=['localhost:9092'])

def emit_video(video_feed):
    video = cv2.VideoCapture(video_feed)

    # set video properties
    video.set(cv2.CAP_PROP_AUTOFOCUS, 1) # enable external camera autofocus
    video.set(3, 608) # horizontal resolution
    video.set(4, 608) # vertical resolution

    frame = 0
    while video.isOpened():
        success, frame = video.read()
        frame += 1

        if not success:
            break

        ret, data = cv2.imencode('.jpg', frame)

        future = cam_producer.send(topic, data.tobytes())
        try:
            future.get(timeout=10)
        except KafkaError as e:
            print(e)

    print('.', end='', flush=True)

def main():
    if len(sys.argv) > 2:
        for video in sys.argv:
            emit_video(video)
    else:
        emit_video(0)

if __name__ == "__main__":
    main()
