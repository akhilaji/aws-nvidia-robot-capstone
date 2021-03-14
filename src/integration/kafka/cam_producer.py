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

    while video.isOpened():
        success, frame = video.read()
        frame = cv2.resize(frame, (608, 608))
        
        if not success:
            break

        ret, data = cv2.imencode('.jpg', frame)
        print(frame.shape)
        future = cam_producer.send(topic, data.tobytes())
        try:
            future.get(timeout=10)
        except KafkaError as e:
            print(e)

    print('.', end='', flush=True)

def main():
    if len(sys.argv) == 2:
        emit_video(sys.argv[1])
    else:
        emit_video(0)

if __name__ == "__main__":
    main()
