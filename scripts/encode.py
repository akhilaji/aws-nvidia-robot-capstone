import argparse
import os
import sys
from cv2 import cv2

def EncodeVideo(encoder, frames):
    for frame in frames:
        read = cv2.imread(frame)
        encoder.write(read)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-i', '--input',
        type = str,
        default = None,
        help = 'path of video to split')

    arg_parser.add_argument('-o', '--output',
        type = str,
        default = None,
        help = 'path to folder for extracted frames')

    arg_parser.add_argument('-c', '--codec',
        type = str,
        default = 'MJPG',
        help = 'video compression codec to use')

    ns, args = arg_parser.parse_known_args(sys.argv)
    codec = cv2.VideoWriter_fourcc(*'MJPG')
    encoder = cv2.VideoWriter(ns.output, codec, 10, (1280, 720))
    frames = sorted(os.listdir(ns.input))
    EncodeVideo(encoder, frames)
    encoder.release()
