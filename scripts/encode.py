import argparse
import os
import sys
from cv2 import cv2

if(__name__ == "__main__"):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input',  type = str,   default = None,    help = 'path of video to split')
    arg_parser.add_argument('-o', '--output', type = str,   default = None,    help = 'path to folder for extracted frames')
    arg_parser.add_argument('-c', '--codec',  type = str,   default = None,    help = 'video compression codec to use')
    arg_parser.add_argument(      '--width',  type = int,   default = None,    help = 'frame width')
    arg_parser.add_argument(      '--height', type = int,   default = None,    help = 'frame height')
    arg_parser.add_argument('-f', '--fps',    type = float, default = 24.9997, help = 'frames per second')

    ns, args = arg_parser.parse_known_args(sys.argv)
    frames = sorted(os.listdir(ns.input))
    fourcc = cv2.VideoWriter_fourcc(*ns.codec)
    if(ns.width == None or ns.height == None):
        img = cv2.imread(os.path.join(ns.input, frames[0]))
        if(ns.width == None):
            ns.width = len(img[0])
        if(ns.height == None):
            ns.height = len(img)
    encoder = cv2.VideoWriter(ns.output, fourcc, ns.fps, (ns.width, ns.height))
    for frame in frames:
        path = os.path.join(ns.input, frame)
        read = cv2.imread(path)
        encoder.write(read)
    encoder.release()
