import argparse
import os
import sys
from cv2 import cv2

def SetFrameNumber(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

def GetFrameNumber(cap):
    return cap.get(cv2.CAP_PROP_POS_FRAMES)

def SplitCapture(cap, output, format, upper, period):
    i = GetFrameNumber(cap)
    while(i < upper and cap.grab()):
        res, frame = cap.read()
        if(res):
            cv2.imwrite(os.path.join(output, str(int(i)) + '.' + format), frame)
        SetFrameNumber(cap, i + period)
        i += period

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

    arg_parser.add_argument('-l', '--lower',
        type = int,
        default = 0,
        help = 'lower bound on frame number extraction')

    arg_parser.add_argument('-u', '--upper',
        default = float('inf'),
        help = 'upper bound on frame number extraction')

    arg_parser.add_argument('-p', '--period',
        type = int,
        default = 1,
        help = 'extracts a frame after every period steps')

    arg_parser.add_argument('-f', '--format',
        type = str,
        default = 'png',
        help = 'the file format to save extracted frames as')

    ns, args = arg_parser.parse_known_args(sys.argv)
    cap = cv2.VideoCapture(ns.input)
    if(not os.path.exists(ns.output)):
        os.makedirs(ns.output)
    SetFrameNumber(cap, ns.lower)
    SplitCapture(cap, ns.output, ns.format, ns.upper, ns.period)
    cap.release()
