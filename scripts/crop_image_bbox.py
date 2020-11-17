import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image
import os
import glob
import argparse
import sys
import re
import cv2

objects = []
frames = []
#TODO: Create frame object and load with all bounding box objects for each frame
class frame:
    def __init__(self, bboxes):
        self.bboxes = bboxes

class bbox:
    def __init__(self,x_min,y_min,x_max,y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

#This function automatically corrects the image and performs edge detection
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def define_image_masks(image, frame):
    img = cv2.imread(image)
    f = frame

def region_of_interest(image, bbox):
    height = image.shape[0]
    polygons = np.array([[bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask) #

    return masked_image
        
#this function calculates bounding box pixel values 
def calculate_bboxes():
    frame_boxes = []
    current_frame_num = 0
    for obj in objects:
        frame_num = obj[0]
        bbox_x = obj[3]
        bbox_y = obj[4]

        bbox_width  = obj[5] - obj[3]
        bbox_height = obj[6] - obj[4]
        
        x_min = bbox_x - bbox_width  / 2.0
        y_min = bbox_y - bbox_height / 2.0
        x_max = bbox_x + bbox_width  / 2.0
        y_max = bbox_y + bbox_height / 2.0

        #create new bounding box object
        new_bbox = bbox(x_min,y_min,x_max, y_max)

        #if the current frame matches the new frame append a new bbox object to the list
        #for that specific frame else create a new frame object and pass all bboxes for that frame and clear
        if frame_num == current_frame_num:
            frame_boxes.append(new_bbox)
        else:
            f = frame(frame_boxes)
            frames.append(f)
            frame_boxes.clear()
            frame_boxes.append(new_bbox)
        
        #update frame num
        current_frame_num = frame_num
    

#this function parses a txt file with bounding box information 
def parse_input(bbox_input):
    bbox_data = open(bbox_input, 'r')

    #objects = []
    line = bbox_data.readline()
    line.rstrip()

    while line:
        # build a tuple from the line with attributes
        regex = re.compile("[^, \n]+")
        obj = re.findall(regex, line)

        # convert data to floats
        for i in range(0, len(obj)):
            obj[i] = float(obj[i])
        
        objects.append(obj)

         # read a new line
        line = bbox_data.readline()
        line.rstrip()  
    
    #return objects



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument('-i', '--input',  type=str,
    #                         default=None,  help='path of image to crop')
    # arg_parser.add_argument('-o', '--output', type=str, default=None,
    #                         help='path to folder for cropped images')
    arg_parser.add_argument('-t', '--bbox_file',   type=str, default=None,
                            help='input text file of bbox data (<frame>, <class_id>, <conf>, <x_min>, <y_min>, <x_max>, <y_max>)')

    # ns = namespace
    ns, args = arg_parser.parse_known_args(sys.argv)
    # cap = cv2.VideoCapture(ns.input)
    # if(not os.path.exists(ns.output)):
        # os.makedirs(ns.output)
    # SetFrameNumber(cap, ns.lower)
    # SplitCapture(cap, ns.output, ns.format, ns.upper, ns.period)
    # cap.release()

    if ns.bbox_file:
        parse_input(ns.bbox_file)
        calculate_bboxes()

        
    
