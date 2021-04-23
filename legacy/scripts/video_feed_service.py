import os
import sys
import datetime
import daemon

import cv2
import numpy as np

import psutil

#Video Settings
framerate = 30
video_src = 0
time_sec = 60
resolution_width = 1280
resolution_height = 720
capture_size = (resolution_width, resolution_height)
dim = (resolution_width, resolution_height)

#Tools
destination_folder_path = '../video/'
clip_time = time_sec * framerate
current_time = ''
output_file_name = ''
diagnostics = False
pid = os.getpid()
py = psutil.Process(pid)

#OpenCv Capture Settings
cap = cv2.VideoCapture(video_src)
capture_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out =cv2.VideoWriter("../video/default.avi",fourcc, 30.0, capture_size)



def get_new_out():
    current_time = datetime.datetime.now().replace(microsecond=0)
    output_file_name = destination_folder_path + str(current_time) + '.avi'
    output_file_name  = output_file_name.replace(':', '_')
    print(output_file_name)
    out = cv2.VideoWriter(output_file_name,fourcc, 30.0, capture_size)
    return out

out = get_new_out()
frame_num = 1
while(cap.isOpened()):
    #capture frames
    ret, frame = cap.read()

    if ((diagnostics == True) and (frame_num %  5 == 0)):
        print(py.cpu_percent())
        print(py.memory_info()[0]/2.**30)
    if ret==True:
        if(frame_num <= clip_time):
            out.write(frame)
            frame_num += 1
        else:
            cap.release()
            out.release()
            cap = cv2.VideoCapture(video_src)
            out = get_new_out()
            frame_num = 1
            out.write(frame)
            frame_num += 1
