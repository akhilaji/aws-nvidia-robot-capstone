#inputvideo does not work for MOV files must be MP4 or image extraction will fail
#conversion or workaround must be found to use recorded MOV files from the ASU campus

import cv2
import os
import sys

def ExtractImages(inputvideo, outputdir, extension, frameperiod):
    if(not os.path.exists(outputdir)):
        os.makedirs(outputdir)
    cap = cv2.VideoCapture(inputvideo)
    count = 0
    while(cap.grab()):
        ret, frame = cap.retrieve()
        if(ret and (count % frameperiod) == 0):
            filename = os.path.join(outputdir, str(count) + "." + extension)
            if(os.path.exists(filename)):
                os.remove(filename)
            cv2.imwrite(filename, frame)
        count += 1
    cap.release()
