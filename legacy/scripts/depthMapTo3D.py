import argparse
import os
import sys
import numpy as np
import cv2
import math

def create3DCoord(depthPath, outputPath):
    img = cv2.imread(depthPath, -1).astype(np.float32) / 1000.0
    w = img.shape[1]
    h = img.shape[0]
    FOV = math.pi/4
    D = (img.shape[0]/2)/math.tan(FOV/2)
    with open(outputPath,"w") as f:    
        ids = np.zeros((img.shape[1], img.shape[0]), int)
        vid = 1
        for u in range(0, w):
            for v in range(h-1, -1, -1):
                d = img[v, u]
                ids[u,v] = vid
                if d == 0.0:
                    ids[u,v] = 0
                vid += 1
                x = u - w/2
                y = v - h/2
                z = -D
                norm = 1 / math.sqrt(x*x + y*y + z*z)
                t = d/(z*norm)
                x = -t*x*norm
                y = t*y*norm
                z = -t*z*norm        
                f.write(str(x) + " " + str(y) + " " + str(z) + "\n")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input',  type = str, default = None,  help = 'path to the input frame')
    arg_parser.add_argument('-o', '--output', type = str, default = None,  help = 'path to the output file')
    
    ns, args = arg_parser.parse_known_args(sys.argv)
    if(not os.path.exists(ns.output)):
        os.makedirs(ns.output)
    create3DCoord(ns.input,ns.output)