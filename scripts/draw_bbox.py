import argparse
import math
import os
import sys
import cv2
import numpy as np
import re
import PIL
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

classes = ["Bottle", "Chair", "Computer keyboard", "Computer monitor", "Computer mouse", "Corded phone", "Desk", "Headphones", "Laptop", "Microphone",
           "Mobile phone", "Mug", "Office building", "Office supplies", "Pen", "Person", "Stapler", "Table", "Tablet computer", "Telephone", "Whiteboard"]
edited_frame_nums = []


def SetFrameNumber(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)


def GetFrameCount(cap):
    return cap.get(cv2.CAP_PROP_FRAME_COUNT)


def GetFrameNumber(cap):
    return cap.get(cv2.CAP_PROP_POS_FRAMES)


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                  25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        ymin, xmin, ymax, xmax = tuple(boxes[i])
        display_str = "{}: {}%".format(class_names[i],
                                       int(100 * scores[i]))
        color = colors[hash(class_names[i]) % len(colors)]
        image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
        draw_bounding_box_on_image(
            image_pil,
            ymin,
            xmin,
            ymax,
            xmax,
            color,
            font,
            display_str_list=[display_str])
        np.copyto(image, np.array(image_pil))
    return image


t


def SplitCapture(cap, output, file_format, period):
    # cap:
    # output:
    # file_format:
    # upper: last frame number to write to
    # period:
    upper = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fmt = '%0' + str(int(math.log10(GetFrameCount(cap))) + 1) + 'd'
    i = GetFrameNumber(cap)
    while(i < upper and cap.grab()):
        res, frame = cap.read()
        if(res):
            cv2.imwrite(os.path.join(output, fmt %
                                     int(i) + '.' + file_format), frame)
        SetFrameNumber(cap, i + period)
        i += period


def SaveFrame(video, frame):
    cap = cv2.VideoCapture(video)

    # Define the codec and create VideoWriter object
    #fourcc = cv2.cv.CV_FOURCC(*'DIVX')
    #out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    out = cv2.VideoWriter('output.avi', -1, 20.0, (640, 480))

    frame_num = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if(frame in edited_frame_nums):
                frame = edited_frame_nums[frame_num]

            # write the frame
            out.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        frame_num += 1

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def DrawBoundingBox(filename, video, output):
    bbox_data = open(filename, 'r')
    frame = -1

    objects = []
    line = bbox_data.readline()
    line.rstrip()

    while line:
        # build a tuple from the line with attributes
        regex = re.compile("[^, \n]+")
        obj = re.findall(regex, line)

        # convert data to floats
        for i in range(0, len(obj)):
            obj[i] = float(obj[i])

        # check if the object is on the active frame
        if int(obj[0]) == frame:
            # the current object is on the same frame, so
            # we want to add it to our current list of objects in frame

            objects.append(obj)
        else:
            # the current object is not on the frame, so
            # we want to draw all the boxes for the current frame

            # get the current frame corresponding to these objects
            cap = cv2.VideoCapture(video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((frame - 1)))
            res, image = cap.read()

            if res:
                # build the data to draw the objects on the image
                boxes = np.empty((len(objects), 4))
                names = []
                scores = []
                obj_count = len(objects)
                i = 0

                for curr in objects:
                    x_min = curr[3] - (curr[5] - curr[3]) / 2.0
                    y_min = curr[4] - (curr[6] - curr[4]) / 2.0
                    x_max = curr[3] + (curr[5] - curr[3]) / 2.0
                    y_max = curr[4] + (curr[6] - curr[4]) / 2.0
                    bbox = np.array([y_min, x_min, y_max, x_max])
                    boxes[i] = bbox

                    names.append(classes[int(curr[1])])
                    scores.append(curr[2])

                    i = i + 1

                # draw the bounding boxes
                edited_frame_nums.append(frame)
                result = draw_boxes(image, boxes, names, scores, obj_count)

                # write the frame
                cv2.imwrite(os.path.join(
                    output, str(frame) + '.' + 'jpg'), result)

            # set the new frame number
            frame = int(obj[0])

            # clear the tuple and add new object
            objects.clear()
            objects.append(obj)

        # read a new line
        line = bbox_data.readline()
        line.rstrip()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input',  type=str,
                            default=None,  help='path of video to split')
    arg_parser.add_argument('-o', '--output', type=str, default=None,
                            help='path to folder for extracted frames')
    arg_parser.add_argument('-l', '--lower',  type=int, default=0,
                            help='lower bound on frame number extraction')
    arg_parser.add_argument('-p', '--period', type=int, default=1,
                            help='extracts a frame after every period steps')
    arg_parser.add_argument('-f', '--format', type=str, default='jpg',
                            help='the file format to save extracted frames as')
    arg_parser.add_argument('-u', '--upper',  default=float('inf'),
                            help='upper bound on frame number extraction')
    arg_parser.add_argument('-t', '--text',   type=str, default=None,
                            help='input text file of bbox data (<frame>, <class_id>, <conf>, <x_min>, <y_min>, <x_max>, <y_max>)')

    # ns = namespace
    ns, args = arg_parser.parse_known_args(sys.argv)
    # cap = cv2.VideoCapture(ns.input)
    if(not os.path.exists(ns.output)):
        os.makedirs(ns.output)
    # SetFrameNumber(cap, ns.lower)
    # SplitCapture(cap, ns.output, ns.format, ns.upper, ns.period)
    # cap.release()

    if ns.text and ns.input and ns.output:
        DrawBoundingBox(ns.text, ns.input, ns.output)
