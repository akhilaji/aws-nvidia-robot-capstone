import argparse
import math
import os
import sys
import cv2
import numpy as np
import PIL
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

classes = ["Bottle", "Chair", "Computer keyboard", "Computer monitor", "Computer mouse", "Corded phone", "Desk", "Headphones", "Laptop", "Microphone", "Mobile phone", "Mug", "Office building", "Office supplies", "Pen", "Person", "Stapler", "Table", "Tablet computer", "Telephone", "Whiteboard"]

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

def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
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

def SplitCapture(cap, output, file_format, upper, period):
    fmt = '%0' + str(int(math.log10(GetFrameCount(cap))) + 1) + 'd'
    i = GetFrameNumber(cap)
    while(i < upper and cap.grab()):
        res, frame = cap.read()
        if(res):
            cv2.imwrite(os.path.join(output, fmt % int(i) + '.' + file_format), frame)
        SetFrameNumber(cap, i + period)
        i += period

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input',  type = str, default = None,  help = 'path of video to split')
    arg_parser.add_argument('-o', '--output', type = str, default = None,  help = 'path to folder for extracted frames')
    arg_parser.add_argument('-l', '--lower',  type = int, default = 0,     help = 'lower bound on frame number extraction')
    arg_parser.add_argument('-p', '--period', type = int, default = 1,     help = 'extracts a frame after every period steps')
    arg_parser.add_argument('-f', '--format', type = str, default = 'jpg', help = 'the file format to save extracted frames as')
    arg_parser.add_argument('-u', '--upper',  default = float('inf'),      help = 'upper bound on frame number extraction')
    arg_parser.add_argument('-t', '--text',   type = str, default = None,  help = 'input text file of bbox data (<frame>, <class_id>, <conf>, <x_min>, <y_min>, <x_max>, <y_max>)')
    #ns = namespace
    ns, args = arg_parser.parse_known_args(sys.argv)
    cap = cv2.VideoCapture(ns.input)
    if(not os.path.exists(ns.output)):
        os.makedirs(ns.output)
    SetFrameNumber(cap, ns.lower)
    SplitCapture(cap, ns.output, ns.format, ns.upper, ns.period)
    cap.release()
