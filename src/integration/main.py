from skeleton import calibration
from skeleton import depth
from skeleton import detect
from skeleton import graph
from skeleton import pick
from skeleton import pipeline
from skeleton import reconstruction
from skeleton import track
from skeleton import visualization 

import collections
import itertools

import argparse
from cv2 import cv2
import torch

from typing import Dict, List, TypeVar
ID = TypeVar('ID')

def graph_factory() -> graph.Graph:
    return graph.Graph(
        V=set(),
        E=collections.defeaultdict(
            lambda: collections.defaultdict(
                lambda: None
            )
        )
    )

def construct_detection_pipeline(args: argparse.Namespace) -> pipeline.DetectionPipeline:
    return pipeline.DetectionPipeline(
        object_detector=detect.construct_yolov4_object_detector(
            model_path=args.model_path,
            input_dim=(args.input_w, args.input_h),
        ),
        depth_estimator=depth.construct_midas_large(
            device=torch.device(args.depth_device),
        ),
        point_picker=pick.AverageContourPointPicker(
            camera=calibration.Camera(
                intr=calibration.Intrinsics(
                    fx=1920.0,
                    fy=1080.0,
                    cx=960,
                    cy=540,
                ),
                extr=None,
            ),
        ),
        object_tracker=track.CentroidTracker(
            id_itr=itertools.count(start=0, step=1),
            pruning_age=50,
            dist_thresh=500.0,
        ),
    )

def construct_scene_graph(
        cap: cv2.VideoCapture,
        detection_pipeline: pipeline.DetectionPipeline,
        scene_reconstructor: reconstruction.SceneReconstructor,
    ):
    frame = None
    while cap.grab():
        frame = cap.retrieve(frame)
        detections = detection_pipeline(frame)
        scene_reconstructor.add_frame(detections)
    return scene_reconstructor.finalize()

def run_visualization(
        cap: cv2.VideoCapture,
        out: cv2.VideoWriter,
        detection_pipeline: pipeline.DetectionPipeline,
    ) -> List[Dict[ID, detect.ObjectDetection]]:
    ret, frame = True, None

    objects = dict()
    while cap.grab():
        ret, frame = cap.retrieve(frame)
        detections = detection_pipeline(frame)

        for det in detections:
            objects[det.id] = det.obj_class

        visualization.draw_all_detections(
            img=frame,
            detections=detections,
            color=[255,0,0],
            font_face=cv2.FONT_HERSHEY_PLAIN,
            font_scale=5.0,
            thickness=3
        )

        #print('objects = ' + str(objects))
        out.write(frame)

    return objects.values()

def construct_video_scene(
        args: argparse.Namespace,
        file_name: str) -> List[Dict[ID, detect.ObjectDetection]]:
    detection_pipeline = construct_detection_pipeline(args)
    out_file_name = file_name + '-out.mp4'

    cap = cv2.VideoCapture(file_name)
    cap_framerate = cap.get(cv2.CAP_PROP_XI_FRAMERATE)
    cap_w = cap.get(cv2.CAP_PROP_XI_WIDTH)
    cap_h = cap.get(cv2.CAP_PROP_XI_HEIGHT)

    cap_framerate = 60
    cap_w = 3840
    cap_h = 2160

    #print(cap_framerate)
    #print((cap_w, cap_h))
    out = cv2.VideoWriter(
        out_file_name,
        cv2.VideoWriter_fourcc(*'DIVX'),
        cap_framerate,
        (cap_w, cap_h),
    )

    results = run_visualization(
        cap=cap,
        out=out,
        detection_pipeline=detection_pipeline,
    )

    cap.release()
    out.release()

    return results

def main(args: argparse.Namespace) -> None:
    objects_vid_1 = list(construct_video_scene(args, args.input_videos[0]))
    objects_vid_2 = list(construct_video_scene(args, args.input_videos[1]))

    fr_one = 0;
    fr_two = 0;

    while fr_one < len(objects_vid_1):
        if fr_one < len(objects_vid_2) and fr_two < len(objects_vid_2):
            if objects_vid_1[fr_one] != objects_vid_2[fr_two]:
                print('SCENE DIFFERENCE DETECTION FOUND')
                print('Expected: {}'.format(objects_vid_1[fr_one]))
                print('Found: {}'.format(objects_vid_2[fr_two]))

                # objects don't match at same index, move to the next expected image
                fr_one += 1
            else:
                # objects match at same index, look at next two
                fr_one += 1
                fr_two += 1
        else:
            if object_frames_one > object_frames_two:
                print('EXTRA OBJECTS FOUND IN FIRST VIDEO STREAM')

            fr_one += 1
            fr_two += 1

    #print('objects_vid_1=%r' % objects_vid_1)
    #print('objects_vid_2=%r' % objects_vid_2)

    return None

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_path',   type=str, default='./yolov4-608/')
    arg_parser.add_argument('--class_names',   type=str, default='./data/classes/obj.names')
    arg_parser.add_argument('--input_w',      type=int, default=608)
    arg_parser.add_argument('--input_h',      type=int, default=608)
    arg_parser.add_argument('--depth_device', type=str, default='cuda')
    arg_parser.add_argument('--input_videos', type=str, nargs=2)

    main(arg_parser.parse_args())
