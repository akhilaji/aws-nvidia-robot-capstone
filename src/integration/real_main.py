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

def graph_factory() -> graph.Graph:
    return graph.Graph(
        V=set(),
        E=collections.defeaultdict(
            lambda: collections.defaultdict(
                lambda: None
            )
        )
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

def main(args: argparse.Namespace) -> None:
    detection_pipeline = pipeline.DetectionPipeline(
        object_detector=detect.construct_yolov4_object_detector(
            model_path=args.model_path,
            input_dim=(args.input_w, args.input_h),
        ),
        depth_estimator=depth.construct_midas_large(
            device=torch.device('cuda'),
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
            dist_thresh=100.0,
        ),
    )

    read_img = lambda fname: cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    img = read_img('20181031205142-shutterstock-1031148421-crop.jpeg')
    detections = detection_pipeline(img)
    for det in detections:
        print('\t' + '\n\t'.join(str(det).split('\n')), '\n')
    # print('real_main.py - 70:', detections)
    return None

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_path', type=str, default='./yolov4-608/')
    arg_parser.add_argument('--input_w', type=int, default=608)
    arg_parser.add_argument('--input_h', type=int, default=608)
    main(arg_parser.parse_args())