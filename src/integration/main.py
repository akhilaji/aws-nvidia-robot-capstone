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
            dist_thresh=100.0,
        ),
    )

def main(args: argparse.Namespace) -> None:
    detection_pipeline = construct_detection_pipeline(args)

    filename = '20181031205142-shutterstock-1031148421-crop.jpeg'
    bgr_img = cv2.imread(filename)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    detections = detection_pipeline(rgb_img)
    for det in detections:
        visualization.draw_detection(
            bgr_img,
            det,
            [255,0,200],
            cv2.FONT_HERSHEY_PLAIN,
            1.0,
        )
        print(det)
    resized = cv2.resize(bgr_img, (1920, 1080))
    cv2.imshow(filename, resized)
    cv2.waitKey(0)
    return None

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model-path', type=str, default='./yolov4-608/')
    arg_parser.add_argument('--input-w', type=int, default=608)
    arg_parser.add_argument('--input-h', type=int, default=608)
    arg_parser.add_argument('--depth-device', type=str, default='cuda')
    main(arg_parser.parse_args())
