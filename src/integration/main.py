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
    ) -> None:
    ret, frame = True, None
    while cap.grab():
        ret, frame = cap.retrieve(frame)
        detections = detection_pipeline(frame)
        visualization.draw_all_detections(
            img=frame,
            detections=detections,
            color=[255,0,0],
            font_face=cv2.FONT_HERSHEY_PLAIN,
            font_scale=5.0,
            thickness=3
        )
        out.write(frame)

def main(args: argparse.Namespace) -> None:
    detection_pipeline = construct_detection_pipeline(args)
    file_name = args.file_name
    out_file_name = file_name + 'out.mp4'
    cap = cv2.VideoCapture(file_name)
    out = cv2.VideoWriter(out_file_name, cv2.VideoWriter_fourcc(*'DIVX'), 60, (1920, 1080))
    run_visualization(
        cap=cap,
        out=out,
        detection_pipeline=detection_pipeline,
    )
    cap.release()
    out.release()
    return None

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_path',   type=str, default='./yolov4-608/')
    arg_parser.add_argument('--input_w',      type=int, default=608)
    arg_parser.add_argument('--input_h',      type=int, default=608)
    arg_parser.add_argument('--depth_device', type=str, default='cuda')
    arg_parser.add_argument('--file_name',    type=str)
    main(arg_parser.parse_args())
