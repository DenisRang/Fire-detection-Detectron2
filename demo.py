import argparse
import os
import time
import cv2
import tqdm
import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer




def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.config_file))
    cfg.merge_from_list(args.opts)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Demo for fire detection in video")
    parser.add_argument(
        "--config-file",
        default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file from model zoo",
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def run_on_video(video, cfg):
    """
    Visualizes predictions on frames of the input video.
    Args:
        video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
            either a webcam or a video file.
    Yields:
        ndarray: BGR visualizations of each video frame.
    """
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get("__unused")
    video_visualizer = VideoVisualizer(metadata)

    def process_predictions(frame, predictions):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions = predictions["instances"].to('cpu')
        vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)

        # Converts Matplotlib RGB format to OpenCV BGR format
        vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
        return vis_frame

    def frame_from_video(video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    frame_gen = frame_from_video(video)
    for frame in frame_gen:
        frame = np.array(frame)
        yield process_predictions(frame, predictor(frame))


if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    if args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(run_on_video(video, cfg), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
