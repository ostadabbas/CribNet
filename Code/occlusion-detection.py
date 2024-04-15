import cv2
import numpy as np
import json
from detectron2.utils.visualizer import Visualizer, ColorMode
import torch, detectron2
import os
import cv2
import argparse

from datetime import datetime

# DATA SET PREPARATION AND LOADING
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

# from detectron2.utils.visualizer import Visualizer
# from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
ARCHITECTURE = "mask_rcnn_R_101_FPN_3x"
CONFIG_FILE_PATH = f"COCO-InstanceSegmentation/{ARCHITECTURE}.yaml"
MAX_ITER = 6000
EVAL_PERIOD = 200
BASE_LR = 0.001
NUM_CLASSES = 2
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE_PATH))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_FILE_PATH)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.TEST.EVAL_PERIOD = EVAL_PERIOD
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.INPUT.MASK_FORMAT='bitmask'
cfg.SOLVER.BASE_LR = BASE_LR
cfg.SOLVER.MAX_ITER = MAX_ITER
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

cfg.MODEL.WEIGHTS = '/content/drive/MyDrive/Cribnet/models/model_final.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
predictor = DefaultPredictor(cfg)


def is_occluded(keypoint, masks):
    x, y = int(keypoint[0]), int(keypoint[1])
    for mask in masks:
        if mask[y, x]:
            return True
    return False

def check_occlusions(keypoints, masks, keypoints_labels):
    occlusions = {}
    for label, keypoint in zip(keypoints_labels, keypoints):
        occlusions[label] = is_occluded(keypoint, masks)
    return occlusions


def visualize_keypoints(image, keypoints, skeleton, keypoints_labels, masks):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    occlusions = check_occlusions(keypoints, masks, keypoints_labels)

    for i, keypoint in enumerate(keypoints):
        x, y = int(keypoint[0]), int(keypoint[1])
        color = (0, 0, 255) if occlusions[keypoints_labels[i]] else (0, 255, 0)
        cv2.circle(image, (x, y), 7, color, thickness=-5, lineType=cv2.FILLED)

    for line in skeleton:
        start_point, end_point = line
        start_keypoint = keypoints[start_point - 1]
        end_keypoint = keypoints[end_point - 1]
        start_occluded = occlusions[keypoints_labels[start_point - 1]]
        end_occluded = occlusions[keypoints_labels[end_point - 1]]

        if start_occluded and end_occluded:
            line_color = (132,136,255)  # Both keypoints occluded
        elif start_occluded or end_occluded:
            line_color = (219, 201, 154)  # One keypoint occluded
        else:
            line_color = (219, 201, 154)  # No keypoints occluded

        cv2.line(image, (int(start_keypoint[0]), int(start_keypoint[1])), (int(end_keypoint[0]), int(end_keypoint[1])), line_color, 10)

    with open(args.output_txt_path, 'w') as f:
        for part, is_covered in occlusions.items():
            if is_covered:
                f.write(f"{part} is covered by the blanket\n")
            else:
                f.write(f"{part} is not covered by the blanket\n")

    cv2.imwrite(args.output_image_path, image)


keypoints_labels = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

skeleton = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
    [4, 6], [5, 7]
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and check occlusions for keypoints.")
    parser.add_argument("image_path", help="Path to the input image file")
    parser.add_argument("json_data_path", help="Path to the JSON file containing keypoints data")
    parser.add_argument("output_image_path", help="Path where the output image will be saved")
    parser.add_argument("output_txt_path", help="Path where the output text file will be saved")

    args = parser.parse_args()


    with open(args.json_data_path) as f:
      json_data = json.load(f)

    img = cv2.imread(args.image_path)


    outputs = predictor(img)
    masks = outputs["instances"].to("cpu").pred_masks.numpy() if outputs["instances"].has("pred_masks") else []
    masks = [mask.astype(np.uint8) for mask in masks]

    visualizer = Visualizer(img[:, :, ::-1], instance_mode=ColorMode.IMAGE_BW)
    out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    visualized_image = out.get_image()[:, :, ::-1]

    keypoints = np.array(json_data['keypoints']).reshape(-1, 3)[:, :2]

    visualize_keypoints(visualized_image, keypoints, skeleton, keypoints_labels, masks)
