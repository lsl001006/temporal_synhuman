import cv2
import numpy as np
from utils.bbox_utils import get_largest_centred_bounding_box, select_bboxes_gtcrop
import configs

import sys
sys.path.append(configs.DETECTRON2_PATH)
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

def setup_j2d_predictor(backbone='r50'):
    # Keypoint-RCNN
    if backbone=='r501x':
        kprcnn_config_file = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"
    elif backbone=='r50':
        kprcnn_config_file = f"COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    elif backbone=='r101':
        kprcnn_config_file = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"
    elif backbone=='x101':
        kprcnn_config_file = "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"
    kprcnn_cfg = get_cfg()
    kprcnn_cfg.merge_from_file(f"{configs.DETECTRON2_PATH}/configs/{kprcnn_config_file}")
    kprcnn_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    kprcnn_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(kprcnn_config_file)
    kprcnn_cfg.freeze()
    joints2D_predictor = DefaultPredictor(kprcnn_cfg)
    return joints2D_predictor

def predict_joints2D(input_image, predictor):
    """
    Predicts 2D joints (17 2D joints in COCO convention along with prediction confidence)
    given a cropped and centred input image.
    :param input_images: (wh, wh)
    :param predictor: instance of detectron2 DefaultPredictor class, created with the
    appropriate config file.
    """
    image = np.copy(input_image)
    orig_h, orig_w = image.shape[:2]
    outputs = predictor(image)  # Multiple bboxes + keypoints predictions if there are multiple people in the image
    bboxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
    if bboxes.shape[0] == 0:  # Can't find any people in image
        keypoints = np.zeros((17, 3))
    else:
        largest_centred_bbox_index = get_largest_centred_bounding_box(bboxes, orig_w, orig_h)  # Picks out centred person that is largest in the image.
        keypoints = outputs['instances'].pred_keypoints.cpu().numpy()
        keypoints = keypoints[largest_centred_bbox_index]

        for j in range(keypoints.shape[0]):
            cv2.circle(image, (keypoints[j, 0], keypoints[j, 1]), 5, (0, 255, 0), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            fontColor = (0, 0, 255)
            cv2.putText(image, str(j), (keypoints[j, 0], keypoints[j, 1]),
                                     font, fontScale, fontColor, lineType=2)

    return keypoints, image



def predict_joints2D_new(input_image, predictor, center, scale):
    """
    Predicts 2D joints (17 2D joints in COCO convention along with prediction confidence)
    given a cropped and centred input image.
    :param input_images: (wh, wh)
    :param predictor: instance of detectron2 DefaultPredictor class, created with the
    appropriate config file.
    """
    img_h, img_w = input_image.shape[:2]
    outputs = predictor(input_image)  # Multiple bboxes + keypoints predictions if there are multiple people in the image
    bboxes = outputs['instances'].pred_boxes.tensor
    keypoints = outputs['instances'].pred_keypoints.cpu().numpy()#pred_keypoint_heatmaps(1,17,56,56)
    # 
    if scale>0:
        verified, idx = select_bboxes_gtcrop(input_image, bboxes, center, scale, target_wh=256, vispath_ambiguous='')
    else:
        verified = True if bboxes.shape[0]>0 else False
        if verified:
            idx = get_largest_centred_bounding_box(bboxes.cpu().numpy(), img_w, img_h)
    if verified: 
        return keypoints[idx]
    else:
        return None

def predict_joints2D_box(input_image, predictor):
    img_h, img_w = input_image.shape[:2]
    outputs = predictor(input_image)  # Multiple bboxes + keypoints predictions if there are multiple people in the image
    bboxes = outputs['instances'].pred_boxes.tensor
    joints_candidates = outputs['instances'].pred_keypoints.cpu().numpy()#pred_keypoint_heatmaps(1,17,56,56)
    # 
    if joints_candidates.shape[0]==1:
        return joints_candidates[0], bboxes[0]
    elif joints_candidates.shape[0]>1:
        idx = get_largest_centred_bounding_box(bboxes.cpu().numpy(), img_w, img_h)
        return joints_candidates[idx], bboxes[idx]
    else:
        return None, None