import torch
import configs

import sys
sys.path.append(configs.DETECTRON2_PATH)
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
sys.path.append(f"{configs.DETECTRON2_PATH}/projects/DensePose")
from densepose.config import add_densepose_config

from utils.bbox_utils import get_largest_centred_bounding_box, select_bboxes_gtcrop, expand_bbox

def setup_densepose_silhouettes(backbone='dl101'):
    if backbone=='base50':
        densepose_config_file = f"{configs.DETECTRON2_PATH}/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
        densepose_model_weight = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
    elif backbone=='base101':
        densepose_config_file = f"{configs.DETECTRON2_PATH}/projects/DensePose/configs/densepose_rcnn_R_101_FPN_s1x.yaml"
        densepose_model_weight = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/165712084/model_final_c6ab63.pkl"
    elif backbone=='dl50':
        densepose_config_file = f"{configs.DETECTRON2_PATH}/projects/DensePose/configs/densepose_rcnn_R_50_FPN_DL_s1x.yaml"
        densepose_model_weight = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_DL_s1x/165712097/model_final_0ed407.pkl"
    elif backbone=='dl101':
        densepose_config_file = f"{configs.DETECTRON2_PATH}/projects/DensePose/configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml"
        densepose_model_weight = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_s1x/165712116/model_final_844d15.pkl"
    elif backbone=='dl101_wc2m':
        densepose_config_file = f"{configs.DETECTRON2_PATH}/projects/DensePose/configs/densepose_rcnn_R_101_FPN_DL_WC2M_s1x.yaml"
        densepose_model_weight = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC2M_s1x/216245790/model_final_de6e7a.pkl"

    densepose_cfg = get_cfg()
    add_densepose_config(densepose_cfg)
    densepose_cfg.merge_from_file(densepose_config_file)
    densepose_cfg.MODEL.WEIGHTS = densepose_model_weight
    densepose_cfg.freeze()
    silhouette_predictor = DefaultPredictor(densepose_cfg)

    return silhouette_predictor


def predict_silhouette_densepose(input_image, predictor, extractor, center, scale, pad_to_imgsize=True):
    img_h, img_w = input_image.shape[:2] 
    with torch.no_grad():
        outputs = predictor(input_image)["instances"]
    # segm = outputs.get("pred_densepose").fine_segm
    bboxes = outputs.get("pred_boxes").tensor #x0,y0,x1,y1
    # 
    if scale>0:
        verified, idx = select_bboxes_gtcrop(input_image, bboxes, center, scale,target_wh=256, vispath_ambiguous='')
    else:
        verified = True if bboxes.shape[0]>0 else False
        if verified:
            idx = get_largest_centred_bounding_box(bboxes.cpu().numpy(), img_w, img_h)

    if verified:
        assert outputs.has("pred_densepose")
        # import ipdb; ipdb.set_trace()
        densepose = extractor(outputs[int(idx)])[0][0]
        imap = densepose.labels/24.0
        uvmap = densepose.uv
        uvmap = torch.clip(uvmap, min=0, max=1)
        # import ipdb; ipdb.set_trace()
        iuv = torch.cat([imap[None], uvmap])#(3,h,w)
        if pad_to_imgsize:
            iuv = expand_bbox(iuv, img_h, img_w, bboxes[idx].cpu().numpy())
        return iuv.permute(1,2,0)
    else:
        return None