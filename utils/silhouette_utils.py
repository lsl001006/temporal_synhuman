import torch
import numpy as np
from utils.label_conversions import TWENTYFOUR_PART_SEG_TO_COCO_JOINTS_MAP, BODY_24_6_CONVERT

def random_remove_bodyparts_IUV(iuv, classes_to_remove, probabilities_to_remove, joints2D_vis=None, probability_to_remove_joints=0.5):
    """
    :param iuv: (bs, wh, wh, 3)
    :param classes_to_remove: list of classes to remove. Classes are integers (as defined in
    nmr_renderer.py).
    :param probabilities_to_remove: probability of removal for each class.
    """
    partI = (iuv[:,:,:,0]*24).round()
    if len(classes_to_remove)==6:
        seg = torch.zeros_like(partI)
        for s_id, t_id in BODY_24_6_CONVERT.items():
            seg[partI==s_id] = t_id
    elif len(classes_to_remove)==24:
        seg = partI.clone()
    assert len(classes_to_remove) == len(probabilities_to_remove)
    assert len(classes_to_remove)==6 or len(classes_to_remove)==24
    
    batch_size = iuv.shape[0]
    for i in range(len(classes_to_remove)):
        class_to_remove = classes_to_remove[i]
        prob_to_remove = probabilities_to_remove[i]

        # Determine which samples to augment in the batch
        rand_vec = np.random.rand(batch_size) < prob_to_remove
        iuv_to_augment = iuv[rand_vec].clone()
        seg_to_augment = seg[rand_vec].clone()
        iuv_to_augment[seg_to_augment == class_to_remove] = torch.tensor([0,0,0]).float().to(iuv.device)
        iuv[rand_vec] = iuv_to_augment

        if joints2D_vis is not None:
            if class_to_remove in TWENTYFOUR_PART_SEG_TO_COCO_JOINTS_MAP.keys():
                joint_to_remove = TWENTYFOUR_PART_SEG_TO_COCO_JOINTS_MAP[class_to_remove]

                # Determine which samples with removed class_to_remove will also have joints removed
                rand_vec_joints = np.random.rand(batch_size) < probability_to_remove_joints
                rand_vec_joints = np.logical_and(rand_vec, rand_vec_joints)  # Samples with removed class_to_remove AND removed corresponding joints
                joints2D_vis[rand_vec_joints, joint_to_remove] = 0
    return iuv, joints2D_vis

def random_occlude_IUV(iuv, occlude_probability=0.5, occlude_box_dim_min=48, occlude_box_dim_max=48):
    """
    Randomly occlude silhouette/part segmentation with boxes.
    :param seg: (bs, wh, wh,3)
    """
    batch_size = iuv.shape[0]
    seg_wh = iuv.shape[1]
    seg_centre = seg_wh/2
    x_h, x_l = seg_centre - 0.3*seg_wh/2, seg_centre + 0.3*seg_wh/2
    y_h, y_l = seg_centre - 0.3*seg_wh/2, seg_centre + 0.3*seg_wh/2

    x0 = (x_h - x_l) * np.random.rand(batch_size) + x_l
    y0 = (y_h - y_l) * np.random.rand(batch_size) + y_l
    occlude_box_dim_x = occlude_box_dim_min + (occlude_box_dim_max-occlude_box_dim_min)*np.random.rand(batch_size)
    occlude_box_dim_y = occlude_box_dim_min + (occlude_box_dim_max-occlude_box_dim_min)*np.random.rand(batch_size)

    box_x1 = (x0 - occlude_box_dim_x / 2)
    box_x2 = (x0 + occlude_box_dim_x / 2)
    box_y1 = (y0 - occlude_box_dim_y / 2)
    box_y2 = (y0 + occlude_box_dim_y / 2)

    box_x1 = np.maximum(box_x1, 0).astype(np.int16)
    box_y1 = np.maximum(box_y1, 0).astype(np.int16)
    box_x2 = np.minimum(box_x2, seg_wh).astype(np.int16)
    box_y2 = np.minimum(box_y2, seg_wh).astype(np.int16)

    rand_vec = np.random.rand(batch_size)
    # occlude_batch_mask = (rand_vec<occlude_probability)
    # bbox_mask = torch.zeros((batch_size, seg_wh, seg_wh)).unsqueeze(dim=3).to(iuv.device)
    # bbox_mask[:, box_x1:box_x2, box_y1:box_y2,:]=torch.tensor([1,1,1]).float().to(iuv.device)
    # occluded_iuv = torch.zeros_like(iuv).to(iuv.device)
    # import ipdb; ipdb.set_trace()
    
    for i in range(batch_size):
        if rand_vec[i] < occlude_probability:
            iuv[i, box_x1[i]:box_x2[i], box_y1[i]:box_y2[i],:] = torch.tensor([0,0,0]).float().to(iuv.device)

    return iuv