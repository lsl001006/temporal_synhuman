import numpy as np
import torch
from .joints_utils import random_joints2D_deviation
from .silhouette_utils import random_remove_bodyparts_IUV, random_occlude_IUV


def load_aug_settings(device, randomly_remove_sil24=True):
    augment_shape = True
    delta_betas_distribution = 'normal'
    delta_betas_std_vector = torch.tensor([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                                        device=device).float()  # used if delta_betas_distribution is 'normal'
    delta_betas_range = [-3., 3.]  # used if delta_betas_distribution is 'uniform'
    smpl_augment_params = {'augment_shape': augment_shape,
                        'delta_betas_distribution': delta_betas_distribution,
                        'delta_betas_std_vector': delta_betas_std_vector,
                        'delta_betas_range': delta_betas_range}
                     
    # Camera
    xy_std = 0.05
    delta_z_range = [-5, 5]
    cam_augment_params = {'xy_std': xy_std,
                        'delta_z_range': delta_z_range}
    
    # BBox
    crop_input = True
    mean_scale_factor = 1.2
    delta_scale_range = [-0.2, 0.2]
    delta_centre_range = [-5, 5]
    bbox_augment_params = {'crop_input': crop_input,
                        'mean_scale_factor': mean_scale_factor,
                        'delta_scale_range': delta_scale_range,
                        'delta_centre_range': delta_centre_range}
    # Proxy Representation
    remove_appendages = True
    deviate_joints2D = True
    deviate_verts2D = True
    occlude_seg = True

    
    remove_appendages_classes = [1, 2, 3, 4, 5, 6]
    remove_appendages_probabilities = [0.1, 0.1, 0.1, 0.1, 0.05, 0.05]
    delta_j2d_dev_range = [-8, 8]
    delta_j2d_hip_dev_range = [-8, 8]
    delta_verts2d_dev_range = [-0.01, 0.01]
    occlude_probability = 0.5
    occlude_box_dim_min = 48
    occlude_box_dim_max = 48
    
    proxy_rep_augment_params = {'remove_appendages': remove_appendages,
                                'deviate_joints2D': deviate_joints2D,
                                'deviate_verts2D': deviate_verts2D,
                                'occlude_seg': occlude_seg,
                                'remove_appendages_classes': remove_appendages_classes,
                                'remove_appendages_probabilities': remove_appendages_probabilities,
                                'delta_j2d_dev_range': delta_j2d_dev_range,
                                'delta_j2d_hip_dev_range': delta_j2d_hip_dev_range,
                                'delta_verts2d_dev_range': delta_verts2d_dev_range,
                                'occlude_probability': occlude_probability,
                                'occlude_box_dim_min': occlude_box_dim_min,
                                'occlude_box_dim_max': occlude_box_dim_max}
    
    if randomly_remove_sil24:
        proxy_rep_augment_params['remove_appendages_classes'] = list(range(1,25))
        proxy_rep_augment_params['remove_appendages_probabilities'] = 24*[0.1]

    # logging.info('SMPL augment params:')
    # logging.info(smpl_augment_params)
    # logging.info('Cam augment params:')
    # logging.info(cam_augment_params)
    # logging.info('BBox augment params')
    # logging.info(bbox_augment_params)
    # logging.info('Proxy rep augment params')
    # logging.info(proxy_rep_augment_params)
    
    return smpl_augment_params, cam_augment_params, bbox_augment_params, proxy_rep_augment_params


def augment_cam_t(mean_cam_t, xy_std=0.05, delta_z_range=[-5, 5]):
    batch_size = mean_cam_t.shape[0]
    device = mean_cam_t.device
    new_cam_t = mean_cam_t.clone()
    delta_tx_ty = torch.randn(batch_size, 2, device=device) * xy_std
    new_cam_t[:, :2] = mean_cam_t[:, :2] + delta_tx_ty

    h, l = delta_z_range
    delta_tz = (h - l) * torch.rand(batch_size, device=device) + l
    new_cam_t[:, 2] = mean_cam_t[:, 2] + delta_tz

    return new_cam_t

def augment_proxy_representation_IUV(orig_segs, orig_joints2D,
                                 proxy_rep_augment_params):
    new_segs = orig_segs.clone()
    new_joints2D = orig_joints2D.clone()
    # import ipdb; ipdb.set_trace()
    
    if proxy_rep_augment_params['remove_appendages']:
        new_segs,_ = random_remove_bodyparts_IUV(new_segs,
                                           classes_to_remove=proxy_rep_augment_params['remove_appendages_classes'],
                                           probabilities_to_remove=proxy_rep_augment_params['remove_appendages_probabilities'])
    if proxy_rep_augment_params['occlude_seg']:
        new_segs = random_occlude_IUV(new_segs,
                                  occlude_probability=proxy_rep_augment_params['occlude_probability'],
                                  occlude_box_dim_min=proxy_rep_augment_params['occlude_box_dim_min'],
                                  occlude_box_dim_max=proxy_rep_augment_params['occlude_box_dim_max'])

    if proxy_rep_augment_params['deviate_joints2D']:
        new_joints2D = random_joints2D_deviation(new_joints2D,
                                                 delta_j2d_dev_range=proxy_rep_augment_params['delta_j2d_dev_range'],
                                                 delta_j2d_hip_dev_range=proxy_rep_augment_params['delta_j2d_hip_dev_range'])
    return new_segs, new_joints2D

def random_verts2D_deviation(vertices, delta_verts2d_dev_range=[-0.01, 0.01]):
    """
    Randomly add 2D uniform noise to vertices to create silhouettes/part segmentations with
    corrupted edges.
    :param vertices: (bs, 6890, 3)
    :param delta_verts2d_dev_range: range of uniform noise.
    """
    batch_size = vertices.shape[0]
    num_verts = vertices.shape[1]
    device = vertices.device

    noisy_vertices = vertices.clone()

    h, l = delta_verts2d_dev_range
    delta_verts2d_dev = (h - l) * torch.rand(batch_size, num_verts, 2, device=device) + l
    noisy_vertices[:, :, :2] = noisy_vertices[:, :, :2] + delta_verts2d_dev

    return noisy_vertices