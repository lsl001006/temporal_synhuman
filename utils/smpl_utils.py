import torch
from smplx.lbs import batch_rodrigues
from utils.rigid_transform_utils import rot6d_to_rotmat
from model.model import Build_SMPL

def smpl_forward(shape, pose, smpl_model=None):
    '''
        pose:b*seql*72
        shape:b*seql*10
    '''
    # pose_rotmats, glob_rotmats = convert_theta_to_rotmats(pose[:, 3:], pose[:, :3]) 
    # target_rotmats = torch.cat([glob_rotmats, pose_rotmats], dim=1)
    # Convert pred pose to rotation matrices
    # import pdb;pdb.set_trace()
    assert pose.shape[0] == shape.shape[0]
    assert pose.device == shape.device
    # 为保证shape维度和pose维度一致，使用将bs和seqlen两个维度合并
    if len(shape.shape) == 3:
        shape = torch.flatten(shape, 0, 1)
    if len(pose.shape) == 3:
        pose = torch.flatten(pose, 0, 1)
    
    if pose.shape[-1] == 24*3:
        all_rotmats = batch_rodrigues(pose.contiguous().view(-1, 3))
        all_rotmats = all_rotmats.view(-1, 24, 3, 3)
    elif pose.shape[-1] == 24*6:
        all_rotmats = rot6d_to_rotmat(pose.contiguous()).view(-1, 24, 3, 3)

    glob_rotmats, pose_rotmats = all_rotmats[:, 0].unsqueeze(1), all_rotmats[:, 1:]
    if smpl_model is None:
        batch_size = pose.shape[0]
        smpl_model = Build_SMPL(batch_size, pose.device)

    smpl_vertices, smpl_joints = smpl_model(body_pose=pose_rotmats.contiguous(), # [bs*seql, 23, 3, 3]
                                            global_orient=glob_rotmats.contiguous(), # [bs*seql, 1, 3, 3]
                                            betas=shape.contiguous(), #[bs, seq, 10]
                                            pose2rot=False)
    
    reposed_smpl_vertices, reposed_smpl_joints = smpl_model(betas=shape)
    
    return all_rotmats, smpl_vertices, smpl_joints, reposed_smpl_vertices, reposed_smpl_joints


def uniform_sample_shape(batch_size, mean_shape, delta_betas_range):
    """
    Uniform sampling of shape parameter deviations from the mean.
    """
    device = mean_shape.device
    h, l = delta_betas_range
    delta_betas = (h-l)*torch.rand(batch_size, 10, device=device) + l
    shape = delta_betas + mean_shape
    return shape  # (bs, 10)


def normal_sample_shape(batch_size, seqlen, mean_shape, std_vector):
    """
    Gaussian sampling of shape parameter deviations from the mean.
    """
    device = mean_shape.device
    delta_betas = torch.randn(batch_size, seqlen, 10, device=device)*std_vector
    shape = delta_betas + mean_shape
    return shape # [bs, seqlen, shape]

def sample_shape(orig_shape, mean_shape, smpl_augment_params=None):
    batch_size = orig_shape.shape[0]
    seqlen = orig_shape.shape[1]
    seq_mean_shape = mean_shape.unsqueeze(0).repeat(seqlen, 1) # 使得mean_shape与输入shape维度保持一致
    new_shape = orig_shape

    if smpl_augment_params is not None:
        augment_shape = smpl_augment_params['augment_shape']
        delta_betas_distribution = smpl_augment_params['delta_betas_distribution']  # 'normal' or 'uniform' shape sampling distribution
        delta_betas_range = smpl_augment_params['delta_betas_range']  # Range of uniformly-distributed shape parameters.
        delta_betas_std_vector = smpl_augment_params['delta_betas_std_vector']  # std of normally-distributed the shape parameters.
        if augment_shape:
            assert delta_betas_distribution in ['uniform', 'normal']
            if delta_betas_distribution == 'uniform':
                new_shape = uniform_sample_shape(batch_size, seq_mean_shape, delta_betas_range)
            elif delta_betas_distribution == 'normal':
                assert delta_betas_std_vector is not None
                new_shape = normal_sample_shape(batch_size, seqlen, seq_mean_shape, delta_betas_std_vector)
    return new_shape