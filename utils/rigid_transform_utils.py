import cv2
import torch
import numpy as np
from torch.nn import functional as F


def rotate_translate_verts_torch(vertices, axis, angle, trans):
    """
    Rotates and translates batch of vertices.
    :param vertices: B, N, 3
    :param axis: 3,
    :param angle: angle in radians
    :param trans: 3,
    :return:
    """
    r = angle * axis
    R = cv2.Rodrigues(r)[0]
    R = torch.from_numpy(R.astype(np.float32)).to(vertices.device)
    trans = torch.from_numpy(trans.astype(np.float32)).to(vertices.device)

    vertices = torch.einsum('ij,bkj->bki', R, vertices)
    vertices = vertices + trans

    return vertices


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)  # Ensuring columns are unit vectors
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)  # Ensuring column 1 and column 2 are orthogonal
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

def rotmat_to_rot6d(R, stack_columns=False):
    """
    :param R: (B, 3, 3)
    :param stack_columns:
        if True, 6D pose representation is [1st col of R, 2nd col of R]^T = [R11, R21, R31, R12, R22, R32]^T
        if False, 6D pose representation is [R11, R12, R21, R22, R31, R32]^T
        Set to False if doing inverse of rot6d_to_rotmat
    :return: rot6d: (B, 6)
    """
    if stack_columns:
        rot6d = torch.cat([R[:, :, 0], R[:, :, 1]], dim=1)
    else:
        rot6d = R[:, :, :2].contiguous().view(-1, 6)
    return rot6d


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def aa_rotate_translate_points_pytorch3d(points, axes, angles, translations):
    """
    Rotates and translates batch of points from a mesh about given axes and angles.
    :param points: B, N, 3, batch of meshes with N points each
    :param axes: (B,3) or (3,), rotation axes
    :param angles: (B,1) or scalar, rotation angles in radians
    :param translations: (B,3) or (3,), translation vectors
    :return:
    """
    r = axes * angles
    if r.dim() < 2:
        r = r[None, :].expand(points.shape[0], -1)
    R = so3_exponential_map(log_rot=r)  # (B, 3, 3)
    points = torch.einsum('bij,bkj->bki', R, points)
    points = points + translations

    return points
