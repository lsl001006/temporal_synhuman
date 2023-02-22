"""
Contains functions for label conversions.
"""
import numpy as np
import torch

# ------------------------ Joint label conventions ------------------------
# The SMPL model (im smpl_official.py) returns a large superset of joints.
# Different subsets are used during training - e.g. H36M 3D joints convention and COCO 2D joints convention.
# You may wish to use different subsets in accordance with your training data/inference needs.

# The joints superset is broken down into: 45 SMPL joints (24 standard + additional fingers/toes/face),
# 9 extra joints, 19 cocoplus joints and 17 H36M joints.
# The 45 SMPL joints are converted to COCO joints with the map below.
# (Not really sure how coco and cocoplus are related.)

# Indices to get 17 COCO joints and 17 H36M joints from joints superset.
#"COCOkeypoints": [ "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", 
# "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle" ]
ALL_JOINTS_TO_COCO_MAP = [24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
ALL_JOINTS_TO_H36M_MAP = list(range(73, 90))

# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]
J17_TO_J14 = list(range(14))

J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
# J24_to_J14 = [0,1,2,3,4,5,6,7,8,9,10,11,17,18] #?????TOCHECK
# joints_name = ("R_Ankle0", "R_Knee1", "R_Hip2", "L_Hip3", "L_Knee4", "L_Ankle5", "R_Wrist6",
#         "R_Elbow7", "R_Shoulder8", "L_Shoulder9", "L_Elbow10", "L_Wrist11", "Thorax12",
#         "Head13", "HeadTop14")

# **********24 body-part convention:
# 1, 2 = Torso, 3 = Right Hand, 4 = Left Hand, 5 = Left Foot, 6 = Right Foot, 
# 7, 9 = Upper Leg Right, 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right, 12, 14 = Lower Leg Left, 
# 15, 17 = Upper Arm Left, 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left, 20, 22 = Lower Arm Right, 
# 23, 24 = Head;
# **********6 body-part convention:
# 0 - background
# 1 - left arm
# 2 - right arm
# 3 - head
# 4 - left leg
# 5 - right leg
# 6 - torso
# *********14 body-part convention:
# 1 = Head, 2 = Torso
# 3 = Upper Arm Left, 4 = Lower Arm Left, 5 = Left Hand,
# 6 = Upper Arm Right, 7 = Lower Arm Right, 8 = Right Hand,
# 9 = Upper Leg Left, 10 = Lower Leg Left, 11 = Left Foot,
# 12 = Upper Leg Right, 13 = Lower Leg Right, 14 = Right Foot,

BODY_24_6_CONVERT ={1:6, 2:6, 3:2, 4:1, 5:4, 6:5, 7:5, 8:4, 9:5, 10:4, 11:5, 
            12:4, 13:5, 14:4, 15:1, 16:2, 17:1, 18:2, 19:1, 20:2, 21:1, 22:2, 23:3, 24:3}

BODY_24_14_CONVERT = {1:2, 2:2, 3:8, 4:5, 5:11, 6:14, 
                    7:12, 8:9, 9:12, 10:9, 11:13, 12:10, 13:13, 14:10,
                    15:3, 17:3, 16:6, 18:6, 19:4, 21:4, 20:7, 22:7,
                    23:1, 24:1}


# Joint label and body part seg label matching
# 24 part seg: COCO Joints
TWENTYFOUR_PART_SEG_TO_COCO_JOINTS_MAP = {19: 7,
                                          21: 7,
                                          20: 8,
                                          22: 8,
                                          4: 9,
                                          3: 10,
                                          12: 13,
                                          14: 13,
                                          11: 14,
                                          13: 14,
                                          5: 15,
                                          6: 16}

def convert_densepose_to_6part_lsp_labels(densepose_seg):
    lsp_6part_seg = np.zeros_like(densepose_seg)

    lsp_6part_seg[densepose_seg == 1] = 6
    lsp_6part_seg[densepose_seg == 2] = 6
    lsp_6part_seg[densepose_seg == 3] = 2
    lsp_6part_seg[densepose_seg == 4] = 1
    lsp_6part_seg[densepose_seg == 5] = 4
    lsp_6part_seg[densepose_seg == 6] = 5
    lsp_6part_seg[densepose_seg == 7] = 5
    lsp_6part_seg[densepose_seg == 8] = 4
    lsp_6part_seg[densepose_seg == 9] = 5
    lsp_6part_seg[densepose_seg == 10] = 4
    lsp_6part_seg[densepose_seg == 11] = 5
    lsp_6part_seg[densepose_seg == 12] = 4
    lsp_6part_seg[densepose_seg == 13] = 5
    lsp_6part_seg[densepose_seg == 14] = 4
    lsp_6part_seg[densepose_seg == 15] = 1
    lsp_6part_seg[densepose_seg == 16] = 2
    lsp_6part_seg[densepose_seg == 17] = 1
    lsp_6part_seg[densepose_seg == 18] = 2
    lsp_6part_seg[densepose_seg == 19] = 1
    lsp_6part_seg[densepose_seg == 20] = 2
    lsp_6part_seg[densepose_seg == 21] = 1
    lsp_6part_seg[densepose_seg == 22] = 2
    lsp_6part_seg[densepose_seg == 23] = 3
    lsp_6part_seg[densepose_seg == 24] = 3

    return lsp_6part_seg


def convert_multiclass_to_binary_labels(multiclass_labels):
    """
    Converts multiclass segmentation labels into a binary mask.
    """
    binary_labels = np.zeros_like(multiclass_labels)
    binary_labels[multiclass_labels != 0] = 1

    return binary_labels

def convert_multiclass_to_binary_labels_torch(multiclass_labels):
    """
    Converts multiclass segmentation labels into a binary mask.
    """
    binary_labels = torch.zeros_like(multiclass_labels)
    binary_labels[multiclass_labels != 0] = 1

    return binary_labels

def convert_multiclass_to_multichannel_binarylabels_torch(multiclass_labels, N_label=24):
    BS, H, W = multiclass_labels.shape
    binary_labels = torch.zeros((BS, N_label, H, W)).to(multiclass_labels.device)
    for n in range(N_label):
        binary_labels[:,n,:,:] = (multiclass_labels==n+1)
    return binary_labels


def convert_2Djoints_to_gaussian_heatmaps_torch_strapv3(joints2D,
                                                img_wh,
                                                std=4):
    """
    :param joints2D: (B, N, 2) tensor - batch of 2D joints.
    :param img_wh: int, dimensions of square heatmaps
    :param std: standard deviation of gaussian blobs
    :return heatmaps: (B, N, img_wh, img_wh) - batch of 2D joint heatmaps (channels first).
    """
    device = joints2D.device

    xx, yy = torch.meshgrid(torch.arange(img_wh, device=device),
                            torch.arange(img_wh, device=device))
    xx = xx[None, None, :, :].float()
    yy = yy[None, None, :, :].float()

    j2d_u = joints2D[:, :, 0, None, None]  # Horizontal coord (columns)
    j2d_v = joints2D[:, :, 1, None, None]  # Vertical coord (rows)
    heatmap = torch.exp(-(((xx - j2d_v) / std) ** 2) / 2 - (((yy - j2d_u) / std) ** 2) / 2)
    return heatmap

def convert_2Djoints_to_gaussian_heatmaps(joints2D, img_wh, std=4):
    """
    Converts 2D joints locations to img_wh x img_wh x num_joints gaussian heatmaps with given
    standard deviation var.
    """
    num_joints = joints2D.shape[0]
    size = 2 * std  # Truncate gaussian at 2 std from joint location.
    heatmaps = np.zeros((img_wh, img_wh, num_joints), dtype=np.float32)
    for i in range(joints2D.shape[0]):
        if np.all(joints2D[i] > -size) and np.all(joints2D[i] < img_wh-1+size):
            x, y = np.meshgrid(np.linspace(-size, size, 2*size),
                               np.linspace(-size, size, 2*size))
            d = np.sqrt(x * x + y * y)
            gaussian = np.exp(-(d ** 2 / (2.0 * std ** 2)))

            joint_centre = joints2D[i]
            hmap_start_x = max(0, joint_centre[0] - size)
            hmap_end_x = min(img_wh-1, joint_centre[0] + size)
            hmap_start_y = max(0, joint_centre[1] - size)
            hmap_end_y = min(img_wh-1, joint_centre[1] + size)

            g_start_x = max(0, size - joint_centre[0])
            g_end_x = min(2*size, 2*size - (size + joint_centre[0] - (img_wh-1)))
            g_start_y = max(0, size - joint_centre[1])
            g_end_y = min(2 * size, 2 * size - (size + joint_centre[1] - (img_wh-1)))

            heatmaps[hmap_start_y:hmap_end_y,
            hmap_start_x:hmap_end_x, i] = gaussian[g_start_y:g_end_y, g_start_x:g_end_x]

    return heatmaps


def convert_2Djoints_to_gaussian_heatmaps_torch(joints2D_rounded, img_wh, joints2d_visibility=None, std=4):
    """
    Converts 2D joints locations to img_wh x img_wh x num_joints gaussian heatmaps with given
    standard deviation var.
    :param joints2D: (B, N, 2) tensor - batch of 2D joints.
    :return heatmaps: (B, N, img_wh, img_wh) - batch of 2D joint heatmaps.
    """
    if joints2d_visibility is None:
        joints2d_visibility = torch.ones(joints2D_rounded.shape[:2], device=joints2D_rounded.device, dtype=torch.bool)
    batch_size = joints2D_rounded.shape[0]
    num_joints = joints2D_rounded.shape[1]
    device = joints2D_rounded.device
    # heatmaps0 = torch.zeros((batch_size, num_joints, img_wh, img_wh), device=device).float()
    # heatmaps = torch.zeros((batch_size, num_joints, img_wh, img_wh), device=device).float()

    size = 2 * std  # Truncate gaussian at 2 std from joint location.
    x, y = torch.meshgrid(torch.linspace(-size, size, 2 * size),
                          torch.linspace(-size, size, 2 * size))
    x = x.to(device)
    y = y.to(device)
    d = torch.sqrt(x * x + y * y)
    gaussian = torch.exp(-(d ** 2 / (2.0 * std ** 2)))
    # import ipdb; ipdb.set_trace()
    '''
    print('Start 1', datetime.now().strftime("%m%d%H%M%S"))
    usable_joints = torch.bitwise_and((joints2D_rounded> -size).all(dim=2), (joints2D_rounded< img_wh-1+size).all(dim=2)) #(bs,17)
    usable_joints = usable_joints.int()[:,:, None, None] #(bs,17,1,1)
    #
    ts_0 = torch.tensor(0).to(device)
    ts_wh = torch.tensor(img_wh).to(device)
    h_x0 = torch.maximum(ts_0, joints2D_rounded[:,:,0]-size)#(bs,17)
    h_x1 = torch.minimum(ts_wh-1, joints2D_rounded[:,:,0]+size)
    h_y0 = torch.maximum(ts_0, joints2D_rounded[:,:,1]-size)
    h_y1 = torch.minimum(ts_wh-1, joints2D_rounded[:,:,1]+size)
    ts_gs = torch.tensor(size).to(device)
    g_x0 = torch.maximum(ts_0, size-joints2D_rounded[:,:,0])
    g_x1 = torch.minimum(2*ts_gs, ts_gs-joints2D_rounded[:,:,0]+ ts_wh -1)
    g_y0 = torch.maximum(ts_0, size-joints2D_rounded[:,:,1])
    g_y1 = torch.minimum(2*ts_gs, ts_gs-joints2D_rounded[:,:,1]+ ts_wh -1)
    #
    for i in range(batch_size):
        for j in range(num_joints):
            heatmaps[i, j, h_y0[i,j]:h_y1[i,j], h_x0[i,j]:h_x1[i,j]] = gaussian[g_y0[i,j]:g_y1[i,j], g_x0[i,j]:g_x1[i,j]]
    heatmaps = usable_joints*heatmaps + (1-usable_joints)*heatmaps0
    
    print('Start 2', datetime.now().strftime("%m%d%H%M%S"))
    '''
    ###
    heatmaps2 = torch.zeros((batch_size, num_joints, img_wh, img_wh), device=device).float()
    for i in range(batch_size):
        for j in range(num_joints):
            if torch.all(joints2D_rounded[i, j] > -size) and torch.all(joints2D_rounded[i, j] < img_wh-1+size):
                if joints2d_visibility[i,j]:
                    joint_centre = joints2D_rounded[i, j]
                    hmap_start_x = max(0, joint_centre[0].item() - size)
                    hmap_end_x = min(img_wh-1, joint_centre[0].item() + size)
                    hmap_start_y = max(0, joint_centre[1].item() - size)
                    hmap_end_y = min(img_wh-1, joint_centre[1].item() + size)

                    g_start_x = max(0, size - joint_centre[0].item())
                    g_end_x = min(2*size, 2*size - (size + joint_centre[0].item() - (img_wh-1)))
                    g_start_y = max(0, size - joint_centre[1].item())
                    g_end_y = min(2 * size, 2 * size - (size + joint_centre[1].item() - (img_wh-1)))
                    if (hmap_end_y-hmap_start_y==g_end_y-g_start_y) and (hmap_end_x-hmap_start_x==g_end_x-g_start_x):
                        heatmaps2[i, j, hmap_start_y:hmap_end_y, hmap_start_x:hmap_end_x] = gaussian[g_start_y:g_end_y, g_start_x:g_end_x]
    # print('End 2', datetime.now().strftime("%m%d%H%M%S"))
    # import ipdb; ipdb.set_trace()
    
    return heatmaps2



def generate_target(joints, joints_vis, sz_hm=[64, 64], sigma=2, gType='gaussian'):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: n_jt vec     #  original n_jt x 3
    :param sigma: for gaussian gen, 3 sigma rule for effective area.  hrnet default 2.
    :return: target, target_weight(1: visible, 0: invisible),  n_jt x 1
    history: gen directly at the jt position, stride should be handled outside
    '''
    n_jt = len(joints)  #
    target_weight = np.ones((n_jt, 1), dtype=np.float32)
    # target_weight[:, 0] = joints_vis[:, 0]
    target_weight[:, 0] = joints_vis        # wt equals to vis

    assert gType == 'gaussian', \
            'Only support gaussian map now!'

    if gType == 'gaussian':
            target = np.zeros((n_jt,
                                sz_hm[1],
                                sz_hm[0]),
                                dtype=np.float32)

            tmp_size = sigma * 3

            for joint_id in range(n_jt):
                # feat_stride = self.image_size / sz_hm
                # mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                # mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                mu_x = int(joints[joint_id][0] + 0.5)   # in hm joints could be in middle,  0.5 to biased to the position.
                mu_y = int(joints[joint_id][1] + 0.5)

                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= sz_hm[0] or ul[1] >= sz_hm[1] \
                                or br[0] < 0 or br[1] < 0:
                        # If not, just return the image as is
                        target_weight[joint_id] = 0
                        continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], sz_hm[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], sz_hm[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], sz_hm[0])
                img_y = max(0, ul[1]), min(br[1], sz_hm[1])

                v = target_weight[joint_id]
                if v > 0.5:
                        target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    # print('min max', target.min(), target.max())
    # if self.use_different_joints_weight:
    #         target_weight = np.multiply(target_weight, self.joints_weight)

    return target, target_weight


def convert_densepose_seg_to_14part_labels(densepose_seg):
        """
        Convert 24 body-part labels (DensePose convention) to 14 body-part labels.
        """
        if isinstance(densepose_seg, torch.Tensor):
            fourteen_part_seg = torch.zeros_like(densepose_seg)
        elif isinstance(densepose_seg, np.ndarray):
            fourteen_part_seg = np.zeros_like(densepose_seg)

        fourteen_part_seg[densepose_seg == 1] = 1
        fourteen_part_seg[densepose_seg == 2] = 1
        fourteen_part_seg[densepose_seg == 3] = 11
        fourteen_part_seg[densepose_seg == 4] = 12
        fourteen_part_seg[densepose_seg == 5] = 14
        fourteen_part_seg[densepose_seg == 6] = 13
        fourteen_part_seg[densepose_seg == 7] = 8
        fourteen_part_seg[densepose_seg == 8] = 6
        fourteen_part_seg[densepose_seg == 9] = 8
        fourteen_part_seg[densepose_seg == 10] = 6
        fourteen_part_seg[densepose_seg == 11] = 9
        fourteen_part_seg[densepose_seg == 12] = 7
        fourteen_part_seg[densepose_seg == 13] = 9
        fourteen_part_seg[densepose_seg == 14] = 7
        fourteen_part_seg[densepose_seg == 15] = 2
        fourteen_part_seg[densepose_seg == 16] = 4
        fourteen_part_seg[densepose_seg == 17] = 2
        fourteen_part_seg[densepose_seg == 18] = 4
        fourteen_part_seg[densepose_seg == 19] = 3
        fourteen_part_seg[densepose_seg == 20] = 5
        fourteen_part_seg[densepose_seg == 21] = 3
        fourteen_part_seg[densepose_seg == 22] = 5
        fourteen_part_seg[densepose_seg == 23] = 10
        fourteen_part_seg[densepose_seg == 24] = 10

        return fourteen_part_seg
    
def lspjoints_root_centered(S):    
    center = (S[:,2,:3] +  S[:,3,:3])/2. #between two hip points
    center = np.expand_dims(center, axis=1)
    return S