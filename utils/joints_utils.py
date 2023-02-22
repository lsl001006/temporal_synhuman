import numpy as np
import torch




def undo_keypoint_normalisation(normalised_keypoints, img_wh):
    """
    Converts normalised keypoints from [-1, 1] space to pixel space i.e. [0, img_wh]
    """
    keypoints = (normalised_keypoints + 1) * (img_wh/2.0)
    return keypoints


def check_joints2d_visibility(joints2d, img_wh):
    vis = np.ones(joints2d.shape[1])
    vis[joints2d[0] > img_wh] = 0
    vis[joints2d[1] > img_wh] = 0
    vis[joints2d[0] < 0] = 0
    vis[joints2d[1] < 0] = 0

    return vis


def check_joints2d_visibility_torch(joints2d, img_wh, vis=None):
    """
    Checks if 2D joints are within the image dimensions.
    """
    if vis is None:
        vis = torch.ones(joints2d.shape[:2], device=joints2d.device, dtype=torch.bool)
    vis[joints2d[:, :, 0] > img_wh] = 0
    vis[joints2d[:, :, 1] > img_wh] = 0
    vis[joints2d[:, :, 0] < 0] = 0
    vis[joints2d[:, :, 1] < 0] = 0

    return vis


#24 body-part convention:
# 1, 2 = Torso, 3 = Right Hand, 4 = Left Hand, 5 = Left Foot, 6 = Right Foot, 
# 7, 9 = Upper Leg Right, 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right, 12, 14 = Lower Leg Left, 
# 15, 17 = Upper Arm Left, 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left, 20, 22 = Lower Arm Right, 
# 23, 24 = Head
#14 body-part convention:
# 1 = Head, 2 = Torso
# 3 = Upper Arm Left, 4 = Lower Arm Left, 5 = Left Hand,
# 6 = Upper Arm Right, 7 = Lower Arm Right, 8 = Right Hand,
# 9 = Upper Leg Left, 10 = Lower Leg Left, 11 = Left Foot,
# 12 = Upper Leg Right, 13 = Lower Leg Right, 14 = Right Foot,

#"COCOkeypoints": [ "nose", "left_eye", "right_eye", "left_ear", "right_ear", 
# "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", 
# "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle" ]

Body_24_14_Convert = {1:2, 2:2, 3:8, 4:5, 5:11, 6:14, 
                    7:12, 8:9, 9:12, 10:9, 11:13, 12:10, 13:13, 14:10,
                    15:3, 17:3, 16:6, 18:6, 19:4, 21:4, 20:7, 22:7,
                    23:1, 24:1}
KPcoco14_body14_affiliation = {1:[1], 2:[1], 3:[1], 4:[1], 5:[1],
                           6:[2,3], 7:[2,6], 8:[3,4], 9:[6,7], 10:[4,5], 11:[7,8],
                           12:[2,9], 13:[2,12], 14:[9,10], 15:[12,13], 16:[10,11], 17:[13,14]}

def check_joints2d_occluded_torch(j2d, target_joints2d_vis_coco, Imap, search_box_dim=15):
    """
    j2d: torch.tensor(bs,17,2)
    Imap: (bs, hw, hw)
    """
    bs = j2d.shape[0]
    device = Imap.device
    wh = Imap.shape[1]
    #conver 24 to 14
    partmap = torch.zeros_like(Imap).to(device)
    for s_id, t_id in Body_24_14_Convert.items():
        partmap[Imap==s_id] = t_id
    #
    j2d_noocclude_visibility = target_joints2d_vis_coco.clone()
    x_joints, y_joints = j2d[:,:,0], j2d[:,:,1]
    x1 = torch.max(x_joints-search_box_dim, torch.zeros_like(x_joints))
    y1 = torch.max(y_joints-search_box_dim, torch.zeros_like(y_joints))
    x2 = torch.min(x_joints+search_box_dim, wh*torch.ones_like(x_joints))
    y2 = torch.min(y_joints+search_box_dim, wh*torch.ones_like(y_joints))
    # assert (x1<x2).all() and (y1<y2).all()
    # import ipdb; ipdb.set_trace()
    for b in range(bs):
        for jid, bodyid in KPcoco14_body14_affiliation.items():
            jid = jid -1
            if j2d_noocclude_visibility[b,jid]:
                if y1[b,jid]>=y2[b,jid] or x1[b,jid]>=x2[b,jid]:
                    import ipdb; ipdb.set_trace()
                body_visible = []
                for bid in bodyid:
                    body_visible.append((partmap[b, y1[b,jid]:y2[b,jid], x1[b,jid]:x2[b,jid]] == bid).any())
                j2d_noocclude_visibility[b, jid] = torch.tensor(body_visible).all()#all adjanct body shows round neighbourhoods 


    # import ipdb; ipdb.set_trace()
    return j2d_noocclude_visibility


def check_joints2d_occluded_torch_augnew(seg14part, vis, pixel_count_threshold=50):
        """
        Check if 2D joints are not self-occluded in the rendered silhouette/seg, by checking if corresponding body parts are
        visible in the corresponding 14 part seg.
        :param seg14part: (B, D, D)
        :param vis: (B, 17)
        """
        new_vis = vis.clone()
        joints_to_check_and_corresponding_bodyparts = {7: 3, 8: 5, 9: 12, 10: 11, 13: 7, 14: 9, 15: 14, 16: 13}

        for joint_index in joints_to_check_and_corresponding_bodyparts.keys():
            part = joints_to_check_and_corresponding_bodyparts[joint_index]
            num_pixels_part = (seg14part == part).sum(dim=(1, 2))  # (B,)
            visibility_flag = (num_pixels_part > pixel_count_threshold)  # (B,)
            new_vis[:, joint_index] = (vis[:, joint_index] & visibility_flag)

        return new_vis

def random_joints2D_deviation(joints2D,
                              delta_j2d_dev_range=[-5, 5],
                              delta_j2d_hip_dev_range=[-15, 15]):
    """
    Deviate 2D joint locations with uniform random noise.
    :param joints2D: (bs, num joints, num joints)
    :param delta_j2d_dev_range: uniform noise range.
    :param delta_j2d_hip_dev_range: uniform noise range for hip joints. You may wish to make
    this bigger than for other joints since hip joints are semantically hard to localise and
    can be predicted inaccurately by joint detectors.
    """
    hip_joints = [11, 12]
    other_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14,15, 16]
    batch_size = joints2D.shape[0]
    device = joints2D.device

    h, l = delta_j2d_dev_range
    delta_j2d_dev = (h - l) * torch.rand(batch_size, len(other_joints), 2, device=device) + l
    joints2D[:, other_joints, :] = joints2D[:, other_joints, :] + delta_j2d_dev

    h, l = delta_j2d_hip_dev_range
    delta_j2d_hip_dev_range = (h - l) * torch.rand(batch_size, len(hip_joints), 2, device=device) + l
    joints2D[:, hip_joints, :] = joints2D[:, hip_joints, :] + delta_j2d_hip_dev_range

    return joints2D


