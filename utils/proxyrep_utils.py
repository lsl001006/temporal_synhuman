import torch
import configs


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

def convert_multiclass_to_binary_labels_torch(multiclass_labels):
    """
    Converts multiclass segmentation labels into a binary mask.
    """
    binary_labels = torch.zeros_like(multiclass_labels)
    binary_labels[multiclass_labels != 0] = 1

    return binary_labels

def convert_to_proxyfeat_batch(IUV, joints2d_coco, joints2d_no_occluded_coco=None, pr_mode='bj'):
    """
    IUV: (b*256*256*3)
    joints2d_coco: (b*17*2)
    """
    partseg = (IUV[:,:,:,0]*24).round()
    if joints2d_no_occluded_coco is None:
        joints2d_no_occluded_coco = torch.ones(joints2d_coco.shape[:2], device=joints2d_coco.device, dtype=torch.bool)
    
    assert pr_mode=='bj'
    
    
    j2d_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(joints2d_coco, 
                    configs.REGRESSOR_IMG_WH, joints2d_no_occluded_coco)#[0,1](BS,17,H,W)
    binaryseg = convert_multiclass_to_binary_labels_torch(partseg).unsqueeze(1)#(BS,1,H,W)
    inter_represent = torch.cat([binaryseg, j2d_heatmaps], dim=1)
    

    return inter_represent


