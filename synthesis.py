import numpy as np
import torch
import torch.nn.functional as nnf

import configs
import utils.label_conversions as LABELCONFIG
from utils.renderer import build_cam_renderer
from utils.aug_utils import load_aug_settings, augment_cam_t, augment_proxy_representation_IUV, random_verts2D_deviation
from utils.smpl_utils import smpl_forward, sample_shape
from utils.bbox_utils import convert_bbox_centre_hw_to_corners, convert_bbox_corners_to_centre_hw
from utils.cam_utils import perspective_project_torch
from utils.joints_utils import check_joints2d_occluded_torch, check_joints2d_visibility_torch
from einops import rearrange


class RenderGenerate():
    def __init__(self, smpl_model, 
                 batch_size, 
                 val_aug_proxy, 
                 device='cpu', 
                 render_options={'j2D':1,'depth':1, 'normal':1, 'iuv':1}):
        self.smpl_model = smpl_model
        self.device = device
        self.batch_size= batch_size
        self.val_aug_proxy = val_aug_proxy
        self.render_depth = render_options.get('depth')
        self.render_normal = render_options.get('normal')
        self.render_iuv = render_options.get('iuv')
        self.render_j2d = render_options.get('j2D')

        self.mean_cam_t, self.cam_K, self.cam_R, self.renderer = build_cam_renderer(
                    batch_size=batch_size*configs.SEQLEN, device=device, render_options=render_options)
        
        self.mean_shape = torch.from_numpy(np.load(configs.SMPL_MEAN_PARAMS_PATH)['shape']).float().to(device)

        self.smpl_augment_params, self.cam_augment_params, self.bbox_augment_params, self.proxy_rep_augment_params = load_aug_settings(device)
        

    def batch_crop_bounding_box_resize(self, iuv_map, joints2D=None, depth_map=None, normal_map=None):
        """
        parts_map: (bs, wh, wh)
        depth_map: (bs, wh, wh)
        normal_map: (bs, wh, wh,3)
        iuv_map: (bs, wh, wh, 3)
        joints2D: (bs, num joints, 2)
        scale: bbox expansion scale
        """
        orig_scale_factor=self.bbox_augment_params['mean_scale_factor'] #1.2
        delta_scale_range=self.bbox_augment_params['delta_scale_range'] #[-0.2,0.2]
        delta_centre_range=self.bbox_augment_params['delta_centre_range'] #[-5,5]

        img_wh = configs.REGRESSOR_IMG_WH
        device = self.device
        batch_size = self.batch_size*configs.SEQLEN
        #all in torch and batch!!!
        all_joints2D = []
        all_depth_map = []
        all_normal_map = []
        all_iuv_map = []
        # 
        parts_map = (24*iuv_map[:,:,:,0]).round()
        seg =  parts_map.cpu().detach().numpy()
        for i in range(batch_size): #numpy need to be torch?
            body_pixels = np.argwhere(seg[i] != 0)
            bbox_corners = np.amin(body_pixels, axis=0), np.amax(body_pixels, axis=0)
            bbox_corners = np.concatenate(bbox_corners) #y1,x1,y2,x2
            centre, height, width = convert_bbox_corners_to_centre_hw(bbox_corners)

            if delta_scale_range is not None:
                h, l = delta_scale_range
                delta_scale = (h - l) * np.random.rand() + l
                scale_factor = orig_scale_factor + delta_scale
            else:
                scale_factor = orig_scale_factor
            
            if delta_centre_range is not None:
                h, l = delta_centre_range
                delta_centre = (h - l) * np.random.rand(2) + l
                centre = centre + delta_centre

            wh = max(height, width) * scale_factor
            bbox_corners = convert_bbox_centre_hw_to_corners(centre, wh, wh)#(y1,x1,y2,x2)

            top_left = bbox_corners[:2].astype(np.int16)
            bottom_right = bbox_corners[2:].astype(np.int16)
            top_left[top_left < 0] = 0
            bottom_right[bottom_right < 0] = 0
            bottom_right[0] = min(img_wh, bottom_right[0])
            bottom_right[1] = min(img_wh, bottom_right[1])
            #crop and pad iuv tensor
            crop_h, crop_w = bottom_right[0]-top_left[0], bottom_right[1]-top_left[1]
            sqaure_hw = max(crop_h, crop_w)
            pad_h, pad_w = (sqaure_hw-crop_h)//2, (sqaure_hw-crop_w)//2

            #JOINTS
            if self.render_j2d:
                cropped_joints2d = joints2D[i] + torch.as_tensor([pad_w-top_left[1], pad_h-top_left[0]])[None].to(device)
                # orig_height, orig_width = bottom_right[0]- top_left[0], bottom_right[1]- top_left[1]
                # print(orig_height, orig_width)
                # import ipdb; ipdb.set_trace()
                resize_scale = img_wh/float(sqaure_hw)
                resized_joints2D = cropped_joints2d *resize_scale
                all_joints2D.append(resized_joints2D[None])

            if self.render_depth:
                cropped_depth_map = depth_map[i, top_left[0]:bottom_right[0],  top_left[1]: bottom_right[1]]
                resized_depth_map = nnf.interpolate(cropped_depth_map[None][None], size=(img_wh, img_wh), mode='nearest')
                all_depth_map.append(resized_depth_map)

            if self.render_normal:
                cropped_normal_map = normal_map[i, top_left[0]:bottom_right[0],  top_left[1]: bottom_right[1], :]
                resized_normal_map = nnf.interpolate(cropped_normal_map.permute(2,0,1)[None], size=(img_wh, img_wh), mode='nearest').permute(0,2,3,1)
                all_normal_map.append(resized_normal_map)
            
            if self.render_iuv:
                iuv_padded = torch.zeros((sqaure_hw, sqaure_hw, 3))
                iuv_padded[pad_h:pad_h+crop_h, pad_w:pad_w+crop_w, :] = iuv_map[i, top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]
                # cropped_iuv_map = iuv_map[i, top_left[0]:bottom_right[0],  top_left[1]: bottom_right[1],:] 
                resized_iuv_map = nnf.interpolate(iuv_padded.permute(2,0,1)[None], size=(img_wh, img_wh), mode='nearest').permute(0,2,3,1)
                all_iuv_map.append(resized_iuv_map)

        all_joints2D= torch.cat(all_joints2D).to(device) if self.render_j2d else None
        all_depth_map= torch.cat(all_depth_map).squeeze() if self.render_depth else None
        all_normal_map= torch.cat(all_normal_map) if self.render_normal else None
        all_iuv_map= torch.cat(all_iuv_map).to(device) if self.render_iuv else None

        return all_joints2D, all_iuv_map, all_depth_map, all_normal_map

    def render_forward(self, pose, shape, is_train):
        '''
        pose:b*seqlenx72
        shape:b*seqlenx10
        '''
        # print(datetime.now().strftime("%m%d%H%M%S"))
        pose = pose.to(self.device).float()
        shape = shape.to(self.device).float()
        if is_train:
            augment_shape, augment_verts, augment_camT, augment_bbox, augment_proxy = True, True, False, True, True
        else:
            augment_shape, augment_verts, augment_camT, augment_bbox, augment_proxy = False, False, False, True, self.val_aug_proxy 

        if augment_shape:
            # import pdb;pdb.set_trace()
            # during train phase, shape will be sampled from normal/uniform distribution
            shape = sample_shape(shape, self.mean_shape, smpl_augment_params=self.smpl_augment_params)
        
        # perhaps no use when there is bounding box
        if augment_camT:
            cam_T = augment_cam_t(self.mean_cam_t,
                    xy_std = self.cam_augment_params['xy_std'],
                    delta_z_range = self.cam_augment_params['delta_z_range'])#b*3
        else:
            cam_T = self.mean_cam_t

        all_rotmats, vertices, joints_all, reposed_vertices, _ = smpl_forward(shape, pose, self.smpl_model)
        
        # convert to 3D/2D joints (b*90*3)
        joints_h36m = joints_all[:, LABELCONFIG.ALL_JOINTS_TO_H36M_MAP, :]
        joints_h36mlsp = joints_h36m[:, LABELCONFIG.H36M_TO_J14, :]
        joints_coco = joints_all[:, LABELCONFIG.ALL_JOINTS_TO_COCO_MAP, :]#17 
        joints2d_coco = perspective_project_torch(joints_coco, self.cam_R,
                                                            cam_T,
                                                            cam_K=self.cam_K)#b*3*3
        # joints2d_coco [bs*seql, 17, 2]
        
        # perturb vertices
        if augment_verts and self.proxy_rep_augment_params['deviate_verts2D']:
            # Vertex noise augmentation to give noisy proxy representation edges
            aug_vertices = random_verts2D_deviation(vertices,delta_verts2d_dev_range=self.proxy_rep_augment_params['delta_verts2d_dev_range'])
        else:
            aug_vertices = vertices
        depth_map, normal_map, iuv_map, _ = self.renderer(aug_vertices, cam_T)
        
        # print('render', datetime.now().strftime("%m%d%H%M%S"))
        
        if augment_bbox and self.bbox_augment_params['crop_input']:
            # Crop inputs according to bounding box
            # + add random scale and centre augmentation
            joints2d_coco, iuv_map, depth_map, normal_map = self.batch_crop_bounding_box_resize(
                iuv_map, joints2D=joints2d_coco, depth_map=depth_map, normal_map=normal_map)
                
        # print('crop', datetime.now().strftime("%m%d%H%M%S"))
        print(f'iuv_map:{iuv_map.shape}\njoints2d_coco:{joints2d_coco.shape}')
        import pdb;pdb.set_trace()

        
        # saveKPIUV(joints2d_coco, iuv_map,rootpath='vis2')
         
        #CHECK OCCLUSION BEFORE PROXY AUGMENT
        joints2d_coco = joints2d_coco.int()
        joints2d_vis_coco = check_joints2d_visibility_torch(joints2d_coco, configs.REGRESSOR_IMG_WH)#(bs,17)
        joints2d_no_occluded_coco = check_joints2d_occluded_torch(joints2d_coco, joints2d_vis_coco, (iuv_map[:,:,:,0]*24).round())
        
        #AUGMENT INPUT ONLY
        if augment_proxy:
            aug_iuv_map, aug_joints2d_coco = augment_proxy_representation_IUV(iuv_map, joints2d_coco.float(),
                                                                self.proxy_rep_augment_params)
        # rearrange tensors to [bs, seql, *] format
        all_rotmats = rearrange(all_rotmats, '(b1 b2) c h w -> b1 b2 c h w', b2=configs.SEQLEN)
        vertices = rearrange(vertices, '(b1 b2) c h -> b1 b2 c h', b2=configs.SEQLEN)
        aug_vertices = rearrange(aug_vertices, '(b1 b2) c h -> b1 b2 c h', b2=configs.SEQLEN)
        reposed_vertices = rearrange(reposed_vertices, '(b1 b2) c h -> b1 b2 c h', b2=configs.SEQLEN)
        joints_h36mlsp = rearrange(joints_h36mlsp, '(b1 b2) c h -> b1 b2 c h', b2=configs.SEQLEN)
        
        
        smpl_dict = {"shape":shape, "all_rotmats":all_rotmats}
        vertices_dict = {"vertices": vertices, # bs*seql, 6890, 3
                         "aug_vertices": aug_vertices, # bs*seql, 6890, 3
                         "reposed_vertices": reposed_vertices} # bs*seql, 6890, 3
        iuv_dict = {"iuv": iuv_map, "aug_iuv": aug_iuv_map} # bs, 256, 256, 3
        joints_dict = {"joints_h36mlsp": joints_h36mlsp, # bs*seql, 14, 3
                       "joints2d_coco": joints2d_coco.int(),
                       "aug_joints2d_coco": aug_joints2d_coco.int(),
                       "joints2d_vis_coco": joints2d_vis_coco,
                       "joints2d_no_occluded_coco": joints2d_no_occluded_coco}
        depth_dict = {"depth": depth_map,
                      "normal": normal_map}
        # import pdb;pdb.set_trace()
        
        return smpl_dict, iuv_dict, joints_dict, vertices_dict, depth_dict

def visualize_joints_iuvmap(joints2d_coco, iuv_map, save_path=configs.VIS_DIR):
    """visualize joints and iuvmap of a set of images

    Args:
        joints2d_coco (_type_): joints keypoints [bs*seql, 17, 2]
        iuv_map (_type_): iuv masks [bs*seql, 256, 256, 3]
        save_path (_type_, optional): path to save visualization results. Defaults to configs.VIS_DIR.
    """
    # 选取1个batch的数据
    joints = rearrange(joints2d_coco, '(b s) c d -> b s c d', s=configs.SEQLEN)[0]
    iuvmap = rearrange(iuv_map, '(b s) h w c -> b s h w c', s=configs.SEQLEN)[0]
    