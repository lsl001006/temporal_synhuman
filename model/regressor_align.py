import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as nnf


from .resnet import resnet18
from .ief_module import IEFModule, SpatialIEFModule, FuseIEFModule
from .decoder import ResNet18Dec
from .regressor import SingleInputRegressor


from smplx.lbs import batch_rodrigues
from utils.rigid_transform_utils import rot6d_to_rotmat
from utils.cam_utils import get_intrinsics_matrix, orthographic_project_torch
from utils.renderer import BlendParams, P3dRenderer
import utils.uv_utils as uv_utils 
from pytorch3d.structures import Meshes 
from pytorch3d.renderer.mesh import Textures
from pytorch3d.renderer.cameras import OrthographicCameras
from pytorch3d.renderer import MeshRasterizer
from pytorch3d.renderer.blending import hard_rgb_blend

EPS = 1e-5

class FuseAlignRegressor(SingleInputRegressor):
    def __init__(self,
                batch_size, 
                smpl_model,
                device,
                resnet_in_channels=[1, 17],
                feat_size=[8,8,8],
                itersup=False,
                filter_channels =[512, 256, 32, 8],
                encoddrop=False,
                fuse_r1=False,
                add_channels=[[[0,0],[0,0]], [[0,0],[0,0]]],
                add_fuser=False,
                add_ief=False,
                recon=False):
        super(FuseAlignRegressor, self).__init__()
        self.add_fuser = add_fuser
        self.add_ief = add_ief

        self.feat_size = feat_size
        self.itersup = itersup
        num_pose_params = 24*6
        num_output_params = 3 + num_pose_params + 10

        self.in_channels = resnet_in_channels
        
        self.image_encoder1 = resnet18(in_channels=resnet_in_channels[0], drop = encoddrop,
                                        pretrained=False, pool_out=False, pool1=True)
        
        self.image_encoder2 = resnet18(in_channels=resnet_in_channels[1], drop = encoddrop,
                                        pretrained=False, pool_out=False, pool1=True)
        
        self.setup_mlp(filter_channels=filter_channels)
        #R1
        self.fuse_r1 = fuse_r1
        
        
        if fuse_r1:
            ief_channel1 = filter_channels[0]*2 
            self.fuse_fc = nn.Linear(filter_channels[0]*2 ,filter_channels[0])
        else:
            ief_channel1 = filter_channels[0] 

        self.ief_module_1_m1 = SpatialIEFModule([ief_channel1, ief_channel1],
                                    ief_channel1,
                                    num_output_params,
                                    iterations=1)
        self.ief_module_1_m2 = SpatialIEFModule([ief_channel1, ief_channel1],
                                    ief_channel1,
                                    num_output_params,
                                    iterations=1)
        #R2
        ief_channel2_m1 = filter_channels[0] + add_channels[0][0][0]*feat_size[1]*feat_size[1]+add_channels[0][0][1]*2*17
        ief_channel2_m2 = filter_channels[0] + add_channels[0][1][0]*feat_size[1]*feat_size[1]+add_channels[0][1][1]*2*17

        self.ief_module_2_m1 = SpatialIEFModule([ief_channel2_m1, ief_channel2_m1],
                                    ief_channel2_m1,
                                    num_output_params,
                                    iterations=1) 
        self.ief_module_2_m2 = SpatialIEFModule([ief_channel2_m2, ief_channel2_m2],
                                    ief_channel2_m2,
                                    num_output_params,
                                    iterations=1) 
        if add_ief:
            ief_channel3_m1 = filter_channels[0] + add_channels[1][0][0]*feat_size[2]*feat_size[2]+add_channels[1][0][1]*2*17
            ief_channel3_m2 = filter_channels[0] + add_channels[1][1][0]*feat_size[2]*feat_size[2]+add_channels[1][1][1]*2*17

            self.ief_module_3_m1 = SpatialIEFModule([ief_channel3_m1, ief_channel3_m1],
                                        ief_channel3_m1,
                                        num_output_params,
                                        iterations=1) 
            self.ief_module_3_m2 = SpatialIEFModule([ief_channel3_m2, ief_channel3_m2],
                                        ief_channel3_m2,
                                        num_output_params,
                                        iterations=1)  

        #R3
        ief_channel3 = 2*filter_channels[0] + (add_channels[1][0][0]+add_channels[1][1][0])* feat_size[2]*feat_size[2]+ \
            (add_channels[1][0][1]+add_channels[1][1][1])* 17*2
        self.ief_module_fusion = FuseIEFModule([ief_channel3, ief_channel3],
                                    ief_channel3,
                                    num_output_params*2, #input
                                    num_output_params,
                                    iterations=1)
        #for align
        self.add_channels = add_channels
        self.gt_IUV = None
        self.gt_IUV_mask = None
        self.gt_joints2d_coco = None
        #
        self.smpl_model = smpl_model
        self.device = device
        #Init for rendering
        IUV_processed = uv_utils.IUV_Densepose(device=device)
        I = IUV_processed.get_I() #(7829,1) 
        U, V = IUV_processed.get_UV() #(7829,1)
        IUVnorm= torch.cat([I/24, U, V], dim=1)
        self.IUVnorm_list = [IUVnorm for _ in range(batch_size)] #(bs, 7829, 3)
        self.batch_size = batch_size
        self.blendparam = BlendParams()  

        #for reconstruct
        self.recon = recon
        if recon:
            self.decoder1_2 = ResNet18Dec(num_Blocks=[2,2,2,2], in_planes=filter_channels[0], 
                out_planes=resnet_in_channels[1], activate=False)#b/i/iuv to joints
            activate_m1 = True if resnet_in_channels[0]==1 else False
            out_plane_m1 = 2 if resnet_in_channels[0]==1 else resnet_in_channels[0]
            self.decoder2_1 = ResNet18Dec(num_Blocks=[2,2,2,2], in_planes=filter_channels[0], 
                out_planes=out_plane_m1, activate=activate_m1)#joints to b/i/iuv

    def setup_mlp(self, filter_channels):    
        self.pool2dto1= nn.AdaptiveAvgPool2d((1, 1))
        self.pool2dto4= nn.AdaptiveAvgPool2d((4, 4))
        self.add_mlp(filter_channels=filter_channels, module_name='m1conv')#not shared
        self.add_mlp(filter_channels=filter_channels, module_name='m2conv')#not shared
        

    def forward_feat(self, feat1, feat2, mlp_feats, feat_hw=8):
        assert feat_hw==1 or feat_hw==4 or feat_hw==8
        for n,feat in enumerate([feat1, feat2]):
            if feat_hw==1:
                feat = self.pool2dto1(feat)
                feat = torch.flatten(feat, 1) #(b,512)
            else:
                if feat_hw==4:
                    feat = mlp_feats[n][-2]#(bs, 32, 8*8)
                    feat = feat.reshape(feat.shape[0], feat.shape[1], 8, 8)#(bs, 32, 8, 8)
                    feat = self.pool2dto4(feat)#(bs, 32, 4, 4)
                elif feat_hw==8:    
                    feat = mlp_feats[n][-1]#(bs, 8, 8*8)
                feat = feat.view(feat.shape[0], -1)#(bs,512)
            if n==0:
                feat1 = feat
            else:
                feat2 = feat
        # if self.fuse_r1:
        #     fuse_feats = torch.cat([feat1, feat2], dim=1)
        #     fuse_feats = self.fuse_fc(fuse_feats)
        #     feat1 = torch.cat([feat1, fuse_feats], dim=1)
        #     feat2 = torch.cat([feat2, fuse_feats], dim=1)
        return feat1, feat2
        

    def assemble_last_param_estimate(self, cam_param, pose_param, shape_param):
        params_estimate = torch.cat([cam_param[0], pose_param[0], shape_param[0]], dim=1)
        if self.itersup:
            self.cam_params.append(cam_param[0])
            self.pose_params.append(pose_param[0])
            self.shape_params.append(shape_param[0])
        return params_estimate

    def smpl_forward(self, shape, pose):
        # pose_rotmats, glob_rotmats = convert_theta_to_rotmats(pose[:, 3:], pose[:, :3]) 
        # target_rotmats = torch.cat([glob_rotmats, pose_rotmats], dim=1)
        # Convert pred pose to rotation matrices
        if pose.shape[-1] == 24*3:
            all_rotmats = batch_rodrigues(pose.contiguous().view(-1, 3))
            all_rotmats = all_rotmats.view(-1, 24, 3, 3)
        elif pose.shape[-1] == 24*6:
            all_rotmats = rot6d_to_rotmat(pose.contiguous()).view(-1, 24, 3, 3)

        glob_rotmats, pose_rotmats = all_rotmats[:, 0].unsqueeze(1), all_rotmats[:, 1:]
        # import ipdb; ipdb.set_trace()
        smpl_vertices, smpl_joints = self.smpl_model(body_pose=pose_rotmats.contiguous(),
                                global_orient=glob_rotmats.contiguous(),
                                betas=shape.contiguous(),
                                pose2rot=False)
        reposed_smpl_vertices, reposed_smpl_joints = self.smpl_model(betas=shape)
        return all_rotmats, smpl_vertices, smpl_joints, reposed_smpl_vertices

    def get_camK(self, pred_cam):
        cam_K_list = []
        for b in range(self.batch_size):
            pred_cam_wp = pred_cam[b]
            focal_length = pred_cam_wp[0]
            tx, ty = -pred_cam_wp[1]*focal_length, -pred_cam_wp[2]*focal_length
            cam_K =  get_intrinsics_matrix(tx*2, ty*2, focal_length)
            cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(self.device)
            cam_K_list.append(cam_K)
        # import ipdb; ipdb.set_trace()
        # cam_K = cam_K[None, :, :].expand(batch_size, -1, -1)
        cam_K = torch.stack(cam_K_list, dim=0) #(64,3,3)
        return cam_K

    def get_camK_fast(self, pred_cam):
        focal_length = pred_cam[:,0] #bs
        t_x, t_y= -pred_cam[:,1]*focal_length, -pred_cam[:,2]*focal_length#bs
        focal_length = focal_length[:,None, None]
        t_x = t_x[:,None, None]
        t_y = t_y[:,None, None]
        fill_zeros = torch.zeros_like(t_x)
        fill_ones = torch.ones_like(t_x)
        #cam_K: (bs,3,3)
        cam_K_row1 = torch.cat([focal_length,fill_zeros, t_x], dim=2)#(bs,1,3)
        cam_K_row2 = torch.cat([fill_zeros, focal_length, t_y], dim=2)#(bs,1,3)
        cam_K_row3 = torch.cat([fill_zeros, fill_zeros, fill_ones], dim=2)#(bs,1,3)

        cam_K = torch.cat([cam_K_row1, cam_K_row2, cam_K_row3], dim=1)
        return cam_K

    def pred_verts_to_IUV(self,  vertices, pred_cam, REGRESSOR_IMG_WH, cam_R=None, cam_T=None):
        # cam_K = self.get_camK(pred_cam)
        cam_K = self.get_camK_fast(pred_cam)#verified
        # import ipdb; ipdb.set_trace()
        renderer= P3dRenderer(self.batch_size, cam_K, cam_R=cam_R, device=self.device, img_wh=REGRESSOR_IMG_WH, 
                render_options={'pseg':0, 'depth':0, 'normal':0, 'iuv':1})    
        ######## forward
        unique_verts = [vertices[:,vid] for vid in renderer.to_1vertex_id] #(7829,bs,3)
        unique_verts = torch.stack(unique_verts, dim=1) #(bs, 7829, 3)
        #convert to camera coordinate R
        unique_verts = persepctive_project(unique_verts, renderer.cam_R_trans, cam_T=None, cam_K=None)

        #mesh
        verts_list = [unique_verts[nb] for nb in range(self.batch_size)] #(bs, 7829, 3)
        # desired_colors = [torch.ones((7829,3)).float().to(device) for _ in range(batch_size)] #(bs, 7829, 3)
        mesh_batch = Meshes(verts=verts_list, faces=renderer.faces_list, textures =  Textures(verts_rgb=self.IUVnorm_list))##norm I to [0,1]?

        cam_T = renderer.rectify_cam_T(None)
        cam_T[0,2] = 5
        cameras = OrthographicCameras(#PerspectiveCameras(
            focal_length=((renderer.focal_x, renderer.focal_y),),
            principal_point=((renderer.t_x, renderer.t_y),),
            # K=torch.from_numpy(K)[None].float(), 
            R = renderer.cam_R_render,
            # R = None,
            T = cam_T,
            device=self.device, 
            image_size=((-1, -1),)
            # image_size=((img_wh, img_wh),)
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=renderer.raster_settings)
        fragments = rasterizer(mesh_batch)
        # import ipdb; ipdb.set_trace()
        colors = mesh_batch.sample_textures(fragments)#(BS, H, W, 1, 4)
        images = hard_rgb_blend(colors, fragments, self.blendparam)# (BS, H, W, 4)
        I_map = images[:,:,:,0]#24 unique normalized
        # U_map = images[:,:,:,1]#nearest, require interpolation?
        # V_map = images[:,:,:,2]#nearest, require interpolation?
        # import ipdb; ipdb.set_trace()
        return I_map, images[:,:,:,1:3]

    def forward_align_pr(self, pred_params, feat_wh, feat_to_concat, align_iuv_j2d=[1,1]):
        pred_cam_up, pred_pose, pred_shape =  pred_params[:, :3], pred_params[:, 3:3 + 24*6], pred_params[:, 3 + 24*6:]
        # PREDICTED VERTICES AND JOINTS
        pred_pose_rotmats, pred_vertices, pred_joints_all, pred_reposed_vertices = self.smpl_forward(pred_shape, pred_pose)
        #j2d
        if align_iuv_j2d[1]:
            pred_joints_h36m = pred_joints_all[:, config.ALL_JOINTS_TO_H36M_MAP, :]
            pred_joints_h36mlsp = pred_joints_h36m[:, config.H36M_TO_J14, :]
            pred_joints_coco = pred_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
            pred_joints2d_coco = orthographic_project_torch(pred_joints_coco, pred_cam_up) #[-1,1]
            self.aligned_j2d.append(pred_joints2d_coco)
            joints2D_label = (2.0*self.gt_joints2d_coco) / config.REGRESSOR_IMG_WH - 1.0  # normalising j2d label [-1,1]
            offset_joints2d_coco = pred_joints2d_coco.detach() - joints2D_label #(bs,17,2)#detach in v1
            offset_joints2d_coco = offset_joints2d_coco.view(offset_joints2d_coco.shape[0], -1)
            feat_to_concat = torch.cat([feat_to_concat, offset_joints2d_coco], dim=1)
        #IUV
        if align_iuv_j2d[0]:
            pred_I, pred_UV = self.pred_verts_to_IUV(pred_vertices, pred_cam_up, feat_wh)
            self.aligned_i.append(pred_I)
            self.aligned_uv.append(pred_UV)

            gt_I = nnf.interpolate((self.gt_IUV[:,:,:,0]*24)[:,None], size=(feat_wh,feat_wh), mode='nearest').squeeze(dim=1)
            diff_I = nn.L1Loss(reduce=False)(pred_I, gt_I)
            diff_I = diff_I/(diff_I.detach()+EPS)#(bs,8,8)
            if self.gt_IUV_mask is not None:
                gt_I_mask = nnf.interpolate(self.gt_IUV_mask[:,None], size=(feat_wh,feat_wh), mode='nearest').squeeze(dim=1)
                diff_I = diff_I*gt_I_mask
                # import ipdb; ipdb.set_trace()
            diff_I = diff_I.view(diff_I.shape[0], -1)
            feat_to_concat = torch.cat([feat_to_concat, diff_I], dim=1)
        return feat_to_concat

    def forward(self, input):
        self.cam_params, self.pose_params, self.shape_params = [], [], []
        self.aligned_j2d = []
        self.aligned_i = []
        self.aligned_uv = []

        in_channel_split = self.in_channels[0]
        # modality 1
        feat1, _ , _ = self.image_encoder1(input[:,:in_channel_split,:,:])
        # modality 2
        feat2, _, _ = self.image_encoder2(input[:,in_channel_split:,:,:])
        
        mlp_feats1 = self.reduce_dim(
            feat1.view(feat1.shape[0], feat1.shape[1],-1),  #(bs,512,8*8)
            module_name='m1conv')[-2:]
        mlp_feats2 = self.reduce_dim(
            feat2.view(feat2.shape[0], feat2.shape[1],-1),  #(bs,512,8*8)
            module_name='m2conv')[-2:]
        # import ipdb; ipdb.set_trace()

        #reconstruct
        if self.recon:
            self.recon2 = self.decoder1_2(feat1)
            self.recon1 = self.decoder2_1(feat2)
        
        
        # regress 1
        feat1_1, feat2_1 = self.forward_feat(feat1, feat2, [mlp_feats1,mlp_feats2], feat_hw=self.feat_size[0]) 

        cam_param, pose_param, shape_param = self.ief_module_1_m1(feat1_1, params_estimate=None)
        params_estimate_1 = self.assemble_last_param_estimate(cam_param, pose_param, shape_param)
        
        cam_param, pose_param, shape_param = self.ief_module_1_m2(feat2_1, params_estimate=None)
        params_estimate_2 = self.assemble_last_param_estimate(cam_param, pose_param, shape_param)

        #regress 2
        feat1_2, feat2_2 = self.forward_feat(feat1, feat2,  [mlp_feats1,mlp_feats2], feat_hw=self.feat_size[1]) #(bs,8*8*8)
        if sum(self.add_channels[0][0])>0:
            feat1_2 = self.forward_align_pr(params_estimate_1, self.feat_size[1], 
                    feat_to_concat=feat1_2, align_iuv_j2d= self.add_channels[0][0])
        
        if sum(self.add_channels[0][1])>0:
            feat2_2 = self.forward_align_pr(params_estimate_2, self.feat_size[1],
                    feat_to_concat=feat2_2, align_iuv_j2d= self.add_channels[0][1])
        
        # import ipdb; ipdb.set_trace()
        # print('Finish forward', datetime.now().strftime("%m%d%H%M%S"))        
        cam_param, pose_param, shape_param = self.ief_module_2_m1(feat1_2, params_estimate=params_estimate_1)
        params_estimate_1 = self.assemble_last_param_estimate(cam_param, pose_param, shape_param)
        
        cam_param, pose_param, shape_param = self.ief_module_2_m2(feat2_2, params_estimate=params_estimate_2)
        params_estimate_2 = self.assemble_last_param_estimate(cam_param, pose_param, shape_param)

        #regress 3
        if self.add_ief:
            feat1_a3, feat2_a3 = self.forward_feat(feat1, feat2, [mlp_feats1,mlp_feats2], feat_hw=self.feat_size[2]) #(bs,8*8*8)
            if sum(self.add_channels[1][0])>0:
                feat1_a3 = self.forward_align_pr(params_estimate_1, self.feat_size[2], 
                        feat_to_concat=feat1_a3, align_iuv_j2d= self.add_channels[1][0])
            
            if sum(self.add_channels[1][1])>0:
                feat2_a3 = self.forward_align_pr(params_estimate_2, self.feat_size[2],
                        feat_to_concat=feat2_a3, align_iuv_j2d= self.add_channels[1][1])
            
            cam_param, pose_param, shape_param = self.ief_module_3_m1(feat1_a3, params_estimate=params_estimate_1)
            params_estimate_1 = self.assemble_last_param_estimate(cam_param, pose_param, shape_param)
            
            cam_param, pose_param, shape_param = self.ief_module_3_m2(feat2_a3, params_estimate=params_estimate_2)
            params_estimate_2 = self.assemble_last_param_estimate(cam_param, pose_param, shape_param)
            # import ipdb; ipdb.set_trace()

        feat1_3, feat2_3 = self.forward_feat(feat1, feat2, [mlp_feats1,mlp_feats2], feat_hw=self.feat_size[2]) #(bs,8*8*8)
        if sum(self.add_channels[1][0])>0:
            feat1_3 = self.forward_align_pr(params_estimate_1, self.feat_size[2], 
                    feat_to_concat=feat1_3, align_iuv_j2d= self.add_channels[1][0])
        
        if sum(self.add_channels[1][1])>0:
            feat2_3 = self.forward_align_pr(params_estimate_2, self.feat_size[2],
                    feat_to_concat=feat2_3, align_iuv_j2d= self.add_channels[1][1])
        
        r3_feat = torch.cat([feat1_3, feat2_3], dim=1)
        # 
        #final regression#TODO if [[1,1],[1,1]] optimize the code
        cam_param, pose_param, shape_param = self.ief_module_fusion(r3_feat, params_estimate_1, params_estimate_2)
        if self.add_fuser:
            params_estimate = self.assemble_last_param_estimate(cam_param, pose_param, shape_param)
            #align?same as r3
            feat1_4, feat2_4 = self.forward_feat(feat1, feat2, [mlp_feats1,mlp_feats2], feat_hw=self.feat_size[2]) #(bs,8*8*8)
            if sum(self.add_channels[1][0])>0:
                feat1_4 = self.forward_align_pr(params_estimate, self.feat_size[2], 
                    feat_to_concat=feat1_4, align_iuv_j2d= self.add_channels[1][0])
            if sum(self.add_channels[1][1])>0:
                feat2_4 = self.forward_align_pr(params_estimate, self.feat_size[2], 
                    feat_to_concat=feat2_4, align_iuv_j2d= self.add_channels[1][1])
            r4_feat = torch.cat([feat1_4, feat2_4], dim=1)
            cam_param, pose_param, shape_param = self.ief_module_fusion(r4_feat, params_estimate, params_estimate)
            # import ipdb; ipdb.set_trace()
        self.cam_params.append(cam_param[0])
        self.pose_params.append(pose_param[0])
        self.shape_params.append(shape_param[0])

        return self.cam_params, self.pose_params, self.shape_params

    def set_align_target(self, joints_dict, IUV_dict, align_aug_IUV=1, align_aug_j2d=1):
        if align_aug_IUV:
            self.regressor.gt_IUV = IUV_dict["aug_IUV"]  
            self.regressor.gt_IUV_mask = (IUV_dict["aug_IUV"]==IUV_dict["target_IUV"]).all(dim=3).float()#(bs, h, w), IUV(bs,h,w,3)
        else:
            self.regressor.gt_IUV = IUV_dict["target_IUV"]
        if align_aug_j2d:
            self.regressor.gt_joints2d_coco = joints_dict["aug_joints2d_coco"]  
        else:
            self.regressor.gt_joints2d_coco = joints_dict["target_joints2d_coco"]

    def set_align_target_infer(self, IUV, joints2D):
        self.regressor.gt_IUV = IUV
        self.regressor.gt_joints2d_coco = joints2D
        


class MultiscaleAlignRegressor(FuseAlignRegressor):#only align at last regressor
    def __init__(self,batch_size, 
                smpl_model,
                device,
                resnet_in_channels=1,
                itersup=False,
                filter_channels =[512, 256, 32, 8],
                encoddrop=False):
        super(MultiscaleAlignRegressor, self).__init__(batch_size, 
                smpl_model,
                device)
        self.itersup = itersup
        num_pose_params = 24*6
        num_output_params = 3 + num_pose_params + 10
        self.smpl_model = smpl_model
        self.batch_size = batch_size
        self.device = device
        self.image_encoder = resnet18(in_channels=resnet_in_channels, drop = encoddrop,
                                        pretrained=False, pool_out=False)
        
        

        self.add_mlp(filter_channels=filter_channels)
        self.ief_module_1 = SpatialIEFModule([512, 512],
                                    512,
                                    num_output_params,
                                    iterations=1)
        self.ief_module_2 = SpatialIEFModule([512, 512],
                                    512,
                                    num_output_params,
                                    iterations=1)
        ief_channel3 = 512 +  8*8 + 17*2
        self.add_channels =[1,1]
        self.ief_module_3 = SpatialIEFModule([ief_channel3, ief_channel3],
                                    ief_channel3,
                                    num_output_params,
                                    iterations=1)

    def forward(self, input, channel_used=[0,2,3]):
        input_feats = self.image_encoder(input)[0]
        # import ipdb; ipdb.set_trace()
        input_feats = input_feats.view(input_feats.shape[0], input_feats.shape[1],-1)#(bs,512,8*8)
        # import ipdb; ipdb.set_trace()
        out_feats = self.reduce_dim(input_feats)#only use the last one
        params_estimate = None

        cam_params, pose_params, shape_params = [], [], []
        for n,c in enumerate(channel_used):
            feat = out_feats[c] #(bs,c,8*8)
            feat = feat.reshape(feat.shape[0], feat.shape[1], 8, 8)
            target_wh = np.sqrt(512//feat.shape[1]).astype('int')
            if not target_wh==8:
                feat = nn.AdaptiveAvgPool2d((target_wh, target_wh))(feat)
            feat = feat.view(feat.shape[0], -1)#(bs,512)
            # import ipdb; ipdb.set_trace()
            if n==2:#align
                feat = self.forward_align_pr(params_estimate, 8, feat_to_concat=feat, align_iuv_j2d= [1,1])
                
            cam_param, pose_param, shape_param = self._modules['ief_module_' + str(n+1)](feat, params_estimate=params_estimate)
            params_estimate = torch.cat([cam_param[0], pose_param[0], shape_param[0]], dim=1)
            cam_params.append(cam_param[0])
            pose_params.append(pose_param[0])
            shape_params.append(shape_param[0])
        
        if not self.itersup:
            cam_params, pose_params, shape_params = [cam_params[-1]],  [pose_params[-1]],  [shape_params[-1]]
        return cam_params, pose_params, shape_params