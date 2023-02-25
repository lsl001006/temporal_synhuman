import os
import torch
import numpy as np

from model.model import Build_SMPL
import eval.metrics as metrics
from eval.predict_densepose import setup_densepose_silhouettes, predict_silhouette_densepose
from eval.predict_joints2d import setup_j2d_predictor, predict_joints2D_new
import utils.label_conversions as LABELCONFIG
from utils.smpl_utils import smpl_forward
from utils.proxyrep_utils import convert_to_proxyfeat_batch
from utils.image_utils import crop_and_resize_iuv_joints2D_torch
from .vis_prediction import VisMesh
from einops import rearrange

import sys, configs
sys.path.append(f"{configs.DETECTRON2_PATH}/projects/DensePose")
from densepose.vis.extractor import DensePoseResultExtractor

class testHMRImg():
    def __init__(self, 
                regressor, 
                smpl_model, 
                device, 
                args):
        self.regressor = regressor
        self.smpl_model = smpl_model
        self.device = device
        self.pr_mode = args.pr
        self.pr_wh = configs.REGRESSOR_IMG_WH
        self.eval_j14 = args.j14
        self.batch_size = args.batch_size*configs.SEQLEN
        if args.visnum_per_batch:
            self.visdir = f'{configs.VIS_DIR}/{args.data}/{args.ckpt}'
            self.visnum_per_batch = args.visnum_per_batch
            self.vispr = args.vispr
        
        if args.wgender: #only SSP3D has valid gender #NOT valid for now
            self.smpl_model_male = Build_SMPL(args.batch_size, self.device, gender='male')
            self.smpl_model_female = Build_SMPL(args.batch_size, self.device, gender='female')
        self.wgender = args.wgender
        
        self.set_detector2D(args)


    def set_detector2D(self, args):
        # Set-up proxy representation predictors.
        self.silhouette_predictor = setup_densepose_silhouettes()
        self.extractor = DensePoseResultExtractor()
        self.joints2D_predictor = setup_j2d_predictor()
        # Record if 2D detection not good
        self.skip_idx = []
        self.black_idx = []
        self.bbox_scale = args.bbox_scale
    
    def get_proxy_rep(self, samples_batch):#batch_size=1
        image = samples_batch['image'][0].numpy()
        center = samples_batch['center'][0].numpy()
        scale = samples_batch['scale'][0].item()
        IUV = predict_silhouette_densepose(image, self.silhouette_predictor, self.extractor, center, scale)#(h,w,3) torch
        joints2D = predict_joints2D_new(image, self.joints2D_predictor, center, scale)#(17,3) numpy
        
        if (IUV is None) or (joints2D is None):
            self.skip_idx.append(samples_batch['n_sample'])
            return None, None
        else:
            IUV, joints2D, cropped_img = crop_and_resize_iuv_joints2D_torch(IUV, 
                                                                configs.REGRESSOR_IMG_WH, 
                                                                joints2D=joints2D, 
                                                                image=image, 
                                                                bbox_scale_factor=self.bbox_scale)
            if hasattr(self, 'vis'):
                self.vis.crop_img = cropped_img
            bodymask = (24*IUV[:,:,0]).round().cpu().numpy()
            fg_ids = np.argwhere(bodymask != 0) 
            if fg_ids.shape[0]<256:
                self.black_idx.append(samples_batch['n_sample'])
                return None, None
            
            return IUV[None].to(self.device), torch.tensor(joints2D)[:,:2][None].to(self.device).int()

    def forward_batch(self, samples_batch):
        IUV, joints2D = self.get_proxy_rep(samples_batch) #(bs,h,w,3),(bs,17,2) 
        
        if IUV is None:
            return None, None, None, None
        
        proxy_rep = convert_to_proxyfeat_batch(IUV, joints2D)
        #
        if self.vispr and hasattr(self, 'vis'):
            self.vis.iuv = IUV
            self.vis.j2d = joints2D
        #
        with torch.no_grad():
            if hasattr(self.regressor, 'add_channels'):
                if torch.tensor(self.regressor.add_channels).bool().any().item():
                    self.regressor.set_align_target_infer(IUV, joints2D)
            
            pred_cam_wp_list, pred_pose_list, pred_shape_list = self.regressor(proxy_rep)
                    
            _, pred_vertices, pred_joints_all, pred_reposed_vertices, _ = smpl_forward(
                pred_shape_list[-1], 
                pred_pose_list[-1],
                self.smpl_model)
            
            pred_joints_h36m = pred_joints_all[:, LABELCONFIG.ALL_JOINTS_TO_H36M_MAP, :]
            pred_joints_h36mlsp = pred_joints_h36m[:, LABELCONFIG.H36M_TO_J17, :]
            
            
        return pred_vertices, pred_reposed_vertices, pred_joints_h36mlsp, pred_cam_wp_list[-1]

    def get_target(self, samples_batch):
        if self.withshape:
            target_pose = samples_batch['pose'].to(self.device).float()#bx72
            target_shape = samples_batch['shape'].to(self.device).float()#bx10
            _, target_vertices, joints_all, target_reposed_vertices, _ = smpl_forward(
                target_shape, 
                target_pose,
                self.smpl_model)

            target_joints_h36m = joints_all[:, LABELCONFIG.ALL_JOINTS_TO_H36M_MAP, :]
            target_joints_h36mlsp = target_joints_h36m[:, LABELCONFIG.H36M_TO_J17, :]
        else:
            target_joints_h36mlsp = samples_batch['j17_3d'].to(self.device).float()
            target_vertices, target_reposed_vertices = None, None
        
        return target_vertices, target_reposed_vertices, target_joints_h36mlsp

    def update_metrics_batch(self, samples_batch, printt=False):
        pred_vertices, pred_reposed_vertices, pred_joints_h36mlsp, pred_cam_wp = self.forward_batch(samples_batch)
        """
        pred_vertices         bs*seql, 6890, 3
        pred_reposed_vertices bs*seql, 6890, 3
        pred_joints_h36mlsp   bs*seql, 17, 3
        pred_cam_wp           bs*seql, 3
        """
        
        if pred_vertices is None:
            return
        target_vertices, target_reposed_vertices, target_joints_h36mlsp = self.get_target(samples_batch)
        if hasattr(self, 'vis'):
            self.vis.forward_verts(pred_vertices, target_vertices, pred_cam_wp, samples_batch['n_sample'])

        if self.eval_j14:
            target_joints_h36mlsp = target_joints_h36mlsp[:, LABELCONFIG.J17_TO_J14, :]
            pred_joints_h36mlsp = pred_joints_h36mlsp[:, LABELCONFIG.J17_TO_J14, :]
        
        # re-center
        pred_joints_h36mlsp = pred_joints_h36mlsp - (pred_joints_h36mlsp[:,[2],:]+pred_joints_h36mlsp[:,[3],:])/2.
        target_joints_h36mlsp = target_joints_h36mlsp - (target_joints_h36mlsp[:,[2],:]+target_joints_h36mlsp[:,[3],:])/2.
        # no use?
        # pred_vertices = pred_vertices - (pred_joints_h36mlsp[:,[2],:]+pred_joints_h36mlsp[:,[3],:])/2
        # target_vertices = target_vertices - (target_joints_h36mlsp[:,[2],:]+target_joints_h36mlsp[:,[3],:])/2
        
        #metric
        batch_size = pred_joints_h36mlsp.shape[0]
        if 'mpjpe_pa' in self.metrics_track:
            mpjpe_pa = metrics.cal_mpjpe_pa(pred_joints_h36mlsp.detach().cpu().numpy(), 
                                            target_joints_h36mlsp.detach().cpu().numpy())
            self.mpjpe_pa.update(mpjpe_pa, n=batch_size)
            if printt:
                print(f'mpjpe_pa for {self.mpjpe_pa.count}: {self.mpjpe_pa.average()}')
        
        if 'mpjpe_sc' in self.metrics_track:
            mpjpe_sc = metrics.cal_mpjpe_sc(pred_joints_h36mlsp.detach().cpu().numpy(), 
                                            target_joints_h36mlsp.detach().cpu().numpy())
            self.mpjpe_sc.update(mpjpe_sc, n=batch_size)
            if printt:
                print(f'mpjpe_sc for {self.mpjpe_sc.count}: {self.mpjpe_sc.average()}')
        
        if 'mpjpe' in self.metrics_track:
            mpjpe = metrics.cal_mpjpe(pred_joints_h36mlsp.detach().cpu().numpy(), 
                                    target_joints_h36mlsp.detach().cpu().numpy())
            self.mpjpe.update(mpjpe, n=batch_size)
            if printt:
                print(f'mpjpe for {self.mpjpe.count}: {self.mpjpe.average()}')

        if 'pve-ts_sc' in self.metrics_track:
            pve = metrics.cal_pve_ts_sc(pred_reposed_vertices.detach().cpu().numpy(), 
                                        target_reposed_vertices.detach().cpu().numpy())
            self.pve_t_sc.update(pve,  n=batch_size)
            if printt:
                print(f'pve_t_sc for {self.pve_t_sc.count}: {self.pve_t_sc.average()}') 

        if 'pve' in self.metrics_track:
            pve = metrics.cal_pve(pred_vertices.detach().cpu().numpy(), 
                                target_vertices.detach().cpu().numpy())
            self.pve.update(pve,  n=batch_size)
            if printt:
                print(f'pve for {self.pve.count}: {self.pve.average()}') 

        if 'pve_pa' in self.metrics_track:
            pve_pa = metrics.cal_pve_pa(pred_vertices.detach().cpu().numpy(), 
                                        target_vertices.detach().cpu().numpy())
            self.pve_pa.update(pve_pa,  n=batch_size)
            if printt:
                print(f'pve_pa for {self.pve_pa.count}: {self.pve_pa.average()}') 

        if 'pck_pa' in self.metrics_track:
            pck_pa, auc_pa = metrics.cal_pck_pa(pred_joints_h36mlsp.detach().cpu().numpy(), 
                                                target_joints_h36mlsp.detach().cpu().numpy())
            self.pck.update(pck_pa, n=batch_size)
            self.auc.update(auc_pa, n=batch_size)
            if printt:
                print(f'pck_pa for {self.pck.count}: {self.pck.average()}')
                print(f'auc_pa for {self.auc.count}: {self.auc.average()}')
        
        

        return 

    def test(self, dataloader, withshape, metrics_track, eval_ep, print_freq=10):
        self.withshape = withshape
        self.metrics_track = metrics_track
        if hasattr(self, 'visdir'):
            self.visdir = f'{self.visdir}_ep{eval_ep}'
            if not os.path.isdir(self.visdir):
                os.makedirs(self.visdir)
            self.vis = VisMesh(self.visdir, self.batch_size, self.device, self.visnum_per_batch)

        
        self.mpjpe_pa = metrics.AverageMeter()
        self.mpjpe_sc = metrics.AverageMeter()
        self.mpjpe = metrics.AverageMeter()
        self.pck = metrics.AverageMeter()
        self.auc = metrics.AverageMeter()
        self.pve = metrics.AverageMeter()
        self.pve_pa = metrics.AverageMeter()
        self.pve_t_sc = metrics.AverageMeter()

        self.regressor.eval()
        for n_sample, samples_batch in enumerate(dataloader):
            samples_batch['n_sample'] = n_sample
            print(f'----idx:{n_sample}----')
            self.update_metrics_batch(samples_batch, printt=(n_sample%print_freq==0))
        
            
        # Complete
        if self.mpjpe_pa.count:  
            print(f'mpjpe_pa for {self.mpjpe_pa.count}: {self.mpjpe_pa.average()}')
        if self.mpjpe_sc.count:
            print(f'mpjpe_sc for {self.mpjpe_sc.count}: {self.mpjpe_sc.average()}')
        if self.mpjpe.count:
            print(f'mpjpe for {self.mpjpe.count}: {self.mpjpe.average()}')
        if self.pck.count:
            print(f'pck for {self.pck.count}: {self.pck.average()}')
        if self.auc.count:
            print(f'auc for {self.auc.count}: {self.auc.average()}')
        if self.pve_t_sc.count:
            print(f'pve_t_sc for {self.pve_t_sc.count}: {self.pve_t_sc.average()}')  
        if self.pve.count:
            print(f'pve for {self.pve.count}: {self.pve.average()}')  
        if self.pve_pa.count:
            print(f'pve_pa for {self.pve_pa.count}: {self.pve_pa.average()}')  
                     
        return self.mpjpe_pa.average()

class testHMRPr(testHMRImg):
    def set_detector2D(self, args):
        pass

    def get_proxy_rep(self, samples_batch):
        images = samples_batch['image'].numpy()
        IUV = samples_batch['iuv'].to(self.device)
        joints2D = samples_batch['j2d'].int().to(self.device)
        
        IUV = rearrange(IUV, 'b1 b2 h w c -> (b1 b2) h w c')
        joints2D = rearrange(joints2D, 'b1 b2 a b -> (b1 b2) a b')
        images = rearrange(images, 'b1 b2 h w c -> (b1 b2) h w c' )
        
        if hasattr(self, 'vis'):
            self.vis.crop_img = images

        return IUV, joints2D