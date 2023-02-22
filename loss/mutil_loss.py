import torch
import torch.nn as nn
import numpy as np
import configs
from .iuv_loss import body_iuv_losses

class TotalWeightedLoss(nn.Module):
    def __init__(self,
                task_criterion,
                losses_on=['latent', 'reconstruct1', 'reconstruct2'],
                var = False,
                init_loss_weights=None,
                reduction='mean',
                pr = 'bj',
                recon_weight=0,
                eps=1e-6):
        super(TotalWeightedLoss, self).__init__()

        self.losses_on = losses_on
        assert reduction in ['mean', 'sum'], "Invalid reduction for loss."

        if init_loss_weights is not None:
            # Initialise log variances using given init loss weights 
            init_latent_log_var = -np.log(init_loss_weights['latent'] + eps)
            init_reconstruct1_log_var = -np.log(init_loss_weights['reconstruct1'] + eps)
            init_reconstruct2_log_var = -np.log(init_loss_weights['reconstruct2'] + eps)    
        else:
            # Initialise log variances to 0.
            init_latent_log_var = 0
            init_reconstruct1_log_var = 0
            init_reconstruct2_log_var = 0

        self.latent_log_var = nn.Parameter(torch.tensor(init_latent_log_var).float(),
                                          requires_grad=False)
        self.reconstruct1_log_var = nn.Parameter(torch.tensor(init_reconstruct1_log_var).float(),
                                          requires_grad=False)
        self.reconstruct2_log_var = nn.Parameter(torch.tensor(init_reconstruct2_log_var).float(),
                                          requires_grad=False)
        if 'latent' in losses_on:
            self.latent_log_var.requires_grad = True
            self.latent_loss = nn.MSELoss(reduction=reduction)
        if 'reconstruct1' in losses_on:
            self.reconstruct1_log_var.requires_grad = True
            self.reconstruct1_loss = nn.MSELoss(reduction=reduction)#j2Dheatmap
        if 'reconstruct2' in losses_on:
            self.reconstruct2_log_var.requires_grad = True
            if pr=='bj':
                self.reconstruct2_loss = nn.BCELoss(reduction=reduction)#binrayMask
            elif pr=='iuvj':
                self.reconstruct2_loss = nn.MSELoss(reduction=reduction)
        
        self.task_criterion = task_criterion
        self.var = var
        self.recon_weight = recon_weight
    def forward(self, target_dict_for_loss, pred_dict_for_loss, z, img1, img2, recon_img1, recon_img2):
        total_loss = 0.

        task_loss, loss_dict = self.task_criterion(target_dict_for_loss, pred_dict_for_loss)
        total_loss += task_loss
        loss_dict['task'] = task_loss
        # import ipdb; ipdb.set_trace()
        total_loss, loss_dict = self.reconstruct(z, img1, img2, recon_img1, recon_img2, total_loss, loss_dict)
        return total_loss, loss_dict

    def reconstruct(self, z, img1, img2, recon_img1, recon_img2):
        reconstruct_loss = 0.
        reconstruct_loss_dict = {}
        if ('latent' in self.losses_on):
            if len(z)==2:
                z1, z2 = z
                if np.random.rand()>0.5:
                    latent_loss = self.latent_loss(z1, z2.detach())
                else:
                    latent_loss = self.latent_loss(z1.detach(), z2)
                if self.var:
                    reconstruct_loss_dict['latent'] = latent_loss * torch.exp(-self.latent_log_var) + self.latent_log_var
                else:
                    reconstruct_loss_dict['latent'] = latent_loss
            else:
                loss_dict['latent'] = torch.tensor(0.).to(self.latent_log_var.device)
            
            reconstruct_loss += reconstruct_loss_dict['latent']
        
        if 'reconstruct1' in self.losses_on:
            # import ipdb; ipdb.set_trace()
            recon1_loss = self.reconstruct1_loss(recon_img1, img1)
            if self.var:
                reconstruct_loss_dict['reconstruct1'] = recon1_loss * torch.exp(-self.reconstruct1_log_var) + self.reconstruct1_log_var     
            else:
                reconstruct_loss_dict['reconstruct1'] = recon1_loss
            reconstruct_loss += self.recon_weight*reconstruct_loss_dict['reconstruct1']     
        
        if 'reconstruct2' in self.losses_on:
            recon2_loss = self.reconstruct2_loss(recon_img2, img2)
            if self.var:
                reconstruct_loss_dict['reconstruct2'] = recon2_loss * torch.exp(-self.reconstruct2_log_var)+ self.reconstruct2_log_var
            else:
                reconstruct_loss_dict['reconstruct2'] = recon2_loss
            reconstruct_loss += self.recon_weight*reconstruct_loss_dict['reconstruct2']     
        # import ipdb; ipdb.set_trace()
        return reconstruct_loss, reconstruct_loss_dict

class HomoscedasticUncertaintyWeightedMultiTaskLoss(nn.Module):
    """
    Multi-task loss function. 
    Loss weights are learnt via homoscedastic uncertainty (Kendall et al.) 
    Losses can be applied on 3D vertices, 2D joints (projected), 3D joints, SMPL pose
    parameters (in the form of rotation matrices) and SMPL shape parameters.
    """
    def __init__(self,
                 init_loss_weights,
                 reduction='mean',
                 eps=1e-6,
                 var=False):
        """
        :param losses_on: List of outputs to apply losses on.
        Subset of ['verts', 'joints2D', 'joints3D', 'pose_params', 'shape_params'].
        :param init_loss_weights: Initial multi-task loss weights.
        :param reduction: 'mean' or 'sum'
        :param eps: small constant
        """
        super(HomoscedasticUncertaintyWeightedMultiTaskLoss, self).__init__()
        self.var = var
        self.losses_on = list(init_loss_weights.keys())
        assert reduction in ['mean', 'sum'], "Invalid reduction for loss."

        if 'verts' in self.losses_on:
            init_verts_log_var = -np.log(init_loss_weights['verts'] + eps)
            self.verts_log_var = nn.Parameter(torch.tensor(init_verts_log_var).float(),
                                          requires_grad=True)
            self.verts_loss = nn.MSELoss(reduction=reduction)
        
        if 'joints2D' in self.losses_on:
            init_joints2D_log_var = -np.log(init_loss_weights['joints2D'] + eps)
            self.joints2D_log_var = nn.Parameter(torch.tensor(init_joints2D_log_var).float(),
                                             requires_grad=True)
            self.joints2D_loss = nn.MSELoss(reduction=reduction)
        
        if 'joints3D' in self.losses_on:
            init_joints3D_log_var = -np.log(init_loss_weights['joints3D'] + eps)
            self.joints3D_log_var = nn.Parameter(torch.tensor(init_joints3D_log_var).float(),
                                             requires_grad=True)
            self.joints3D_loss = nn.MSELoss(reduction=reduction)

        if 'shape_params' in self.losses_on:
            init_shape_params_log_var = -np.log(init_loss_weights['shape_params'] + eps)
            self.shape_params_log_var = nn.Parameter(torch.tensor(init_shape_params_log_var).float(),
                                                 requires_grad=True)
            self.shape_params_loss = nn.MSELoss(reduction=reduction)

        if 'pose_params' in self.losses_on:
            init_pose_params_log_var = -np.log(init_loss_weights['pose_params'] + eps)
            self.pose_params_log_var = nn.Parameter(torch.tensor(init_pose_params_log_var).float(),
                                                requires_grad=True)
            self.pose_params_loss = nn.MSELoss(reduction=reduction)
        
        if 'iuv' in self.losses_on:
            init_iuv_log_var = -np.log(init_loss_weights['iuv'] + eps)  
            self.iuv_log_var = nn.Parameter(torch.tensor(init_iuv_log_var).float(),
                                                 requires_grad=True)
        

    def forward(self, labels, outputs):
        total_loss = 0.
        loss_dict = {}

        if 'verts' in self.losses_on:
            verts_loss = self.verts_loss(outputs['verts'], labels['verts'])
            if self.var:
                loss_dict['verts'] = verts_loss * torch.exp(-self.verts_log_var) + self.verts_log_var
            else:
                loss_dict['verts'] = verts_loss
            total_loss += loss_dict['verts'] 
            

        if 'joints2D' in self.losses_on:
            joints2D_label = labels['joints2D']
            joints2D_pred = outputs['joints2D']

            if 'vis' in labels.keys():
                vis = labels['vis']  # joint visibility label - boolean
                joints2D_label = joints2D_label[vis, :]
                joints2D_pred = joints2D_pred[vis, :]

            joints2D_label = (2.0*joints2D_label) / configs.REGRESSOR_IMG_WH - 1.0  # normalising j2d label [-1,1]
            joints2D_loss = self.joints2D_loss(joints2D_pred, joints2D_label)
            if self.var:
                loss_dict['joints2D'] = joints2D_loss * torch.exp(-self.joints2D_log_var) + self.joints2D_log_var
            else:
                loss_dict['joints2D'] = joints2D_loss
            total_loss += loss_dict['joints2D']

        if 'joints3D' in self.losses_on:
            joints3D_loss = self.joints3D_loss(outputs['joints3D'], labels['joints3D'])
            if self.var:
                loss_dict['joints3D'] =  joints3D_loss * torch.exp(-self.joints3D_log_var) + self.joints3D_log_var
            else:
                loss_dict['joints3D'] =  joints3D_loss
            total_loss += loss_dict['joints3D']

        if 'shape_params' in self.losses_on:
            shape_params_loss = self.shape_params_loss(outputs['shape_params'],
                                                       labels['shape_params'])
            if self.var:
                loss_dict['shape_params'] = shape_params_loss * torch.exp(-self.shape_params_log_var) + self.shape_params_log_var
            else:
                loss_dict['shape_params'] = shape_params_loss
            total_loss += loss_dict['shape_params']
            
        if 'pose_params' in self.losses_on:
            pose_params_loss = self.pose_params_loss(outputs['pose_params_rot_matrices'],
                                                     labels['pose_params_rot_matrices'])
            if self.var:
                loss_dict['pose_params'] = pose_params_loss * torch.exp(-self.pose_params_log_var)+ self.pose_params_log_var
            else:
                loss_dict['pose_params'] = pose_params_loss
            total_loss += loss_dict['pose_params']
        
        # BUG No exist
        if 'silhouette' in self.losses_on:
            silhouette_loss = self.silhouette_loss(outputs['silhouettes'], labels['silhouettes'])
            if self.var:
                loss_dict['silhouette'] = silhouette_loss * torch.exp(-self.silhouette_log_var)+ self.silhouette_log_var
            else:
                loss_dict['silhouette'] = silhouette_loss
            total_loss += loss_dict['silhouette']
        ##added
        if ('iuv' in self.losses_on) and outputs['deconv_IUV'] is not None:
            
            iuv_loss = body_iuv_losses(outputs['deconv_IUV'], labels['IUV'])
            if self.var:
                loss_dict['iuv'] = iuv_loss * torch.exp(-self.iuv_log_var)+ self.iuv_log_var
            else:
                loss_dict['iuv'] = iuv_loss

            total_loss += loss_dict['iuv']
       
        return total_loss, loss_dict