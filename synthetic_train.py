import copy, logging
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
import configs

from eval.train_loss_and_metrics_tracker import LossMetricsTracker
from utils.checkpoint_utils import load_training_info_from_checkpoint
from utils.proxyrep_utils import convert_to_proxyfeat_batch
import utils.label_conversions as LABELCONFIG
from utils.cam_utils import orthographic_project_torch
from utils.smpl_utils import smpl_forward
from einops import rearrange
import torch.distributed as dist

logger = logging.getLogger()
# from smplx.lbs import batch_rodrigues
# from utils.rigid_transform_utils import rot6d_to_rotmat

class my_train_rendering():
    def __init__(self, args, renderSyn, regressor, device, train_dataloader, val_dataloader,
                criterion, losses_to_track, model_savedir, log_path, metrics_to_track, save_val_metrics):     
        
        self.regressor = regressor
        self.renderSyn = renderSyn
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.save_val_metrics = save_val_metrics
        # self.forward_function = self.forwardIterFinetune if args.finetune else self.forwardIter
        self.forward_function = self.forwardIter
        self.args = args
        
        # OPTIMIZER
        params = list(self.regressor.parameters()) + list(self.criterion.parameters())
        if args.optimizer=='adam':
            self.optimiser = torch.optim.Adam(params, lr=args.lr)
            self.scheduler = None
        elif args.optimizer=='sgd':
            self.optimiser = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimiser, args.epochs, eta_min=1e-5)

        # CKPT
        self.model_savedir = model_savedir
        load_logs = self.load_checkpoint(model_savedir, args)
        
        # METRICES
        self.metrics_tracker = LossMetricsTracker(losses_to_track=losses_to_track,
                                                        metrics_to_track=metrics_to_track,
                                                        img_wh=configs.REGRESSOR_IMG_WH,
                                                        log_path=log_path,
                                                        load_logs=load_logs,
                                                        current_epoch=self.start_epoch,
                                                        track_val=True)

    def load_checkpoint(self, model_savedir, args):
        """
        resume
        """
        if args.resume_from_epoch:
            checkpoint_path =  f'{model_savedir}/epoch{args.resume_from_epoch}.tar'
            if dist.get_rank() == 0:
                logger.info(f'Resuming from: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.regressor.load_state_dict(checkpoint['model_state_dict']) #not best????
            self.start_epoch, best_epoch, best_model_wts, self.best_epoch_val_metrics = \
                load_training_info_from_checkpoint(checkpoint, self.save_val_metrics)
            if args.resume_sametr:
                self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
                self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
            load_logs = True # FIXME if resume from existing checkpoints, then load_logs will be set True
        else:
            checkpoint = None
            self.start_epoch = 0
            self.best_epoch_val_metrics = {}
            # metrics that decide whether to save model after each epoch or not
            for metric in self.save_val_metrics:
                self.best_epoch_val_metrics[metric] = np.inf
            best_epoch = self.start_epoch
            best_model_wts = copy.deepcopy(self.regressor.state_dict())
            load_logs = False

        return load_logs

    def forwardIter(self, samples_batch, is_train):
        # print('Start iteration', datetime.now().strftime("%m%d%H%M%S"))

        with torch.no_grad():    
            # 获取真值        
            smpl_dict, iuv_dict, joints_dict, vertices_dict, depth_dict = self.renderSyn.render_forward(
                                            samples_batch['pose'].squeeze().to(self.device), #bx16x72 
                                            samples_batch['shape'].squeeze().to(self.device), #bx16x10
                                            is_train)

          # FINAL INPUT PROXY REPRESENTATION GENERATION WITH JOINT HEATMAPS
            # 这一步只传入了IUV和joints2d, 在函数内部处理joints2d_heatmap
            
            inter_represent = convert_to_proxyfeat_batch(
                iuv_dict["aug_iuv"], 
                joints_dict["aug_joints2d_coco"],
                joints2d_no_occluded_coco=joints_dict["joints2d_no_occluded_coco"].clone(), #clone required?
                pr_mode = self.args.pr)
            
        if self.args.debugging:
            print('----inter_represent----')
            print(f'shape:{inter_represent.shape}')
            print('-----------------------')
        
        if self.args.cra:
            assert hasattr(self.regressor, 'add_channels')
            if torch.tensor(self.regressor.add_channels).bool().any().item():
                self.regressor.set_align_target(joints_dict, iuv_dict)

        # gradients being computed from here on)
        # regressor 将image_encoder和regressor封装在一起
        
        pred_cam_wp_list, pred_pose_list, pred_shape_list = self.regressor(inter_represent) # (pi, theta, beta) in paper 3.3
        if self.args.debugging:
            print('----After Regressor----')
            print(f'pred_cam_wp_list:{pred_cam_wp_list[0].shape}, length:{len(pred_cam_wp_list)}')
            print(f'pred_pose_list:{pred_pose_list[0].shape}, length:{len(pred_pose_list)}')
            print(f'pred_shape_list:{pred_shape_list[0].shape}, length:{len(pred_shape_list)}')
            print('-----------------------')
        """
        pred_cam_wp_list: [(bs*3)]
        pred_pose_list: [(bs*144)] 144=24*6
        pred_shape_list: [(bs*10)]
        """
        loop_nlist = np.arange(len(pred_cam_wp_list)).tolist() if is_train else [-1]
        total_loss = 0.
        for index, nl in enumerate(loop_nlist):
            pred_cam_wp = pred_cam_wp_list[nl]
            pred_shape = pred_shape_list[nl]
            pred_pose = pred_pose_list[nl]
            
            # PREDICTED VERTICES AND JOINTS
            pred_pose_rotmats, pred_vertices, pred_joints_all, pred_reposed_vertices, _ = smpl_forward(
                                                                                                        pred_shape, 
                                                                                                        pred_pose,
                                                                                                        self.renderSyn.smpl_model)
            if self.args.debugging:
                print(f'----After {index}/{len(loop_nlist)} smpl_forward----')
                print(f'pred_pose_rotmats:{pred_pose_rotmats.shape}')
                print(f'pred_vertices:{pred_vertices.shape}')
                print(f'pred_joints_all:{pred_joints_all.shape}')
                print(f'pred_reposed_vertices:{pred_reposed_vertices.shape}')
                print('-----------------------')
            """
            pred_pose_rotmats: b*24*3*3
            pred_vertices: b*6890*3
            pred_joints_all: b*90*3
            pred_reposed_vertices: b*6890*3
            """
            
            pred_joints_h36m = pred_joints_all[:, LABELCONFIG.ALL_JOINTS_TO_H36M_MAP, :] # Human3.6M 17 joints
            pred_joints_h36mlsp = pred_joints_h36m[:,LABELCONFIG.H36M_TO_J14, :] # Human3.6M 14 LSP joints
            pred_joints_coco = pred_joints_all[:, LABELCONFIG.ALL_JOINTS_TO_COCO_MAP, :] # 17 COCO joints
            pred_joints2d_coco = orthographic_project_torch(pred_joints_coco, pred_cam_wp) # 对COCO joints3d 正交投影至 2d
            

            # ---------------- LOSS ----------------
            pred_dict_for_loss = {'joints2D': pred_joints2d_coco,
                                'verts': pred_vertices,
                                'shape_params': pred_shape,
                                'pose_params_rot_matrices': pred_pose_rotmats,
                                'joints3D': pred_joints_h36mlsp}
            
            target_dict_for_loss = {'joints2D': joints_dict["aug_joints2d_coco"] if self.args.sup_aug_j2d else joints_dict["joints2d_coco"],
                                    'verts': vertices_dict["aug_vertices"] if self.args.sup_aug_vertex else vertices_dict["vertices"],
                                    'shape_params': smpl_dict["shape"],
                                    'pose_params_rot_matrices': smpl_dict["all_rotmats"],
                                    'joints3D': joints_dict["joints_h36mlsp"],
                                    'IUV': iuv_dict["aug_iuv"] if self.args.sup_aug_IUV else iuv_dict["iuv"],
                                    'vis': joints_dict["joints2d_vis_coco"]}
            
            
            # rearrage target_dict
            target_dict_for_loss['verts'] = rearrange(target_dict_for_loss['verts'], 'b s c g -> (b s) c g')
            target_dict_for_loss['joints3D'] = rearrange(target_dict_for_loss['joints3D'], 'b s c g -> (b s) c g')
            target_dict_for_loss['pose_params_rot_matrices'] = rearrange(target_dict_for_loss['pose_params_rot_matrices'], 'b s c g w -> (b s) c g w')
            ########################
            if not is_train:
                return pred_dict_for_loss, target_dict_for_loss, pred_reposed_vertices, vertices_dict["reposed_vertices"]
            
            #Cal loss
            if self.args.debugging:
                print('-------- after rearrange --------')
                print(target_dict_for_loss['verts'].shape, pred_dict_for_loss['verts'].shape)
                print(target_dict_for_loss['joints3D'].shape, pred_dict_for_loss['joints3D'].shape)
                print(target_dict_for_loss['joints2D'].shape, pred_dict_for_loss['joints2D'].shape)
                print(target_dict_for_loss['pose_params_rot_matrices'].shape, pred_dict_for_loss['pose_params_rot_matrices'].shape)
                print('----------------------------------')
            # import pdb;pdb.set_trace()
            loss, task_losses_dict = self.criterion(target_dict_for_loss, pred_dict_for_loss) 
            total_loss += loss
        
        
        return total_loss, task_losses_dict, pred_dict_for_loss, target_dict_for_loss, pred_reposed_vertices, vertices_dict["reposed_vertices"]

    def trainEpoch(self):
        # print('Start training', datetime.now().strftime("%m%d%H%M%S"))
        for niter, samples_batch in enumerate(tqdm(self.train_dataloader)):
            
            totalloss, task_losses_dict, pred_dict_for_loss, target_dict_for_loss, pred_reposed_vertices, target_reposed_vertices \
                = self.forward_function(samples_batch, is_train=True)

            # ---------------- BACKWARD PASS ----------------
            self.optimiser.zero_grad()
            totalloss.backward()
            self.optimiser.step()

            # ---------------- TRACK LOSS AND METRICS ----------------#TODO: New metric_tracker
            self.metrics_tracker.update_per_batch('train', totalloss, task_losses_dict,
                                            pred_dict_for_loss, target_dict_for_loss,
                                            num_inputs_in_batch=pred_reposed_vertices.shape[0],
                                            pred_reposed_vertices=pred_reposed_vertices,
                                            target_reposed_vertices=target_reposed_vertices)
            # logging.info('Finish train iteration', datetime.now().strftime("%m%d%H%M%S"))
        
    def validate(self):
        with torch.no_grad():
            for _, samples_batch in enumerate(tqdm(self.val_dataloader)):
                pred_dict_for_loss, target_dict_for_loss, pred_reposed_vertices, target_reposed_vertices \
                    = self.forward_function(samples_batch, is_train=False)
                
                val_loss, val_task_losses_dict = self.criterion(target_dict_for_loss, pred_dict_for_loss)
                # ---------------- TRACK LOSS AND METRICS ----------------
                self.metrics_tracker.update_per_batch('val', val_loss, val_task_losses_dict,
                                                    pred_dict_for_loss, target_dict_for_loss,
                                                    num_inputs_in_batch=pred_reposed_vertices.shape[0],
                                                    pred_reposed_vertices=pred_reposed_vertices,
                                                    target_reposed_vertices=target_reposed_vertices)
                
    

    def trainLoop(self, num_epochs, epochs_per_save=1):
        best_epoch = self.start_epoch
        best_model_wts = copy.deepcopy(self.regressor.state_dict())

        for epoch in range(self.start_epoch, num_epochs):
            if dist.get_rank() == 0:
                logger.info(f'----------Epoch {epoch}/{num_epochs-1}----{datetime.now().strftime("%m%d%H%M%S")}--')
            if self.scheduler is not None:
                print(f'LR:{self.scheduler.get_last_lr()}------')
            self.metrics_tracker.initialise_loss_metric_sums()
            self.regressor.train()
            self.trainEpoch()
            if self.scheduler is not None:
                self.scheduler.step()
            if dist.get_rank() == 0:
                logger.info(f'Validation.....{datetime.now().strftime("%m%d%H%M%S")}')
            self.regressor.eval()
            self.validate()
            #UPDATING LOSS AND METRICS HISTORY
            self.metrics_tracker.update_per_epoch()
            
            #SAVING
            save_model_weights_this_epoch = self.metrics_tracker.determine_save_model_weights_this_epoch(self.save_val_metrics,
                                                                                                    self.best_epoch_val_metrics)
            if save_model_weights_this_epoch:
                for metric in self.save_val_metrics:
                    self.best_epoch_val_metrics[metric] = self.metrics_tracker.history['val_' + metric][-1]
                
                best_model_wts = copy.deepcopy(self.regressor.state_dict())
                best_epoch = epoch
                logger.info("Best model weights updated!")   
            logger.info(f"Best epoch val metrics updated to {self.best_epoch_val_metrics}")
            if epoch % epochs_per_save == 0 and dist.get_rank() == 0:
                # Saving current epoch num, best epoch num, best validation metrics (occurred in best
                # epoch num), current regressor state_dict, best regressor state_dict, current
                # optimiser state dict and current criterion state_dict (i.e. multi-task loss weights).
                save_dict = {'epoch': epoch,
                            'best_epoch': best_epoch,
                            'best_epoch_val_metrics': self.best_epoch_val_metrics,
                            'model_state_dict': self.regressor.state_dict(),
                            'best_model_state_dict': best_model_wts,
                            'optimiser_state_dict': self.optimiser.state_dict(),
                            'criterion_state_dict': self.criterion.state_dict()}
                torch.save(save_dict, f'{self.model_savedir}/epoch{epoch}.tar')
                logger.info(f'Model saved! Best Val Metrics:{self.best_epoch_val_metrics} at epoch{best_epoch}')
        if dist.get_rank() == 0:
            logger.info(f'Training Completed. Best Val Metrics:{self.best_epoch_val_metrics}')

        self.regressor.load_state_dict(best_model_wts)



   

