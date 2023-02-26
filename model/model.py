import logging
from .smpl_official import SMPL
from .regressor import SingleInputRegressor
from .regressor_align import FuseAlignRegressor
import torch
import torch.distributed as dist

import configs
logger = logging.getLogger()

def Build_SMPL(local_rank, batch_size,  gender=''):
    # SMPL model
    if gender:
        smpl_model = SMPL(configs.SMPL_MODEL_DIR, batch_size=batch_size, gender=gender)
    else:
        smpl_model = SMPL(configs.SMPL_MODEL_DIR, batch_size=batch_size) # neutral gender
    device = torch.device("cuda:{}".format(local_rank))
    # smpl_model = nn.DataParallel(smpl_model)
    # smpl_model = nn.DataParallel(smpl_model, device_ids=[0,1]).to(device)
    smpl_model = smpl_model.to(device)
    return smpl_model

def Build_Model(batch_size, 
                local_rank,
                cra_mode=False, 
                pr_mode='bj', 
                itersup=False, 
                reginput_ch=512, 
                reginput_hw=1,  
                gender='',
                phase='train',
                vibe_reg=True,
                use_temporal=True):
    smpl_model = Build_SMPL(local_rank, batch_size,  gender=gender)
    
    if pr_mode=='bj':
        resnet_in_channels = [1, 17]
    elif pr_mode=='i':
        resnet_in_channels = [24, 17]
    elif pr_mode=='biuv':
        resnet_in_channels = [27, 17]
    elif pr_mode=='bij':
        resnet_in_channels = [2, 17]
    elif pr_mode=='biuvj':
        resnet_in_channels = [3, 17]
    elif pr_mode=='iuv':
        resnet_in_channels = [26, 17]
    elif pr_mode=='iuvj':
        resnet_in_channels = [3, 17]
    device = torch.device("cuda:{}".format(local_rank))
    if cra_mode==True:
        regressor = FuseAlignRegressor(batch_size,
                smpl_model,
                device=device,
                resnet_in_channels=resnet_in_channels,
                feat_size=[1,4,8], #multiscale
                itersup=itersup,
                filter_channels =[512, 256, 32, 8],
                encoddrop=False,
                fuse_r1=False,
                add_channels=[[[0,1],[1,0]], [[0,1],[1,0]]],
                add_fuser = False,
                add_ief = False,
                recon=False)
    else: #should support multi-gpu?
        regressor = SingleInputRegressor(sum(resnet_in_channels),  
                                         ief_iters=3, 
                                         itersup=itersup, 
                                         reginput_ch=reginput_ch,
                                         reginput_hw=reginput_hw,
                                         encoddrop=False,
                                         vibe_reg=vibe_reg,
                                         use_temporal=use_temporal)
    if phase == 'train':
        regressor = torch.nn.parallel.DistributedDataParallel(regressor.to(device), 
                                                          broadcast_buffers=False, 
                                                          find_unused_parameters=True)
    elif phase=='test':
        regressor = regressor.to(device)

    num_params = count_parameters(regressor)
    # if dist.get_rank() == 0:
    if True:
        logger.info(f"Regressor model Loaded. {num_params} trainable parameters.")
    
    return regressor, smpl_model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)