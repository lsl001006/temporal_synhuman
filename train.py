import os, argparse, logging, configs
import torch

from datetime import datetime
from dataload.dataloader import Build_Train_Dataloader
from model.model import Build_Model
from loss.loss import Build_Loss
from synthetic_train import my_train_rendering
from synthesis import RenderGenerate
import torch.distributed as dist

#python run_mytrain.py --epochs 240 --batch_size 140 --lossvar --pr bj
# def setup_DDP(backend="nccl", verbose=False):
#     """
#     We don't set ADDR and PORT in here, like:
#         # os.environ['MASTER_ADDR'] = 'localhost'
#         # os.environ['MASTER_PORT'] = '12355'
#     Because program's ADDR and PORT can be given automatically at startup.
#     E.g. You can set ADDR and PORT by using:
#         python -m torch.distributed.launch --master_addr="192.168.1.201" --master_port=23456 ...

#     You don't set rank and world_size in dist.init_process_group() explicitly.

#     :param backend:
#     :param verbose:
#     :return:
#     """
#     rank = int(os.environ["RANK"])
#     local_rank = int(os.environ["LOCAL_RANK"])
#     world_size = int(os.environ["WORLD_SIZE"])
#     # If the OS is Windows or macOS, use gloo instead of nccl
#     dist.init_process_group(backend=backend)
#     # set distributed device
#     device = torch.device("cuda:{}".format(local_rank))
#     if verbose:
#         print(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}")
#     return rank, local_rank, world_size, device


#python train.py 
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epochs_per_save', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=12) #4->8s/iter, 6->7s/iter, 8->6.5s/iter
    parser.add_argument('--local_rank', type=int, default=-1, help="DDP parameter, do not modify")
    
    parser.add_argument('--log', type=str, default='debug')
    parser.add_argument('--resume_from_epoch', type=str, default='') #NOT Veirified
    parser.add_argument('--resume_sametr', action='store_true') #NOT Veirified
    # parser.add_argument('--finetune', action='store_true') #NOT Support for now
    #setting
    parser.add_argument('--pr', type=str, default='bj') #ONLY support bj for now
    parser.add_argument('--cra', action='store_true')
    
    parser.add_argument('--reg_ch', type=int, default=512) 
    parser.add_argument('--reg_hw', type=int, default=2) #[16,8,4,2,1]
    
    parser.add_argument('--valaug', type=int, default=1)
    parser.add_argument('--valdata', type=str, default='') #['', 'h36m', '3dpw'] #ONLY support '' for now
    #Loss
    parser.add_argument('--lossvar', action='store_true')
    parser.add_argument('--shapeloss', action='store_true')
    parser.add_argument('--vertloss', type=int, default=1)
    parser.add_argument('--itersup', action='store_true')
    parser.add_argument('--sup_aug_IUV', type=int, default=1)
    parser.add_argument('--sup_aug_j2d', type=int, default=1)
    parser.add_argument('--sup_aug_vertex', type=int, default=1)

    args = parser.parse_args()
    
    
    return args

if __name__ == '__main__':
    args = get_arguments()
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda:{}".format(args.local_rank))
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # SET UP LOG  
    if args.log=='debug':
        logfile = f'debug'
    else:
        logfile = f'{args.log}-{datetime.now().strftime("%m%d%H%M%S")}'
    if not os.path.isdir(configs.LOG_DIR):
        os.makedirs(configs.LOG_DIR)
    log_path= f'{configs.LOG_DIR}/{logfile}.txt'
    
    model_savedir = f'{configs.CKPT_DIR}/{args.log}'
    if not os.path.isdir(model_savedir):
        os.makedirs(model_savedir)
    #
    handlers = [logging.StreamHandler()]
    handlers.append(logging.FileHandler(log_path, mode='a'))
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers,     
    )
    

    # Run
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # logging.info(f'Device:{device}...GPU:{args.gpu}...')
    # logging.info(args)
    
    # DATA
    train_dataloader, val_dataloader = Build_Train_Dataloader(args.batch_size, 
                                                             num_workers=args.num_workers, 
                                                             valdata=args.valdata) 
    # MODEL
    regressor, smpl_model = Build_Model(args, args.batch_size*configs.SEQLEN, #args.batch_size
                                        cra_mode=args.cra, 
                                        pr_mode=args.pr, 
                                        itersup=args.itersup, 
                                        reginput_ch=args.reg_ch,
                                        reginput_hw=args.reg_hw)
    
    # LOSS
    criterion, losses_to_track = Build_Loss(var=args.lossvar, # 是否启用loss变体
                                            shapeloss=args.shapeloss,  # 是否使用shape loss
                                            vertloss=args.vertloss) # 是否使用 vertices loss
    
    # Training Data Synthesis
    renderSyn = RenderGenerate(smpl_model,
                    args.batch_size,
                    args.valaug, 
                    device=device, 
                    render_options={'j2D':1, 'depth':0, 'normal':0, 'iuv':1})

    train_render = my_train_rendering(args,
                    renderSyn=renderSyn,
                    regressor=regressor,
                    device=device,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    criterion=criterion,
                    losses_to_track=losses_to_track,
                    model_savedir=model_savedir,
                    log_path=log_path, 
                    metrics_to_track=['mpjpes_pa'],
                    save_val_metrics=['mpjpes_pa'])
                    
    # metrics_to_track = ['pves', 'pves_sc', 'pves_pa', 'pve-ts', 'pve-ts_sc', 'mpjpes', 'mpjpes_sc',
                    # 'mpjpes_pa', 'shape_mses', 'pose_mses', 'joints2D_l2es']
    train_render.trainLoop(num_epochs=args.epochs, epochs_per_save=args.epochs_per_save)

    # FINETUNE
    # if args.valdata:
    #     withshape = False if args.vt=='h36m' else True
    #     train_render.trainLoopVTest(num_epochs=args.epochs, withshape=withshape)
