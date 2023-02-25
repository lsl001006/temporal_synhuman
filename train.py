import os, argparse, logging, configs
import torch
import shutil

from datetime import datetime
from dataload.dataloader import Build_Train_Dataloader
from model.model import Build_Model
from loss.loss import Build_Loss
from synthetic_train import my_train_rendering
from synthesis import RenderGenerate
import torch.distributed as dist


#python train.py 
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epochs_per_save', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=12) #4->8s/iter, 6->7s/iter, 8->6.5s/iter
    
    parser.add_argument('--log', type=str, default='debug')
    parser.add_argument('--save_command', action='store_true') # 保存训练sh文件
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
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend='nccl')
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    
    # SET UP LOG  
    if args.log=='debug':
        logfile = f'debug'
    else:
        logfile = f'{args.log}-{datetime.now().strftime("%m%d%H%M%S")}'
    if not os.path.isdir(configs.LOG_DIR):
        os.makedirs(configs.LOG_DIR, exist_ok=True)
    log_path= f'{configs.LOG_DIR}/{logfile}.txt'
    
    
    # Model save dir
    model_savedir = f'{configs.CKPT_DIR}/{args.log}'
    if not os.path.isdir(model_savedir):
        os.makedirs(model_savedir, exist_ok=True)
    
    # copy train.sh to model_savedir
    if os.path.exists('scripts/train.sh'):
        shutil.copy('scripts/train.sh', model_savedir+'/train.sh')
    else:
        print('[Warning] Train Script is not saved!')
    #
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(message)s",
                        datefmt='%d-%b-%y %H:%M:%S',
                        force=True,
                        handlers=[logging.FileHandler(log_path),
                                logging.StreamHandler()])

    # Run
    
    
    # DATA
    train_dataloader, val_dataloader = Build_Train_Dataloader(args.batch_size, 
                                                             num_workers=args.num_workers, 
                                                             valdata=args.valdata) 
    # MODEL
    regressor, smpl_model = Build_Model(args.batch_size*configs.SEQLEN, #args.batch_size
                                        local_rank=local_rank,
                                        cra_mode=args.cra, 
                                        pr_mode=args.pr, 
                                        itersup=args.itersup, 
                                        reginput_ch=args.reg_ch,
                                        reginput_hw=args.reg_hw,
                                        phase='train')
    
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
    dist.destroy_process_group()
    # FINETUNE
    # if args.valdata:
    #     withshape = False if args.vt=='h36m' else True
    #     train_render.trainLoopVTest(num_epochs=args.epochs, withshape=withshape)
