import os, argparse
import torch

import configs
from model.model import Build_Model
from utils.io_utils import fetch_processed_pr_path, fetch_processed_img_path
from dataload.dataloader import Build_Test_Dataloader
from eval.test_hmr import testHMRImg, testHMRPr
from utils.checkpoint_utils import load_ckpt_woSMPL


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--batch_size', type=int, default=128) #125 for mpi?
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--shuffle', type=int, default=1)
    
    #model setting
    parser.add_argument('--gender',type=str, default='')
    parser.add_argument('--pr', type=str, default='bj') #ONLY support bj for now
    parser.add_argument('--cra', action='store_true')
    parser.add_argument('--reg_ch', type=int, default=512) 
    parser.add_argument('--reg_hw', type=int, default=2) #[16,8,4,2,1]

    #checkpoints
    parser.add_argument('--ckpt', type=str, default='ndebug')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)

    #2D detector
    parser.add_argument('--img_crop_scale', type=float, default=0)
    parser.add_argument('--bbox_scale', type=float, default=1.2)
    #keys
    parser.add_argument('--vis',  type=str, default='')
    parser.add_argument('--data', type=str, default='h36mp2')#h36mp2, h36mp1, 3dpw, mpi
    parser.add_argument('--wgender', action='store_true')#NOT valid for now
    parser.add_argument('--j14', action='store_true')#NOT valid for now

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0")
    
    # Data
    pr_path = fetch_processed_pr_path(args.data, args.img_crop_scale, args.bbox_scale)
    img_path= fetch_processed_img_path(args.data, args.img_crop_scale, args.bbox_scale)
    test_w_pr = os.path.exists(pr_path) and os.path.exists(img_path)
    if not test_w_pr:
        pr_path, img_path = '', ''
        args.batch_size = 1
    dataloader, withshape, metrics_track = Build_Test_Dataloader(args, pr_path, img_path)
    
    # Model
    regressor, smpl_model = Build_Model(args.batch_size*configs.SEQLEN, 
                                        cra_mode=args.cra, 
                                        pr_mode=args.pr, 
                                        device=device, 
                                        itersup=0, 
                                        reginput_ch=args.reg_ch,
                                        reginput_hw=args.reg_hw)

    # Initialize 
    if test_w_pr:
        test_HMR = testHMRPr(regressor, smpl_model, device, args)
    else:
        test_HMR = testHMRImg(regressor, smpl_model, device, args)
    
    # Test
    if args.start==0 and args.end==0: #choose latest epoch
        state_dict = load_ckpt_woSMPL(f'{configs.CKPT_DIR}/{args.ckpt}', device, loadbest=True)
        if state_dict:
            regressor.load_state_dict(state_dict, strict=False) 
            mpjpe_pa = test_HMR.test(dataloader, withshape, metrics_track, vis=args.vis)
    else: #test with checkpoints from start to end
        metric_records = {}
        for ep in range(args.start, args.end):
            state_dict = load_ckpt_woSMPL(f'{configs.CKPT_DIR}/{args.ckpt}', device, epoch=ep, loadbest=False)
            if state_dict:
                regressor.load_state_dict(state_dict, strict=False)
                mpjpe_pa = test_HMR.test(dataloader, withshape, metrics_track, vis=args.vis)
                metric_records[ep] = mpjpe_pa
        print(metric_records)