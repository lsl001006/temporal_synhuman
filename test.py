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
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--shuffle', type=int, default=1) 
    #model setting
    parser.add_argument('--gender',type=str, default='')
    parser.add_argument('--pr', type=str, default='bj') #ONLY support bj for now
    parser.add_argument('--cra', action='store_true')
    parser.add_argument('--reg_ch', type=int, default=512) 
    parser.add_argument('--reg_hw', type=int, default=2) #[16,8,4,2,1]
    parser.add_argument('--use_vibe_reg', action='store_true') # use vibe regressor instead of ief module
    parser.add_argument('--debugging', action='store_true') # more info for debugging
    parser.add_argument('--use_temporal', action='store_true') # use the temporal module in regressor

    #checkpoints
    parser.add_argument('--ckpt', type=str, default='debug')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)

    #2D detector
    parser.add_argument('--img_crop_scale', type=float, default=0)
    parser.add_argument('--bbox_scale', type=float, default=1.2)
    #keys
    parser.add_argument('--visnum_per_batch',  type=int, default=1) #num visualized per batch
    parser.add_argument('--vispr', action='store_true')
    parser.add_argument('--data', type=str, default='h36mp2')#h36mp2, h36mp1, 3dpw, mpi
    parser.add_argument('--wgender', action='store_true')#NOT valid for now
    parser.add_argument('--j14', action='store_true')#NOT valid for now

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arguments()
    device = torch.device(f"cuda:{args.gpu}")
    
    # Data
    pr_path = fetch_processed_pr_path(args.data, args.img_crop_scale, args.bbox_scale)
    img_path= fetch_processed_img_path(args.data, args.img_crop_scale, args.bbox_scale)
    test_w_pr = os.path.exists(pr_path) and os.path.exists(img_path)
    if not test_w_pr:
        pr_path, img_path = '', ''
        args.batch_size = 1
        print("Infer from scratch...")
    dataloader, withshape, metrics_track = Build_Test_Dataloader(args, pr_path, img_path)
    print('Data loaded!')

    if args.visnum_per_batch:
        args.shuffle = False #unify name of saved images
    # Model
    regressor, smpl_model = Build_Model(args.batch_size*configs.SEQLEN, 
                                        local_rank=args.gpu,
                                        cra_mode=args.cra, 
                                        pr_mode=args.pr, 
                                        itersup=0, 
                                        reginput_ch=args.reg_ch,
                                        reginput_hw=args.reg_hw,
                                        phase='test',
                                        vibe_reg=args.use_vibe_reg,
                                        use_temporal=args.use_temporal)
    print('Model initialized')
    # Initialize 
    if test_w_pr:
        test_HMR = testHMRPr(regressor, smpl_model, device, args)
    else:
        test_HMR = testHMRImg(regressor, smpl_model, device, args)
    
    # Test
    if args.start==0 and args.end==0: #choose latest epoch
        state_dict, ep = load_ckpt_woSMPL(f'{configs.CKPT_DIR}/{args.ckpt}', device, loadbest=True)
        if state_dict:
            regressor.load_state_dict(state_dict, strict=False) 
            mpjpe_pa = test_HMR.test(dataloader, withshape, metrics_track, eval_ep = ep)
        
    else: #test with checkpoints from start to end
        metric_records = {}
        for ep in range(args.start, args.end):
            print('current epoch:', ep)
            state_dict, _ = load_ckpt_woSMPL(f'{configs.CKPT_DIR}/{args.ckpt}', device, epoch=ep, loadbest=False)
            if state_dict:
                regressor.load_state_dict(state_dict, strict=False)
                mpjpe_pa = test_HMR.test(dataloader, withshape, metrics_track,  eval_ep = ep)
                metric_records[ep] = mpjpe_pa
            else:
                continue
        print(metric_records)
    
    