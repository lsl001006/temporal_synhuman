import numpy as np
import os, argparse, sys, cv2
import torch
from torch.utils.data import DataLoader

import configs
from dataload.dataloader import Build_Test_Dataloader

from eval.predict_densepose import setup_densepose_silhouettes, predict_silhouette_densepose
from eval.predict_joints2d import setup_j2d_predictor, predict_joints2D_new


from utils.image_utils import crop_and_resize_iuv_joints2D_torch, crop_bbox_centerscale
from utils.io_utils import write_sample_iuv_j2d, fetch_processed_pr_path, fetch_processed_img_path

sys.path.append(f"{configs.DETECTRON2_PATH}/projects/DensePose")
from densepose.vis.extractor import DensePoseResultExtractor
from tqdm import tqdm




def debug_pred(dataset, joints2D_predictor, silhouette_predictor, extractor, nlist=[8394]):
    #h36mP1
    # Skip: [19484, 19485, 19486, 19487, 19493, 19494, 19495, 19496, 19497, 19498, 19499, 19500, 19502, 19503, 19504, 
    # 19505, 19506, 19523, 19524, 19543, 19544, 19545, 19560, 19561, 19616, 22724, 22725, 22731, 22732, 25584, 25585, 
    # 25586, 25587, 25588, 25589, 25590, 25591, 25592, 25593, 25594, 25595, 25596, 25597, 25598, 25599, 25600, 25601, 
    # 25641, 34437, 35142, 35144, 35145, 35146, 35147, 35148, 35149, 35150, 35151, 35152, 35153, 35154, 35155, 35156, 
    # 35157, 35158, 35159, 35160, 35161, 35162, 35163, 35181, 35183, 35184, 35185, 35186, 35187, 35188, 35189, 35190, 
    # 35192, 35193, 35196, 35198, 35200, 35201, 35202, 35203, 35205, 35206, 36392, 36393, 36394, 37588, 37592, 37593, 
    # 37594, 37596, 37597, 37598, 37599, 37600, 37601, 37602, 37603, 37604, 37606, 37607, 37608, 37609, 37610, 37611, 
    # 37612, 37613, 37614, 37615, 37616, 37617, 39756, 39998, 40014, 40015, 40317, 40324, 40327, 40931, 40932, 40933,
    # 40934, 40935, 40936, 40937, 40938, 40939, 40940, 40941, 40942, 41515, 41516, 41531, 42109, 42110, 42523, 42524, 42525, 42526, 42531, 44219]
    # Black: [39781, 39782, 39783, 40312, 40316, 40318, 40320, 40967, 41290, 41328, 42111, 42112, 42113, 42114]

    #3dpw
    # Skip: [17385]
    # Black: [13017, 13018, 33776, 33779, 33780, 33781, 33823, 33849, 33855, 33857, 33858, 33859, 33869, 33871, 33872, 33873, 
    # 33876, 33877, 34517, 34518, 34519, 34520, 34521, 34522, 34524, 34538, 34539, 34541, 34551, 34552, 34562, 34565]

    #mpi
    # Skip: [211, 216, 975, 1923]
    # Black: []
    for n in nlist:
        image = cv2.imread(dataset.imgnames[n])
        center = dataset.bbox_centers[n]
        scale = dataset.scales[n]*1.2
        image = crop_bbox_centerscale(image, center, scale, 
                res=256, resize_interpolation=cv2.INTER_LINEAR)

        #pred IUV 
        with torch.no_grad():
            outputs = silhouette_predictor(image)["instances"]
        # segm = outputs.get("pred_densepose").fine_segm
        bboxes = outputs.get("pred_boxes").tensor
        densepose = extractor(outputs[int(0)])[0][0]
        imap = densepose.labels/24.0
        uvmap = densepose.uv
        uvmap = torch.clip(uvmap, min=0, max=1)
        # import ipdb; ipdb.set_trace()
        iuv = torch.cat([imap[None], uvmap])#(3,h,w)

        #pred j2d
        outputs = joints2D_predictor(image)  # Multiple bboxes + keypoints predictions if there are multiple people in the image
        bboxes = outputs['instances'].pred_boxes.tensor
        keypoints = outputs['instances'].pred_keypoints.cpu().numpy()
        


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='3dpw') #[h36mp1, h36mp2, ssp3d, 3dpw, mpi]
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--img_crop_scale', type=float, default=0) #1.5
    parser.add_argument('--bbox_scale', type=float, default=1.2)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2) 
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0")
    
    #model
    joints2D_predictor = setup_j2d_predictor()
    silhouette_predictor = setup_densepose_silhouettes()
    extractor = DensePoseResultExtractor()
    
    #data 
    dataloader, _, _ = Build_Test_Dataloader(args)
    
    save_pr_dir = fetch_processed_pr_path(args.data, args.img_crop_scale, args.bbox_scale)
    save_img_dir = fetch_processed_img_path(args.data, args.img_crop_scale, args.bbox_scale)
    if not os.path.isdir(save_pr_dir):
        os.makedirs(save_pr_dir)
    if not os.path.isdir(save_img_dir):
        os.makedirs(save_img_dir)
    ########
    # debug_pred(dataset, joints2D_predictor, silhouette_predictor, extractor)
    
    skip_idx = []
    black_idx = []
    count=0
    for n_sample, samples_batch in tqdm(enumerate(dataloader)):
        imagename = samples_batch['imgname'][0]
        image = samples_batch['image'][0].numpy()
        center = samples_batch['center'][0].numpy()
        scale = samples_batch['scale'][0].item()

        IUV = predict_silhouette_densepose(image, silhouette_predictor, extractor, center, scale)#(img, imgw, 3)
        joints2D = predict_joints2D_new(image, joints2D_predictor, center, scale)#(17,3)
        if (IUV is None) or (joints2D is None):
            skip_idx.append(n_sample)
            continue
        IUV, joints2D, cropped_img = crop_and_resize_iuv_joints2D_torch(IUV, 
                                                                configs.REGRESSOR_IMG_WH, 
                                                                joints2D=joints2D, 
                                                                image=image, 
                                                                bbox_scale_factor=args.bbox_scale)
        ##
        bodymask = (24*IUV[:,:,0]).round().cpu().numpy()
        fg_ids = np.argwhere(bodymask != 0) 
        if fg_ids.shape[0]<256:
            print(f'{n_sample}only has {fg_ids.shape[0]} body pixels')
            black_idx.append(n_sample)
            continue
        # bodymask = np.argwhere(IUV[:,:,0].cpu().numpy()!=0)
        # x0,y0 = np.amin(bodymask,axis=0)
        # x1,y1 = np.amax(bodymask,axis=0)
        
        savename = imagename.split('/')[-1][:-4]
        write_sample_iuv_j2d(IUV.numpy(), joints2D, savename, save_pr_dir)
        cv2.imwrite(f'{save_img_dir}/{savename}.png', cropped_img)

        count+=1
        if count%1000==0:
            print(f'Saved {savename}..[{count}/{len(dataloader)}]')
    
    print('--------------------Completed------------------------------------')  
    print('Skip:', skip_idx)  
    print('Black:', black_idx)  