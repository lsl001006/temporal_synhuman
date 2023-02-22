import os
import numpy as np
import configs

def write_sample_iuv_j2d(iuv, j2d, name, savedir):
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    np.savez(f'{savedir}/{name}.npz', iuv=iuv, j2d=j2d) 


def fetch_processed_pr_path(datatype, img_crop_scale, bbox_scale):
    path = f'{configs.PROCESS_PATH}/{datatype}/pr_i{img_crop_scale}_s{bbox_scale}'
    return path

def fetch_processed_img_path(datatype, img_crop_scale, bbox_scale):
    path = f'{configs.PROCESS_PATH}/{datatype}/image_i{img_crop_scale}_s{bbox_scale}'
    return path