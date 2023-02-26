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

def fetch_processed_imgpr_name(datatype, rawimg_fullpath):
    if datatype == '3dpw':
        keywords = "_".join(rawimg_fullpath.split('/')[-2:])
        savename = keywords[:-4]
    elif datatype == 'mpi':
        keywords = "_".join(rawimg_fullpath.split('/')[-3:])
        savename = keywords[:-4]
    elif datatype.startswith('h36m'):   
        savename = rawimg_fullpath.split('/')[-1][:-4]
    return savename