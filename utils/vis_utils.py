
import os, cv2, torch
import numpy as np
import colorsys
import sys
import matplotlib.pyplot as plt

def saveKP2D(keypoints, savePath, image=None, H=256, W=256, color=(0,255,0), addText=True, circle_size=5):
    if image is None:
        image = np.zeros((H,W,3)).astype('uint8')
    drawimage = image.copy()
    for j in range(keypoints.shape[0]):
        # import ipdb; ipdb.set_trace()
        drawimage = cv2.circle(drawimage, (int(keypoints[j, 0]), int(keypoints[j, 1])), circle_size, color, -1)
        if addText:
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            fontColor = (0, 255, 0)
            cv2.putText(drawimage, str(j), (int(keypoints[j, 0]), int(keypoints[j, 1])),
                                        font, fontScale, fontColor, lineType=2)
    if savePath is not None:
        cv2.imwrite(savePath, drawimage)
    return drawimage

def hsv_to_bgr(hsv_ts, keep_background=True, background=[1.0,1.0,1.0]): #[N,3]
    H, W, _ = hsv_ts.shape
    # import ipdb; ipdb.set_trace()
    hsv_ts = hsv_ts.view(-1,3)#[HW,3]
    hsv = hsv_ts.cpu().numpy().tolist()
    # verts_bgr = cv2.cvtColor(verts_hsv, cv2.COLOR_HSV2BGR)
    # 
    rgb = [ colorsys.hsv_to_rgb(vert_hsv[0],vert_hsv[1], vert_hsv[2]) for vert_hsv in hsv]#[1,1,1]->[1,0,0]
    # rgb = [ colorsys.hsv_to_rgb(vert_hsv[0], 1,1) for vert_hsv in hsv]

    rgb_ts = torch.tensor(rgb).to(hsv_ts.device)
    # rgb_ts[hsv_ts[:,0]==0,:] = torch.tensor([0.0, 0.0, 0.0])
    if keep_background:
        background = torch.tensor(background).to(hsv_ts.device)
        rgb_ts[(hsv_ts==background).all(dim=1)] = background
    # bgr_ts = rgb_ts[:,::-1]
    bgr_ts = torch.stack([rgb_ts[:,2],rgb_ts[:,1],rgb_ts[:,0]], dim=1)
    # 
    bgr_ts = bgr_ts.reshape(H,W,3)
    return bgr_ts


def saveKPIUV(kp2d, iuv_map, rootpath='vis'):
    kp2d = kp2d.cpu().numpy()
    iuv_map = iuv_map.cpu()
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    batch_size, img_h, img_w, _ = iuv_map.shape
    for nimg in range(batch_size):
        image = np.zeros((img_h, img_w,3)).astype('uint8')
        image[iuv_map[nimg,:,:,0]>0]= (255,255,255)
        saveKP2D(kp2d[nimg], f'{rootpath}/{nimg}_joints.png',image=image, color=(0, 0, 255), addText=True)
        bgr = hsv_to_bgr(iuv_map[nimg], keep_background=True)
        cv2.imwrite( f'{rootpath}/{nimg}_iuv.png', 255*bgr.cpu().numpy())

def vis_j2d_occlusion_batch(IUV_batch, j2d_batch, j2d_no_occluded_mask_batch, visdir=f'visj2d_occlusion'):
    """
    IUV_batch: (bs, hw, hw, 3)
    j2d_batch: (bs, 17, 2)
    j2d_no_occluded_mask_batch: (bs, 17)
    """
    if not os.path.isdir(visdir):
        os.makedirs(visdir)
    bs = IUV_batch.shape[0]
    hw = IUV_batch.shape[1]
    njoints = j2d_batch.shape[1]
    for b in range(bs):
        IUV = IUV_batch[b]
        joints2D = j2d_batch[b]
        joints2D_mask = j2d_no_occluded_mask_batch[b]
        bgr = hsv_to_bgr(IUV, keep_background=True)
        iuv_image = 255*bgr.cpu().numpy()
        vis_image = iuv_image
        for nj in range(njoints):
            color_visible = (0,255,0) #green
            color_invisible = (0, 0, 255) #red
            # import ipdb; ipdb.set_trace()
            color = color_visible if joints2D_mask[nj].item() else color_invisible
            vis_image = saveKP2D(joints2D[nj][None], None, image=vis_image, H=hw, W=hw, color=color, addText=False)
        cv2.imwrite(f'{visdir}/{b}.png', vis_image)
        # import ipdb; ipdb.set_trace()

def vis_bboxs(save_path, image, bboxs, gt_box=None, gt_center=None):
    for n in range(bboxs.shape[0]):
        x0, y0, x1, y1 = bboxs[n]
        x0,y0,x1,y1 = x0.int().item(), y0.int().item(), x1.int().item(), y1.int().item()
        image = cv2.rectangle(image, (x0,y0), (x1,y1), (0,255,0), 5)
    if gt_box is not None:
        x0, y0, x1, y1 = gt_box
        x0,y0,x1,y1 = x0.int().item(), y0.int().item(), x1.int().item(), y1.int().item()
        image = cv2.rectangle(image, (x0,y0), (x1,y1),  (0,0,255), 5)
    if gt_center is not None:
        x , y = gt_center
        image = cv2.circle(image, (int(x), int(y)), 20, (0,0,255), -1)
    cv2.imwrite(save_path, image)


def apply_colormap(image, vmin=None, vmax=None, cmap='viridis', cmap_seed=1):
    """
    Apply a matplotlib colormap to an image.
    This method will preserve the exact image size. `cmap` can be either a
    matplotlib colormap name, a discrete number, or a colormap instance. If it
    is a number, a discrete colormap will be generated based on the HSV
    colorspace. The permutation of colors is random and can be controlled with
    the `cmap_seed`. The state of the RNG is preserved.
    """
    image = image.astype("float64")  # Returns a copy.
    # Normalization.
    if vmin is not None:
        imin = float(vmin)
        image = np.clip(image, vmin, sys.float_info.max)
    else:
        imin = np.min(image)
    if vmax is not None:
        imax = float(vmax)
        image = np.clip(image, -sys.float_info.max, vmax)
    else:
        imax = np.max(image)
    image -= imin
    image /= (imax - imin)
    # Visualization.
    cmap_ = plt.get_cmap(cmap)
    vis = cmap_(image, bytes=True)
    return vis