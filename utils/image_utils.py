import cv2
import torch
import torch.nn.functional as nnf
import numpy as np


def pad_to_square(image):
    """
    Pad image to square shape.
    """
    height, width = image.shape[:2]

    if width < height:
        border_width = (height - width) // 2
        image = cv2.copyMakeBorder(image, 0, 0, border_width, border_width,
                                   cv2.BORDER_CONSTANT, value=0)
    else:
        border_width = (width - height) // 2
        image = cv2.copyMakeBorder(image, border_width, border_width, 0, 0,
                                   cv2.BORDER_CONSTANT, value=0)

    return image

def get_transform(center, scale, H, W):
    tmat = np.zeros((3,3))
    tmat[0, 0] = float(W) / scale
    tmat[1, 1] = float(H) / scale
    tmat[0, 2] = W * (-float(center[0]) / scale + .5)
    tmat[1, 2] = H * (-float(center[1]) / scale + .5)
    tmat[2, 2] = 1 
    # return tmat
    return np.linalg.inv(tmat)

def crop_bbox_centerscale(image, center, scale, res=256, resize_interpolation= cv2.INTER_NEAREST):
    imgh, imgw = image.shape[:2]
    resh, resw = res, res
    tmat = get_transform(center, scale, resh, resw) #res
    ul = np.dot(tmat, np.array([0,0,1]).T)[:2]
    br = np.dot(tmat, np.array([resh-1, resw-1, 1]).T)[:2]#res
    ul, br = ul.astype('int'), br.astype('int')
    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(image.shape) > 2:
        new_shape += [image.shape[2]]
    new_image = np.zeros(new_shape)
    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], imgw) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], imgh) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(imgw, br[0])
    old_y = max(0, ul[1]), min(imgh, br[1])
    # import ipdb; ipdb.set_trace()

    new_image[new_y[0]:new_y[1], new_x[0]:new_x[1]] = image[old_y[0]:old_y[1], 
                                                        old_x[0]:old_x[1]]
    #resize
    new_image = cv2.resize(new_image, (resh, resw), interpolation=resize_interpolation)
    return new_image



def silhouette_from_j2d(j2d, orig_h, orig_w, std=4):
    size = 2 * std
    silhouette = np.zeros((orig_h, orig_w))
    for nj in range(j2d.shape[0]):
        x, y = j2d[nj][:2]
        x, y = int(x), int(y)
        x, y = min(x,orig_w-1), min(y, orig_h-1)#gurantee x1>x0, y1>y0
        x0, x1 = min(max(0, x-size),orig_w), max(min(x+size, orig_w), 0)
        y0, y1 = min(max(0, y-size),orig_h), max(min(y+size, orig_h), 0)
        # print(x0,x1,y0,y1)
        # if y0==y1 and x0==x1:
        #     silhouette[y0, x0] = 1
        # elif y0==y1:
        #     silhouette[y0, ]
        silhouette[y0:y1, x0:x1] = np.ones((y1-y0, x1-x0))
    return silhouette

def convert_bbox_centre_hw_to_corners(centre, height, width):
    x1 = centre[0] - height/2.0
    x2 = centre[0] + height/2.0
    y1 = centre[1] - width/2.0
    y2 = centre[1] + width/2.0

    return np.array([x1, y1, x2, y2])

def convert_bbox_corners_to_centre_hw(bbox_corners):
    """
    Converst bbox coordinates from x1, y1, x2, y2 to centre, height, width.
    """
    x1, y1, x2, y2 = bbox_corners
    centre = np.array([(x1+x2)/2.0, (y1+y2)/2.0])
    height = x2 - x1
    width = y2 - y1

    return centre, height, width

def crop_and_resize_iuv_joints2D_torch(iuv, out_wh, joints2D=None, image=None, bbox_scale_factor=1.2):
    """
    iuv: tensor (h, w, 3)
    image: np.array (h, w, 3)
    j2d： np.array (nj,4)
    """
    # Find bounding box around silhouette
    orig_h, orig_w = iuv.shape[:2]
    silhouette = iuv[:,:,0].cpu().numpy()
    if joints2D is not None:
        j2d_silhouette = silhouette_from_j2d(joints2D, orig_h, orig_w)
        silhouette = silhouette+j2d_silhouette
    body_pixels = np.argwhere(silhouette != 0)
    bbox_centre, height, width = convert_bbox_corners_to_centre_hw(np.concatenate([np.amin(body_pixels, axis=0),
                                                                                   np.amax(body_pixels, axis=0)]))
    wh = max(height, width) * bbox_scale_factor  # Make bounding box square with sides = wh
    bbox_corners = convert_bbox_centre_hw_to_corners(bbox_centre, wh, wh)
    top_left = bbox_corners[:2].astype(np.int16)
    bottom_right = bbox_corners[2:].astype(np.int16)
    top_left[top_left < 0] = 0
    bottom_right[bottom_right < 0] = 0
    bottom_right[0] = min(bottom_right[0], orig_h)
    bottom_right[1] = min(bottom_right[1], orig_w)
    #crop and pad iuv tensor
    crop_h, crop_w = bottom_right[0]-top_left[0], bottom_right[1]-top_left[1]
    sqaure_hw = max(crop_h, crop_w)
    pad_h, pad_w = (sqaure_hw-crop_h)//2, (sqaure_hw-crop_w)//2
    iuv_padded = torch.zeros((sqaure_hw, sqaure_hw, 3))
    iuv_padded[pad_h:pad_h+crop_h, pad_w:pad_w+crop_w, :] = iuv[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]
    #resize
    iuv = nnf.interpolate(iuv_padded.permute(2,0,1)[None], size=(out_wh, out_wh), mode='nearest')[0].permute(1,2,0)
    #Translate and resize joints2D
    if joints2D is not None:
        top_left_shift = np.array([pad_h-top_left[0], pad_w-top_left[1]])
        joints2D = joints2D[:, :2] + top_left_shift[::-1]
        joints2D = joints2D * np.array([out_wh / float(sqaure_hw),
                                        out_wh / float(sqaure_hw)])
    if image is not None:
        image_padded = np.zeros((sqaure_hw, sqaure_hw, 3))
        image_padded[pad_h:pad_h+crop_h, pad_w:pad_w+crop_w, :] = image[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]
        image = cv2.resize(image_padded, (out_wh, out_wh), interpolation=cv2.INTER_LINEAR)

    return iuv, joints2D, image