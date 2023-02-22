import torch
import numpy as np
from utils.image_utils import get_transform
from utils.vis_utils import vis_bboxs

def convert_bbox_corners_to_centre_hw(bbox_corners):
    """
    Converst bbox coordinates from x1, y1, x2, y2 to centre, height, width.
    """
    x1, y1, x2, y2 = bbox_corners
    centre = np.array([(x1+x2)/2.0, (y1+y2)/2.0])
    height = x2 - x1
    width = y2 - y1

    return centre, height, width


def convert_bbox_centre_hw_to_corners(centre, height, width):
    x1 = centre[0] - height/2.0
    x2 = centre[0] + height/2.0
    y1 = centre[1] - width/2.0
    y2 = centre[1] + width/2.0

    return np.array([x1, y1, x2, y2])


def get_largest_centred_bounding_box(bboxes, orig_w, orig_h):
    """
    Given an array of bounding boxes, return the index of the largest + roughly-centred
    bounding box.
    :param bboxes: (N, 4) array of [x1 y1 x2 y2] bounding boxes
    :param orig_w: original image width
    :param orig_h: original image height
    """
    bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    sorted_bbox_indices = np.argsort(bboxes_area)[::-1]  # Indices of bboxes sorted by area.
    bbox_found = False
    i = 0
    while not bbox_found and i < sorted_bbox_indices.shape[0]:
        bbox_index = sorted_bbox_indices[i]
        bbox = bboxes[bbox_index]
        bbox_centre = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)  # Centre (width, height)
        if abs(bbox_centre[0] - orig_w / 2.0) < orig_w/6.0 and abs(bbox_centre[1] - orig_h / 2.0) < orig_w/6.0:
            largest_centred_bbox_index = bbox_index
            bbox_found = True
        i += 1

    # If can't find bbox sufficiently close to centre, just use biggest bbox as prediction
    if not bbox_found:
        largest_centred_bbox_index = sorted_bbox_indices[0]

    return largest_centred_bbox_index


def select_bbox_max_IOU(bboxs, inside_list, center, scale, imgh, imgw, wh=256, bbox_to_square=True):
    #bbox after square
    gt_bmap = np.zeros((imgh, imgw))
    ##
    tmat = get_transform(center, scale, wh, wh) #res
    ul = np.dot(tmat, np.array([0,0,1]).T)[:2]
    br = np.dot(tmat, np.array([wh-1, wh-1, 1]).T)[:2]#res
    ul, br = ul.astype('int'), br.astype('int')
    x0_gt, x1_gt = max(0, ul[0]), min(imgw-1, br[0])
    y0_gt, y1_gt = max(0, ul[1]), min(imgh-1, br[1])
    gt_bmap[y0_gt:y1_gt, x0_gt:x1_gt] = 1
    ##
    nbest = inside_list.index(True)
    IOUmax = 0
    for n in range(bboxs.shape[0]):
        if not inside_list[n]:
            continue
        bbox_bmap = np.zeros((imgh, imgw))
        x0, y0, x1, y1 = bboxs[n]
        x0, y0, x1, y1 = x0.int().item(), y0.int().item(), x1.int().item(), y1.int().item()
        if bbox_to_square:
            height, width = y1-y0,x1-x0
            if width>height:
                border_height = (height - width) // 2 
                y0 = max(y0-border_height, 0)
                y1 = min(y1+border_height,imgh-1)
            elif height>width:
                border_width = (height - width) // 2
                x0 = max(x0-border_width, 0)
                x1 = min(x1+border_width,imgw-1)
        bbox_bmap[y0:y1, x0:x1] = 1
        IOU = np.logical_and(bbox_bmap, gt_bmap).sum()/np.logical_or(bbox_bmap, gt_bmap).sum()
        if IOU>IOUmax:
            nbest = n
            IOUmax = IOU
    return nbest, (x0_gt, y0_gt, x1_gt, y1_gt)


def inside_bbox(point, bbox):
    x0, y0, x1, y1 = bbox
    # import ipdb; ipdb.set_trace()
    x0,y0,x1,y1 = x0.round().item(), y0.round().item(), x1.round().item(), y1.round().item()
    inside = (point[0]>x0) and (point[0]<x1) and (point[1]>y0) and (point[1]<y1)
    return inside


def select_bbox_closest_center(bboxs, inside_list, center, image_area):
    center_x, center_y = center
    nbest = inside_list.index(True)
    distancemin = image_area
    for n in range(bboxs.shape[0]):
        if not inside_list[n]:
            continue
        x0, y0, x1, y1 = bboxs[n]
        x0, y0, x1, y1 = x0.int().item(), y0.int().item(), x1.int().item(), y1.int().item()
        bbox_center_x, bbox_center_y = (x0+x1)/2.0, (y0+y1)/2.0
        distance = np.abs(bbox_center_x-center_x) + np.abs(bbox_center_y-center_y)
        if distance<distancemin:
            distancemin = distance
            nbest = n
    return nbest


def select_bboxes_gtcrop(input_image, bboxes, center, scale, target_wh=256, vispath_ambiguous=''):
    """
    bboxes: torch.tensor (nbox, 4)
    """
    verified = True
    idx = 0
    image = np.copy(input_image)
    img_h, img_w = image.shape[:2]
    if bboxes.shape[0]==0:
        print('No bbox detected...')
        return False, None
    elif bboxes.shape[0]>1:
        inside_list = [inside_bbox(center, bboxes[n]) for n in range(bboxes.shape[0])]
        if sum(inside_list)==0:
            print('No bbox detected...')
            return False, None
        idx = inside_list.index(True) #return the first idx that is True
        # import ipdb; ipdb.set_trace()
        if sum(inside_list)>1:
            # idx1 = select_bbox_largest_area(bbox)
            idx1 = select_bbox_closest_center(bboxes, inside_list, center, img_h*img_w)
            idx2, gt_box = select_bbox_max_IOU(bboxes, inside_list, center, scale, img_h, img_w, wh=target_wh)
            consistent = (idx1==idx2) 
            if vispath_ambiguous and not consistent:
                verified = False
                vis_bboxs(vispath_ambiguous, image, torch.cat([bboxes[idx1][None], bboxes[idx2][None]], dim=0), 
                    gt_box=torch.tensor(gt_box), gt_center=center)
                # import ipdb; ipdb.set_trace()
            idx = idx2
    return verified, idx


def expand_bbox(iuv_ts, whole_h, whole_w, bbox_xyxy):
    """
    iuv_ts: (3,h,w)
    """
    x_start, y_start, _, _ = bbox_xyxy
    x_start, y_start = int(x_start), int(y_start)
    _, h, w = iuv_ts.shape
    whole_iuv = torch.zeros((3, whole_h, whole_w))
    #overpass border?
    if y_start+h>whole_h or x_start+w>whole_w:
        # import ipdb; ipdb.set_trace()
        h = whole_h - y_start
        w = whole_w - x_start
    whole_iuv[:, y_start:y_start+h, x_start:x_start+w] = iuv_ts[:,:h,:w]
    return whole_iuv