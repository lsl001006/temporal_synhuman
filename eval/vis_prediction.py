import os, cv2, torch
import numpy as np
import configs
from utils.renderer import P3dRenderer
from pytorch3d.structures import Meshes 
from pytorch3d.renderer.cameras import OrthographicCameras
from pytorch3d.renderer import (
    MeshRenderer, 
    MeshRasterizer, 
    HardPhongShader,
)
from pytorch3d.renderer.mesh import Textures
from pytorch3d.renderer.blending import hard_rgb_blend

class VisRenderer(P3dRenderer):
    def __init__(self, batch_size, device):
        self.img_wh = configs.REGRESSOR_IMG_WH
        self.batch_size = batch_size
        self.device = device
        #load Densepose
        self.load_densepose(self.img_wh, batch_size, device)

        self.camT = self.rectify_cam_T(None)
        self.camT[:,2] = 5
        self.camR = self.rectify_cam_R(None)
        
        #color
        # batch_color = torch.tensor([1,1,1]).repeat(7829,1)
        # self.colors = [batch_color.float().to(device) for _ in range(batch_size)] #(bs, 7829, 3) 
        self.rgb_encoding = {'gt':[0,1,2], 'pred':[1,2,0]}

    def get_camK_fast(self, pred_cam):
        focal_length = pred_cam[:,0] #bs
        t_x, t_y= -pred_cam[:,1]*focal_length, -pred_cam[:,2]*focal_length#bs
        focal_length = focal_length[:,None, None]
        t_x = t_x[:,None, None]
        t_y = t_y[:,None, None]
        fill_zeros = torch.zeros_like(t_x)
        fill_ones = torch.ones_like(t_x)
        #cam_K: (bs,3,3)
        cam_K_row1 = torch.cat([focal_length,fill_zeros, t_x], dim=2)#(bs,1,3)
        cam_K_row2 = torch.cat([fill_zeros, focal_length, t_y], dim=2)#(bs,1,3)
        cam_K_row3 = torch.cat([fill_zeros, fill_zeros, fill_ones], dim=2)#(bs,1,3)

        cam_K = torch.cat([cam_K_row1, cam_K_row2, cam_K_row3], dim=1)
        return cam_K

    def forward(self, vertices, pred_cam, key='pred'):
        """
        vertices: (bs, nv, 3)
        pred_cam: (bs, 3)
        """
        unique_verts = [vertices[:,vid].cpu().numpy() for vid in self.to_1vertex_id] #(7829,bs,3)
        unique_verts = torch.as_tensor(unique_verts).transpose(0,1).to(self.device) #(bs, 7829, 3)
        verts_list = [unique_verts[nb] for nb in range(self.batch_size)] #(bs, 7829, 3)
        #
        
        mesh_batch = Meshes(verts=verts_list, faces=self.faces_list, textures = Textures(verts_rgb=self.IUVnorm_list)) #self.colors

        camK = self.get_camK_fast(pred_cam)
        cameras = OrthographicCameras(
                focal_length=((self.focal_x, self.focal_y),),
                principal_point=((self.t_x, self.t_y),),
                R = self.cam_R,
                T = self.cam_T,
                device = self.device, 
                in_ndc = False,
                image_size=((-1, -1),)
                # image_size=((img_wh, img_wh),)
                )
        
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings)
        fragments = rasterizer(mesh_batch)
        normal_map = self.get_pixel_normals(mesh_batch, fragments)
        normalimage = 0.5+normal_map/2
        normalimage = (255*normalimage.cpu().numpy()).astype('uint8')
        normalimage = normalimage[:,:,self.rgb_encoding[key]]
        # depth_map = fragments.zbuf.squeeze()
        
        # shader = HardPhongShader(device=self.device, cameras=cameras, blend_params=self.blendparam)  
        # render_all = MeshRenderer(rasterizer=rasterizer, shader=shader)
        # meshimage = render_all(mesh_batch)
        # meshimage = (255*meshimage[0,:,:,:3].cpu().numpy()).astype('uint8')
        # body_mask = (meshimage.sum(axis=2)>0).astype('int')  
        colors = mesh_batch.sample_textures(fragments)#(BS, H, W, 1, 4)
        images = hard_rgb_blend(colors, fragments, self.blendparam)# (BS, H, W, 4)
        IUV = images[:,:,:,:3]#(b,h,w,3)
        # import ipdb; ipdb.set_trace()
        return normalimage, IUV.cpu().numpy()


def fuse_mesh_img_batch(backimg, meshimg, I):
    """
    backimg: (bs, h, w, 3)
    meshimg: (bs, h, w, 3)
    I: (bs, h, w)
    """
    body_mask = (I>0).astype('uint8')
    body_mask = np.expand_dims(body_mask, axis=3)
    fuseimg =  body_mask*meshimg+(1-body_mask)*backimg
    return fuseimg

class VisMesh():
    def __init__(self, vispath, batch_size, device, mesh_gt='r', mesh_pred='g', vis_proxy=False):
        self.visdir = f'{configs.VIS_DIR}/{vispath}'
        if not os.path.isdir(self.visdir):
            os.makedirs(self.visdir)
        
        self.renderer= VisRenderer(batch_size, device=device)
        #Dynamically updated
        self.crop_img = None
        self.whole_img = None #Require bbox for fusion


    def forward_verts(self, pred_verts, target_verts, pred_cam_wp):
        meshimg, IUV = self.renderer.forward(pred_verts, pred_cam_wp, key='pred')
        if self.crop_img is not None:
            meshimg = fuse_mesh_img_batch(self.crop_img, meshimg, IUV[:,:,:,0])
        
        if target_verts is not None:
            meshimg2, IUV = self.renderer.forward(target_verts, pred_cam_wp, key='gt')
            if self.crop_img is not None:
                meshimg2 = fuse_mesh_img_batch(self.crop_img, meshimg, IUV[:,:,:,0])
            meshimg = np.concatenate([meshimg, meshimg2], axis=2) #(bs, h, 2w, 3)
        
        
    

    def fuse_mesh_wholeimg(self, images, bboxs): ##
        if bbox is not None:
            x0, y0, x1, y1 = bbox[0]
            org_h, org_w =(y1-y0).item(), (x1-x0).item()
            max_org_hw = max(org_h,org_w)
            #target bbox
            if org_h>org_w:
                border_w = (org_h-org_w)//2 
                border_h = 0
            else:
                border_w = 0
                border_h = (org_w-org_h)//2
            
            square_start_x = x0 - border_w
            square_start_y = y0 - border_h
            square_end_x = square_start_x + max_org_hw
            square_end_y = square_start_y + max_org_hw


            bbox_image = cv2.resize(normalimage, (max_org_hw, max_org_hw), interpolation = cv2.INTER_NEAREST)
            # bbox_image = bbox_image[border_h:border_h+org_h, border_w:border_w+org_w, :]
            normalimage = np.zeros_like(image)
            
            body_bbox_mask = cv2.resize(body_mask, (max_org_hw, max_org_hw), interpolation = cv2.INTER_NEAREST)
            # body_bbox_mask = body_bbox_mask[border_h:border_h+org_h, border_w:border_w+org_w]
            body_mask = np.zeros_like(image[:,:,0])
            #overpass border?  
            img_h, img_w = image.shape[:2]
            img_start_x = max(0, square_start_x)#>=0
            img_start_y = max(0, square_start_y)
            img_end_x = min(img_w, square_end_x)
            img_end_y = min(img_h, square_end_y)

            shift_start_x = min(0, square_start_x)#<=0
            shift_start_y = min(0, square_start_y)
            shift_end_x = img_w - max(img_w, square_end_x)
            shift_end_y = img_h - max(img_h, square_end_y)
            
            normalimage[img_start_y:img_end_y, img_start_x:img_end_x, :] = \
                bbox_image[-shift_start_y:max_org_hw+shift_end_y, -shift_start_x:max_org_hw+shift_end_x, :]
            body_mask[img_start_y:img_end_y, img_start_x:img_end_x] = \
                body_bbox_mask[-shift_start_y:max_org_hw+shift_end_y, -shift_start_x:max_org_hw+shift_end_x]