
import os, cv2, torch
import numpy as np
import configs
from utils.renderer import P3dRenderer
from utils.vis_utils import hsv_to_bgr_batch, draw_j2d_batch

from pytorch3d.structures import Meshes 
from pytorch3d.renderer.cameras import OrthographicCameras
from pytorch3d.renderer import MeshRasterizer
    
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
        
        #color: r[1,2,0], g[2,0,1], b[0,1,2]
        self.rgb_encoding = {'gt':[0,1,2], 'pred':[2,0,1]}

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

        # focal_length: Focal length of the camera in world units.
        # A tensor of shape (N, 1) or (N, 2) for square and non-square pixels respectively.
        # principal_point: xy coordinates of the center of the principal point of the camera in pixels.
        # A tensor of shape (N, 2).
        focal_length = pred_cam[:,0]
        t_x, t_y = -pred_cam[:,1]*focal_length, -pred_cam[:,2]*focal_length
        principle_point = torch.cat([t_x[:,None], t_y[:,None]], dim=1)
        cameras = OrthographicCameras(focal_length=focal_length, 
                                    principal_point=principle_point, 
                                    R = self.camR,
                                    T = self.camT,
                                    device = self.device, 
                                    in_ndc = True,
                                    image_size=((self.img_wh, self.img_wh),) # image_size=((-1, -1),)
                                    )
        
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings)
        fragments = rasterizer(mesh_batch)
        normal_map = self.get_pixel_normals(mesh_batch, fragments)
        normalimage = 0.5+normal_map/2
        normalimage = (255*normalimage.cpu().numpy()).astype('uint8')
        normalimage = normalimage[:,:,:,self.rgb_encoding[key]]
        # depth_map = fragments.zbuf.squeeze()
        
        colors = mesh_batch.sample_textures(fragments)#(BS, H, W, 1, 4)
        images = hard_rgb_blend(colors, fragments, self.blendparam)# (BS, H, W, 4)
        IUV = images[:,:,:,:3]#(b,h,w,3)
        return normalimage, IUV.cpu().numpy()




class VisMesh():
    def __init__(self, visdir, batch_size, device, visnum_per_batch):
        self.visdir = visdir
        print("Saved Dir:", visdir)
        self.input_batch_size = batch_size

        visnum_per_batch = min(batch_size, visnum_per_batch)
        interval = batch_size//visnum_per_batch
        self.vis_idx_in_batch = [interval*n for n in range(visnum_per_batch)]
        self.render_batch_size = visnum_per_batch

        self.renderer= VisRenderer(visnum_per_batch, device=device)

        #Dynamically updated
        self.crop_img = None
        self.whole_img = None #Require bbox for fusion
        self.iuv = None
        self.j2d = None

    def extract_batch_vis_idx(self, x):
        assert x.shape[0] == self.input_batch_size
        x = x[self.vis_idx_in_batch]        
        return x 

    def fuse_img_batch(self, image, I):
        """
        numpy
        backimg: (bs, h, w, 3) 
        meshimg: (bs, h, w, 3)
        I:  (bs, h, w)
        """
        if self.crop_img is not None:
            body_mask = (I>0).astype('uint8')
            body_mask = np.expand_dims(body_mask, axis=3)
            image =  body_mask*image+(1-body_mask)*self.crop_img
        return image

    def forward_verts(self, pred_verts, target_verts, pred_cam_wp, n_batch): 
        # bs of last batch < defined batch_size
        if pred_verts.shape[0] != self.input_batch_size:
            return
        
        if self.crop_img is not None:
            self.crop_img = self.extract_batch_vis_idx(self.crop_img)
        pred_verts = self.extract_batch_vis_idx(pred_verts)
        pred_cam_wp = self.extract_batch_vis_idx(pred_cam_wp)
        
        meshimg, reproj_IUV = self.renderer.forward(pred_verts, pred_cam_wp, key='pred')
        meshimg = self.fuse_img_batch(meshimg, reproj_IUV[:,:,:,0])
        
        #if GT mesh is valid
        if target_verts is not None:
            target_verts = self.extract_batch_vis_idx(target_verts)
            meshimg2, reproj_IUV = self.renderer.forward(target_verts, pred_cam_wp, key='gt')
            meshimg2 = self.fuse_img_batch(meshimg2, reproj_IUV[:,:,:,0])
            meshimg = np.concatenate([meshimg, meshimg2], axis=2) #(bs, h, 2w, 3)

        # if input iuv/j2d is valid
        if self.iuv is not None:
            iuv = self.extract_batch_vis_idx(self.iuv)
            iuvimg = hsv_to_bgr_batch(iuv).cpu().numpy()
            iuvimg = (iuvimg*255).astype('uint8')
            iuvimg = self.fuse_img_batch(iuvimg, iuv[:,:,:,0].cpu().numpy())
            meshimg = np.concatenate([iuvimg, meshimg], axis=2) 

        if self.j2d is not None:
            j2d = self.extract_batch_vis_idx(self.j2d)
            j2dimg = draw_j2d_batch(j2d.cpu().numpy(), self.crop_img, addText=False, 
                                    H=configs.REGRESSOR_IMG_WH, W=configs.REGRESSOR_IMG_WH)
            meshimg = np.concatenate([j2dimg, meshimg], axis=2) 

        for b in range(self.render_batch_size):
            org_index_in_batch = self.vis_idx_in_batch[b]
            cv2.imwrite(f'{self.visdir}/{n_batch*self.input_batch_size+org_index_in_batch}.png', meshimg[b])
        
        return
    
    
