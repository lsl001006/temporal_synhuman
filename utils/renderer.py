from typing import NamedTuple, Sequence
import torch
import torch.nn as nn
import numpy as np
import configs
#utils
from .uv_utils import IUV_Densepose
from .cam_utils import get_intrinsics_matrix
#pytorch3D
from pytorch3d.renderer import (
    RasterizationSettings, 
    MeshRasterizer, 
    HardPhongShader,
    TexturesUV,
    PointLights
)
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.structures import Meshes 
from pytorch3d.renderer.mesh import Textures
from pytorch3d.renderer.blending import hard_rgb_blend

class BlendParams(NamedTuple):
            sigma: float = 1e-4
            gamma: float = 1e-4
            background_color: Sequence = (0.0, 0.0, 0.0)

class P3dRenderer(nn.Module):
    def __init__(self,
                 batch_size,
                 cam_K,
                 cam_R=None,
                 device='cpu',
                 img_wh=256,
                 render_options={'depth':1, 'normal':1, 'iuv':1, 'j2D':1, 'rgb':1}):
        """
        :param batch_size
        :param cam_K: (bs, 3, 3) camera intrinsics matrix
        :param cam_R: (bs, 3, 3) camera rotation matrix (usually identity).
        :param img_wh: output render width/height
        :param rend_parts_seg: if True, render 6 part segmentation, else render RGB.
        """
        super(P3dRenderer, self).__init__()
        
        self.render_options = render_options
        self.device = device
        self.batch_size = batch_size
        if cam_K.ndim != 3:
            print("Expanding cam_K and cam_R by batch size.")
            cam_K = cam_K[None, :, :].expand(batch_size, -1, -1)
        self.focal_x, self.focal_y, self.t_x, self.t_y = cam_K[0,0,0], cam_K[0,1,1], cam_K[0,0,2], cam_K[0,1,2]
        self.cam_R = self.rectify_cam_R(cam_R)
        self.cam_K = cam_K
        self.img_wh = img_wh
        
        #load Densepose
        self.load_densepose(img_wh, batch_size, device)

        if render_options.get('rgb'):
            self.lights_rgb_render = PointLights(device=device,
                                                 location=((0.0, 0.0, -2.0),),
                                                 ambient_color=((0.5, 0.5, 0.5),),
                                                 diffuse_color=((0.3, 0.3, 0.3),),
                                                 specular_color=((0.2, 0.2, 0.2),))


    def load_densepose(self, img_wh, batch_size, device):
        IUV_processed = IUV_Densepose(device=device)
        self.faces, self.face_indices = IUV_processed.get_faces()
        self.I = IUV_processed.get_I() #(7829,1) 
        self.U, self.V = IUV_processed.get_UV() #(7829,1)
        self.IUVnorm= torch.cat([self.I/24, self.U, self.V], dim=1)
        self.verts_uv_offset = IUV_processed.offsetUV()[None].expand(batch_size, -1, -1) #(batch_size,7829,2)

        self.faces_list=[self.faces for _ in range(batch_size)]#(bs,13774,3)
        self.IUVnorm_list = [self.IUVnorm for _ in range(batch_size)] #(bs, 7829, 3)
        self.to_1vertex_id = (IUV_processed.ALP_UV['All_vertices'][0]-1).tolist()#.to(device) #(7829) differential?
        self.blendparam = BlendParams()  
        self.raster_settings = RasterizationSettings(
            image_size=img_wh,
            blur_radius=0.0,
            faces_per_pixel=1, #topK value
        )
        

    def rectify_cam_R(self, cam_R): #no augmentation
        calibrate_R = torch.tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])#Rz(180)
        calibrate_R = calibrate_R[None].expand(self.batch_size,3,3).to(self.device)
        if cam_R is None:
            cam_R_trans = calibrate_R
        else:
            if cam_R.ndim != 3:
                cam_R = cam_R[None, :, :].expand(self.batch_size, 3, 3)
            cam_R_trans = torch.einsum('bij,bjk->bik', calibrate_R, cam_R)
        return cam_R_trans

    def rectify_cam_T(self, cam_T):
        if cam_T is None:
            cam_T = torch.tensor([0., 0., 0.])
            cam_T = cam_T[None].expand(self.batch_size,-1).to(self.device)
        else:
            calibrate_T = torch.tensor([-1, -1, 1])#Rz(180)
            calibrate_T = calibrate_T[None].expand(self.batch_size,-1).to(self.device)
            cam_T = calibrate_T*cam_T
        return cam_T

    def forward(self, vertices, cam_T=None, textures=None):
        """
        :param vertices: (B, N, 3)
        :param cam_T: (B,  3)
        """
        cam_T = self.rectify_cam_T(cam_T)
        #should be differential?
        unique_verts = [vertices[:,vid].cpu().numpy() for vid in self.to_1vertex_id] #(7829,bs,3)
        unique_verts = torch.as_tensor(np.array(unique_verts)).transpose(0,1).to(self.device) #(bs, 7829, 3)

        #mesh
        verts_list = [unique_verts[nb] for nb in range(self.batch_size)] #(bs, 7829, 3)
        mesh_batch = Meshes(verts=verts_list, faces=self.faces_list, textures = Textures(verts_rgb=self.IUVnorm_list))##norm I to [0,1]?
        
        #render
        cameras = PerspectiveCameras(
            focal_length=((self.focal_x, self.focal_y),),
            principal_point=((self.t_x, self.t_y),),
            R = self.cam_R,
            T = cam_T,
            device = self.device, 
            in_ndc = False,
            image_size=((self.img_wh,self.img_wh),)
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings)
        fragments = rasterizer(mesh_batch)
        colors = mesh_batch.sample_textures(fragments)#(BS, H, W, 1, 4)
        images = hard_rgb_blend(colors, fragments, self.blendparam)# (BS, H, W, 4)
        I_map = images[:,:,:,0]#24 unique normalized
        U_map = images[:,:,:,1]#nearest, require interpolation?
        V_map = images[:,:,:,2]#nearest, require interpolation?

        # background_ts = torch.tensor(self.blendparam.background_color).to(self.device)
        # background_mask = (images[:,:,:,:3] ==background_ts).all(dim=3)# (BS, H, W)
        #background_depth = (depth_map <= 0)
        depth_map, normal_map, iuv_map, rgb_images = None, None, None, None
        if self.render_options.get('depth'):
            depth_map = fragments.zbuf.squeeze() # (BS, H, W) #[-1,36...47]
            # body_depth_min = depth[depth>0].min()
            # body_depth_max = depth[depth>0].max()
        if self.render_options.get('normal'):
            normal_map = self.get_pixel_normals(mesh_batch, fragments)
        if self.render_options.get('iuv'):
            iuv_map = torch.stack([I_map, U_map, V_map],dim=3)
        if self.render_options.get('rgb') and (textures is not None):
            meshes_rgb = Meshes(verts=verts_list, 
                                faces=self.faces_list, 
                                textures=TexturesUV(maps=textures, faces_uvs=self.faces_list, verts_uvs=self.verts_uv_offset))
            fragments = rasterizer(meshes_rgb)
            
            rgb_shader = HardPhongShader(device=self.device, cameras=cameras,
                                              lights=self.lights_rgb_render, blend_params=self.blendparam)
            rgb_images = rgb_shader(fragments, meshes_rgb, lights=self.lights_rgb_render)[:, :, :, :3]
            rgb_images = torch.clamp(rgb_images, max=1.0)
        return  depth_map, normal_map, iuv_map, rgb_images
        

    def get_pixel_normals(self, mesh_batch, fragments):
        vertex_normals = mesh_batch.verts_normals_packed() #1096060=140*7829
        faces = mesh_batch.faces_packed()  # (bs*F, 3)
        faces_normals = vertex_normals[faces]

        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_normals
        )
        return pixel_normals.squeeze(dim=-2)


# Camera and joints/silhouette renderer
def build_cam_renderer(batch_size, device, datatype='normal', render_options={'depth':1, 'normal':1, 'iuv':1, 'j2D':1, 'rgb':1}):
    # Assuming camera rotation is identity (since it is dealt with by global_orients in SMPL)
    mean_cam_t = np.array(configs.MEAN_CAM_T)
    mean_cam_t = torch.from_numpy(mean_cam_t).float().to(device)
    mean_cam_t = mean_cam_t[None, :].expand(batch_size, -1)

    cam_K = get_intrinsics_matrix(configs.REGRESSOR_IMG_WH, configs.REGRESSOR_IMG_WH, configs.FOCAL_LENGTH)
    # import ipdb; ipdb.set_trace()
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(batch_size, -1, -1)

    #adjust Mocap to stand up
    if datatype=='mocap':
        cam_R = torch.tensor([[1,0,0],[0,0,-1],[0,1,0]]).to(device).float()
    else:
        cam_R = torch.eye(3).to(device)

    cam_R = cam_R[None, :, :].expand(batch_size, -1, -1)
 
    renderer = P3dRenderer(batch_size,
                cam_K,
                cam_R=cam_R,
                device=device,
                render_options=render_options)
    return mean_cam_t, cam_K, cam_R, renderer