import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .resnet import resnet18, resnet50
from .ief_module import IEFModule, SpatialIEFModule
from einops import rearrange
import configs
from utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat
from .vibe_smpl import SMPL_MEAN_PARAMS, SMPL, SMPL_MODEL_DIR, H36M_TO_J14

class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            input_size = 2048,
            hidden_size = 2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True
    ):
        super(TemporalEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )
        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size*2, input_size)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, input_size)
        self.use_residual = use_residual
        
    def forward(self, x):
        n,t,f = x.shape # n=batchsize, t=seqlen, f=features vector
        x = x.permute(1,0,2) # NTF -> TNF
        y, _ = self.gru(x)
        if self.linear:
            y = F.relu(y) # seql*bs*hidden_size
            # import pdb;pdb.set_trace()
            y = self.linear(y.view(-1, y.size(-1))) # [seql*bs,512]
            y = y.view(t,n,f)
        if self.use_residual and y.shape[-1] == self.input_size:
            y = y + x
        y = y.permute(1,0,2) # TNF -> NTF
        return y

class SingleInputRegressor(nn.Module):
    """
    Combined encoder + regressor model that takes proxy representation input (e.g.
    silhouettes + 2D joints) and outputs SMPL body model parameters + weak-perspective
    camera.
    """
    def __init__(self,
                 resnet_in_channels=1,
                 ief_iters=3,
                 itersup=False,
                 reginput_ch=512,
                 reginput_hw=1,
                 encoddrop=False):
        """
        :param resnet_in_channels: 1 if input silhouette/segmentation, 1 + num_joints if
        input silhouette/segmentation + joints.
        :param ief_iters: number of IEF iterations.
        """
        super(SingleInputRegressor, self).__init__()
        IEFfunction = SpatialIEFModule if reginput_hw>1 else IEFModule
        self.itersup = itersup
        num_pose_params = 24*6
        num_output_params = 3 + num_pose_params + 10
        self.cam_params, self.pose_params, self.shape_params = [], [], []
        self.reginput_ch = reginput_ch
        self.reginput_hw = reginput_hw
        #channel
        filter_channels = [512]#[512,128,32,8,2,1]
        if reginput_ch!=512:
            filter_channels = [512, reginput_ch] if reginput_ch>=32 else [512, 32, reginput_ch]

        #height,width
        if reginput_hw==16:
            pool1 = False
        else:
            filter_channels =[512, 256, 64, 8]
            ief_channel = 512
            pool1 = True
        
        ief_channel = reginput_ch*reginput_hw*reginput_hw

        self.image_encoder = resnet18(in_channels=resnet_in_channels, 
                                        drop = encoddrop,
                                        pretrained=False, 
                                        pool1=pool1,
                                        pool_out=False, 
                                        )
        
        self.temporal_encoder = TemporalEncoder(n_layers=2,  # 2
                                                input_size=ief_channel,
                                                hidden_size=ief_channel, 
                                                add_linear=True, 
                                                use_residual=True)
        
        self.ief_module = IEFfunction([ief_channel, ief_channel],#[1024,1024],#[512, 512],
                                    ief_channel,#1024, #512,
                                    num_output_params,
                                    iterations=ief_iters)
        
        self.vibe_regressor = VIBERegressor(smpl_mean_params=SMPL_MEAN_PARAMS)
        
        if len(filter_channels)>1:
            self.add_mlp(filter_channels = filter_channels)
        
           
    def add_mlp(self, filter_channels =[512, 256, 64, 8], module_name='rdconv'):#[512, 256, 64, 4]
        self.filters = []
        for l in range(len(filter_channels)-1):
            if 0 != l:
                self.filters.append(
                    nn.Conv1d(
                        filter_channels[l] + filter_channels[0],
                        filter_channels[l + 1],
                        1))
            else:
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))

            # self.add_module("rdconv%d" % l, self.filters[l])
            self.add_module(f"{module_name}{l}", self.filters[l])


    def reduce_dim(self, feature, module_name='rdconv'):
        '''
        Dimension reduction by multi-layer perceptrons
        :param feature: list of [B, C_s, N] point-wise features before dimension reduction
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        '''
        y = feature
        tmpy = feature
        # self.num_views = 1

        # import ipdb; ipdb.set_trace()
        out_feats = [feature]
        for i, f in enumerate(self.filters):
            y = self._modules[module_name + str(i)](
                y if i == 0
                else torch.cat([y, tmpy], 1)
            )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
            else:
                y = nn.ReLU(True) (y)
            out_feats.append(y)
            
        return out_feats

    def forward(self, input):
        
        input_feats, _, _ = self.image_encoder(input)
        
        #keep channel, compress HW
        if self.reginput_hw<8:
            input_feats = nn.AdaptiveAvgPool2d((self.reginput_hw, self.reginput_hw))(input_feats)
        #keep HW, compress channel
        if self.reginput_ch<512: 
            input_feats = input_feats.view(input_feats.shape[0], input_feats.shape[1],-1) #(bs, ch, hw)
            input_feats = self.reduce_dim(input_feats)[-1]#only use the last one #(bs, ch_target, hw)
            
        
        # rearrange input_feats to [bs, seql, feats]
        input_feats = rearrange(input_feats, '(b1 b2) f1 f2 f3 -> b1 b2 (f1 f2 f3)', b2=configs.SEQLEN)
        
        # add temporal encoder FIXME!
        feats = self.temporal_encoder(input_feats)
        
        # rearrange input_feats to [bs*seql, feats]
        input_feats = rearrange(feats, 'b1 b2 f -> (b1 b2) f')
        
        # replace ief_module with vibe regressor FIXME!
        # cam_params, pose_params, shape_params = self.ief_module(input_feats)
        # input_feats:[bs*seql, 512*ch*cw]
        cam_params, pose_params, shape_params = self.vibe_regressor(input_feats)
        # import pdb;pdb.set_trace()
        
        
        self.cam_params, 
        self.pose_params, 
        self.shape_params = [], [], []
        if not self.itersup:
            return [cam_params[-1]],  [pose_params[-1]],  [shape_params[-1]]
        return [cam_params], [pose_params], [shape_params]


class MultiScaleRegressor(SingleInputRegressor):
    def __init__(self,
                 resnet_in_channels=1,
                 ief_iters=3,
                 itersup=False,
                 filter_channels =[512, 256, 32, 8],
                 encoddrop=False):
        super(MultiScaleRegressor, self).__init__()
        pool_out = False
        self.itersup = itersup
        num_pose_params = 24*6
        num_output_params = 3 + num_pose_params + 10

        
        self.image_encoder = resnet18(in_channels=resnet_in_channels, drop = encoddrop,
                                        pretrained=False, pool_out=pool_out)
        
        

        self.add_mlp(filter_channels=filter_channels)
        self.ief_module_1 = SpatialIEFModule([512, 512],
                                    512,
                                    num_output_params,
                                    iterations=1)
        self.ief_module_2 = SpatialIEFModule([512, 512],
                                    512,
                                    num_output_params,
                                    iterations=1)
        self.ief_module_3 = SpatialIEFModule([512, 512],
                                    512,
                                    num_output_params,
                                    iterations=1)
        
    def forward(self, input, channel_used=[0,2,3]):
        input_feats = self.image_encoder(input)[0]
        # import ipdb; ipdb.set_trace()
        input_feats = input_feats.view(input_feats.shape[0], input_feats.shape[1],-1)#(bs,512,8*8)
        # import ipdb; ipdb.set_trace()
        out_feats = self.reduce_dim(input_feats)#only use the last one
        params_estimate = None

        cam_params, pose_params, shape_params = [], [], []
        for n,c in enumerate(channel_used):
            feat = out_feats[c] #(bs,c,8*8)
            feat = feat.reshape(feat.shape[0], feat.shape[1], 8, 8)
            target_wh = np.sqrt(512//feat.shape[1]).astype('int')
            if not target_wh==8:
                feat = nn.AdaptiveAvgPool2d((target_wh, target_wh))(feat)
            feat = feat.view(feat.shape[0], -1)#(bs,512)
            # import ipdb; ipdb.set_trace()
           
            cam_param, pose_param, shape_param = self._modules['ief_module_' + str(n+1)](feat, params_estimate=params_estimate)
            params_estimate = torch.cat([cam_param[0], pose_param[0], shape_param[0]], dim=1)
            cam_params.append(cam_param[0])
            pose_params.append(pose_param[0])
            shape_params.append(shape_param[0])
        
        if not self.itersup:
            cam_params, pose_params, shape_params = [cam_params[-1]],  [pose_params[-1]],  [shape_params[-1]]
        return cam_params, pose_params, shape_params
        

class VIBERegressor(nn.Module):
    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS):
        super(VIBERegressor, self).__init__()

        npose = 24 * 6

        self.fc1 = nn.Linear(512 * 4 + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False
        )

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)



    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3, J_regressor=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)
        # import pdb;pdb.set_trace() 
        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        if J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            pred_joints = pred_joints[:, H36M_TO_J14, :]

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        # output = [{
        #     'theta'  : torch.cat([pred_cam, pose, pred_shape], dim=1),
        #     'verts'  : pred_vertices,
        #     'kp_2d'  : pred_keypoints_2d,
        #     'kp_3d'  : pred_joints,
        #     'rotmat' : pred_rotmat
        # }]
        return pred_cam, pose, pred_shape

def projection(pred_joints, pred_camera):
    pred_cam_t = torch.stack([pred_camera[:, 1],
                              pred_camera[:, 2],
                              2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
                                               translation=pred_cam_t,
                                               focal_length=5000.,
                                               camera_center=camera_center)
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
    return pred_keypoints_2d

def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

