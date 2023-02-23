import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .resnet import resnet18, resnet50
from .ief_module import IEFModule, SpatialIEFModule
from einops import rearrange
import configs

class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True
    ):
        super(TemporalEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )
        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size*2, 2048)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
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
        if self.use_residual and y.shape[-1] == 2048:
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
        
        self.temporal_encoder = TemporalEncoder(n_layers=2, 
                                                hidden_size=2048, 
                                                add_linear=True, 
                                                use_residual=True)
        
        self.ief_module = IEFfunction([ief_channel, ief_channel],#[1024,1024],#[512, 512],
                                    ief_channel,#1024, #512,
                                    num_output_params,
                                    iterations=ief_iters)
        
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
        
        cam_params, pose_params, shape_params = self.ief_module(input_feats)
        
        if not self.itersup:
            cam_params, pose_params, shape_params = [cam_params[-1]],  [pose_params[-1]],  [shape_params[-1]]
        
        self.cam_params, 
        self.pose_params, 
        self.shape_params = [], [], []
        return cam_params, pose_params, shape_params


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
        
            


