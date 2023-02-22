import torch
import numpy as np
from smplx import SMPL as _SMPL
# from smplx.body_models import ModelOutput
from smplx.lbs import vertices2joints

import configs


class SMPL(_SMPL):
    """
    Extension of the official SMPL (from the smplx python package) implementation to
    support more joints.
    """
    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        J_regressor_extra = np.load(configs.J_REGRESSOR_EXTRA_PATH)
        J_regressor_cocoplus = np.load(configs.COCOPLUS_REGRESSOR_PATH)
        J_regressor_h36m = np.load(configs.H36M_REGRESSOR_PATH)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra,
                                                               dtype=torch.float32))
        self.register_buffer('J_regressor_cocoplus', torch.tensor(J_regressor_cocoplus,
                                                                  dtype=torch.float32))
        self.register_buffer('J_regressor_h36m', torch.tensor(J_regressor_h36m,
                                                              dtype=torch.float32))

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        cocoplus_joints = vertices2joints(self.J_regressor_cocoplus, smpl_output.vertices)
        h36m_joints = vertices2joints(self.J_regressor_h36m, smpl_output.vertices)
        
        all_joints = torch.cat([smpl_output.joints, extra_joints, cocoplus_joints,
                                h36m_joints], dim=1)
        #45,9,17
        
        
        return smpl_output.vertices, all_joints 
        """shapes of some variables:
        smpl_output.vertices: [batchsize*6890*3]
        all_joints:[batchsize*90*3]
            smpl_output.joints: [batchsize*45*3]
            extra_joints: [batchsize*9*3]
            cocoplus_joints: [batchsize*19*3]
            h36m_joints: [batchsize*17*3]
        """
        
