import logging
from .mutil_loss import HomoscedasticUncertaintyWeightedMultiTaskLoss

def Build_Loss(var=False, shapeloss=False, vertloss=True):
    # Initial loss weights - these will be updated during training.
    init_loss_weights = {'verts': 1.0, 
                         'joints3D': 1.0,
                         'joints2D': 0.1, 
                         'pose_params': 0.1, 
                         'shape_params': 0.1,
                        }
    # Loss on
    # losses_on = ['pose_params', 'joints2D', 'joints3D']
    # 
    if not shapeloss:
        del init_loss_weights['shape_params']

    if not vertloss:
        del init_loss_weights['verts']

    logging.info(f"Loss weights:  {init_loss_weights}")
    
    task_criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(
                                                            init_loss_weights=init_loss_weights,
                                                            reduction='mean',
                                                            var=var)
    losses_to_track = list(init_loss_weights.keys())
    return task_criterion, losses_to_track