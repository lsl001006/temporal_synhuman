import numpy as np
import os, torch

def load_training_info_from_checkpoint(checkpoint, save_val_metrics):
    current_epoch = checkpoint['epoch'] + 1
    best_epoch = checkpoint['best_epoch']
    best_model_wts = checkpoint['best_model_state_dict']
    best_epoch_val_metrics = checkpoint['best_epoch_val_metrics']
    # ^ best val metrics, happened at best_epoch

    # If different save_val_metrics used upon re-starting training, set best values for those
    # metrics to infinity.
    for metric in save_val_metrics:
        if metric not in best_epoch_val_metrics.keys():
            best_epoch_val_metrics[metric] = np.inf
    metrics_to_del = [metric for metric in best_epoch_val_metrics.keys() if
                      metric not in save_val_metrics]
    for metric in metrics_to_del:
        del best_epoch_val_metrics[metric]

    print('\nTraining information loaded from checkpoint.')
    print('Current epoch:', current_epoch)
    print('Best epoch val metrics from last training run:', best_epoch_val_metrics,
          ' - achieved in epoch:', best_epoch)

    return current_epoch, best_epoch, best_model_wts, best_epoch_val_metrics

def load_ckpt_woSMPL(ckpt_dir, device, epoch=None, loadbest=True):
    if epoch is None:
        ckpt_files = os.listdir(ckpt_dir)
        if len(ckpt_files)==0:
            return {}
        ckpt_epochs = [int(f[5:-4]) for f in ckpt_files] #'epochXXX.tar' -> 'XXX' 
        epoch = sorted(ckpt_epochs)[-1]    
    ckpt_path = f'{ckpt_dir}/epoch{epoch}.tar'
    if not os.path.exists(ckpt_path):
        return {}
    print("Loading", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_key = 'best_model_state_dict' if loadbest else 'model_state_dict' 

    new_state_dict = {}
    for key, value in checkpoint[model_key].items():
        if not key.startswith('smpl_model'):
            new_state_dict[key] = value
    return new_state_dict





