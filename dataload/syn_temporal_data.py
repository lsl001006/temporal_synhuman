import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticTempoData(Dataset):
    def __init__(self, npz_path, seqlen):
        data = np.load(npz_path)
        self.seqlen = seqlen
        self.poses = data['theta']
        self.shapes = data['beta']

    def __len__(self):
        return len(self.poses)//self.seqlen

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        index = index*self.seqlen
        pose = self.poses[index:index+self.seqlen]
        shape = self.shapes[index:index+self.seqlen]
        
        pose = torch.from_numpy(pose.astype(np.float32))
        shape = torch.from_numpy(shape.astype(np.float32))

        # assert pose.shape == (72,) and shape.shape == (10,)#23+1

        return {'pose': pose,
                'shape': shape}
