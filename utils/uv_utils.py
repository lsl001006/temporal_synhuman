import numpy as np
import torch
from scipy.io import loadmat
import configs

class IUV_Densepose():
    def __init__(self, mat_path=configs.UV_MAT_PATH, device='cpu'):
        self.ALP_UV = loadmat(mat_path)
        self.N_face = 13774
        self.N_vertex = 7829
        self.device = device

    def get_faces(self):
        faces = torch.from_numpy((self.ALP_UV['All_Faces'] - 1).astype(int))  # (13774, 3), indicates vertex id [0...7828] for each face
        face_indices  = torch.Tensor(self.ALP_UV['All_FaceIndices']).squeeze()  # (13774,), indicates classn [1...24]
        return faces.to(self.device), face_indices.to(self.device)

    def get_UV(self):
        U = torch.Tensor(self.ALP_UV['All_U_norm']).to(self.device) # (7829, 1)
        V = torch.Tensor(self.ALP_UV['All_V_norm']).to(self.device) # (7829, 1)
        return U,V
    
    def get_I(self):
        faces, face_indices =  self.get_faces()
        I = []
        for n in range(self.N_vertex):
            faces_id = torch.where(faces==n)#[n_face,3], all the faces are for one classn
            faces_id = faces_id[0][0]
            # print(faces_id)
            I.append(face_indices[faces_id])
        I = torch.as_tensor(I).unsqueeze(dim=1)# (7829,1)
        return I.to(self.device)

    def offsetUV(self):
        U, V = self.get_UV()
        faces, face_indices = self.get_faces()
        # Map each face to a (u, v) offset
        offset_per_part = {}
        already_offset = set()
        cols, rows = 4, 6
        for i, u in enumerate(np.linspace(0, 1, cols, endpoint=False)):
            for j, v in enumerate(np.linspace(0, 1, rows, endpoint=False)):
                part = rows * i + j + 1  # parts are 1-indexed in face_indices
                offset_per_part[part] = (u, v)

        U_norm = U.clone()
        V_norm = V.clone()

        # iterate over faces and offset the corresponding vertex u and v values
        for i in range(len(faces)):
            face_vert_idxs = faces[i]
            part = face_indices[i]
            offset_u, offset_v = offset_per_part[int(part.item())]
            
            for vert_idx in face_vert_idxs:   
                # vertices are reused, but we don't want to offset multiple times
                if vert_idx.item() not in already_offset:
                    # offset u value
                    U_norm[vert_idx] = U[vert_idx] / cols + offset_u
                    # offset v value
                    # this also flips each part locally, as each part is upside down
                    V_norm[vert_idx] = (1 - V[vert_idx]) / rows + offset_v
                    # add vertex to our set tracking offsetted vertices
                    already_offset.add(vert_idx.item())

        # invert V values
        V_norm = 1 - V_norm
        verts_uv = torch.cat([U_norm, V_norm], dim=1) #(7829,2)
        # import ipdb; ipdb.set_trace()
        return verts_uv