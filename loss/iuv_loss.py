import torch
import torch.nn.functional as F
import torch.nn.functional as nnf

def iuv2IUVAGT(iuv, resize_hw):
    IUV= nnf.interpolate(iuv.permute(0,3,1,2), size=(resize_hw, resize_hw), mode='nearest')

    Index2mask = [[0], [1, 2], [3], [4], [5], [6], [7, 9], [8, 10], [11, 13], [12, 14], [15, 17], [16, 18], [19, 21],
                  [20, 22], [23, 24]]

    recon_I = torch.round(IUV[:,0,:,:] * 24)
    

    recon_U = []
    recon_V = []
    recon_Index_UV = []#one-hot
    recon_Ann_Index = []

    for i in range(25):
        recon_Index_UV_i = (recon_I==i)
        recon_U_i = recon_Index_UV_i * IUV[:,1,:,:]
        recon_V_i = recon_Index_UV_i * IUV[:,2,:,:]

        recon_Index_UV.append(recon_Index_UV_i)
        recon_U.append(recon_U_i)
        recon_V.append(recon_V_i)
    
    for i in range(len(Index2mask)):
        if len(Index2mask[i]) == 1:
            recon_Ann_Index_i = recon_Index_UV[Index2mask[i][0]]
        elif len(Index2mask[i]) == 2:
            p_ind0 = Index2mask[i][0]
            p_ind1 = Index2mask[i][1]
            # recon_Ann_Index[:, i, :, :] = torch.where(recon_Index_UV[:, p_ind0, :, :] > 0.5, recon_Index_UV[:, p_ind0, :, :], recon_Index_UV[:, p_ind1, :, :])
            # recon_Ann_Index[:, i, :, :] = torch.eq(recon_I, p_ind0) | torch.eq(recon_I, p_ind1)
            recon_Ann_Index_i = recon_Index_UV[p_ind0] + recon_Index_UV[p_ind1]

        recon_Ann_Index.append(recon_Ann_Index_i)
    
    recon_U = torch.stack(recon_U, dim=1) 
    recon_V = torch.stack(recon_V, dim=1)
    recon_Index_UV = torch.stack(recon_Index_UV, dim=1) #bool
    recon_Ann_Index = torch.stack(recon_Ann_Index, dim=1)#bool
    recon_Ann = torch.argmax(recon_Ann_Index.int(), dim=1)
    
    return recon_U, recon_V, recon_Index_UV, recon_I, recon_Ann


def body_iuv_losses(pred_iuv_dict, labels_iuv, uv_weight=0.5):
    index_pred = pred_iuv_dict['predict_uv_index']#(bs, 25, hw, hw)
    u_pred = pred_iuv_dict['predict_u']#(bs, 25, hw, hw)
    v_pred = pred_iuv_dict['predict_v']#(bs, 25, hw, hw)
    ann_pred = pred_iuv_dict['predict_ann_index']#(bs, 15, hw, hw)
    batch_size, _, _, hw = index_pred.shape

    U_GT, V_GT, I_mask, I_GT, Ann_GT = iuv2IUVAGT(labels_iuv, resize_hw=hw)
    loss_IndexUV = F.cross_entropy(index_pred, I_GT.long(), ignore_index=0)#(bs,c,hw,hw),(bs,hw,hw)

    loss_U = F.smooth_l1_loss(u_pred[I_mask > 0], U_GT[I_mask > 0], reduction='sum') / batch_size
    loss_V = F.smooth_l1_loss(v_pred[I_mask > 0], V_GT[I_mask > 0], reduction='sum') / batch_size
    loss_U *= uv_weight
    loss_V *= uv_weight

    loss_segAnn = F.cross_entropy(ann_pred, Ann_GT, ignore_index=0)
    
    return loss_U+loss_V+loss_IndexUV+loss_segAnn
