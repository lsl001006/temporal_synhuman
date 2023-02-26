LOG_DIR = '../exp_out/temporal_synhuman/logs'
CKPT_DIR =  '../exp_out/temporal_synhuman/ckpts' # comment this for testing singleframe results
# CKPT_DIR='../exp_out/singleframe/ckpts'
VIS_DIR =  '../exp_out/temporal_synhuman/vis'

ROOT_PATH = '/a2il/data/xuangong/3dhmr'
SMPL_MODEL_DIR = f'{ROOT_PATH}/additional/smpl'
SMPL_FACES_PATH = f'{ROOT_PATH}/additional/smpl_faces.npy'
SMPL_MEAN_PARAMS_PATH = f'{ROOT_PATH}/additional/neutral_smpl_mean_params_6dpose.npz'
J_REGRESSOR_EXTRA_PATH = f'{ROOT_PATH}/additional/J_regressor_extra.npy'
COCOPLUS_REGRESSOR_PATH = f'{ROOT_PATH}/additional/cocoplus_regressor.npy'
H36M_REGRESSOR_PATH = f'{ROOT_PATH}/additional/J_regressor_h36m.npy'
VERTEX_TEXTURE_PATH = f'{ROOT_PATH}/additional/vertex_texture.npy'
CUBE_PARTS_PATH = f'{ROOT_PATH}/additional/cube_parts.npy'
UV_MAT_PATH =f'{ROOT_PATH}/additional/UV_Processed.mat'
# ------------------------ Constants ------------------------
# SEQLEN = 16
SEQLEN = 1
FOCAL_LENGTH = 5000.
REGRESSOR_IMG_WH = 256
MEAN_CAM_T = [0., 0.2, 42.]
# ------------------------ SMPL Prior for Train/Val ------------------------
STRAP_TRAIN_PATH = '../data/strap_train.npz'
STRAP_VAL_PATH ='../data/strap_val.npz'

# ------------------------ Test Data --------------------------------------#TODO: h36m SMPL GT
PROCESS_PATH = f'{ROOT_PATH}/ProcessedTest' 

SSP3D_IMG_PATH = f'{ROOT_PATH}/ssp_3d/images'#311
SSP3D_GT = f'{ROOT_PATH}/labels.npz'#311

MPI3DHP_IMG_PATH = f'{ROOT_PATH}/mpii-inf-3dhp'#2875
MPI3DHP_GT = f'{ROOT_PATH}/mpii-inf-3dhp/mpi_inf_3dhp_test.npz'#2875

H36M_IMG_PATH = f'{ROOT_PATH}/h36m'#110233

H36M_P1_GT = f'{ROOT_PATH}/h36m/h36m_valid_protocol1.npz'#109867
# H36M_P1_IMG_DIR = f'{PROCESS_PATH}/h36mp1/image_i0_s1.2'
# H36M_P1_PR_DIR = f'{PROCESS_PATH}/h36mp1/pr_i0_s1.2'

H36M_P2_GT = f'{ROOT_PATH}/h36m/h36m_valid_protocol2.npz'#27558
# H36M_P2_IMG_DIR = f'{PROCESS_PATH}/h36mp2/image_i0_s1.2'
# H36M_P2_PR_DIR = f'{PROCESS_PATH}/h36mp2/pr_i0_s1.2'

D3PW_IMG_PATH = f'{ROOT_PATH}/3dpw/testImgs' #53796
D3PW_GT = f'{ROOT_PATH}/3dpw/3dpw_test.npz' #35515
# D3PW_IMG_DIR = f'{PROCESS_PATH}/3dpw/image_i0_s1.2'
# D3PW_PR_DIR = f'{PROCESS_PATH}/3dpw/pr_i0_s1.2'

# ----------------------Detectron2 Path ----------------------------------

DETECTRON2_PATH = '/home/csgrad/xuangong/hmr/detectron2'