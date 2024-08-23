import os
try:
    import pickle5 as pickle
except:
    import pickle
import shutil

import numpy as np
import random
import torch
from scipy.interpolate import interp1d
from libs.extrapolation_modes import do_extrapolation

global global_seed

global_seed = 123
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)

def _init_fn(worker_id):

    seed = global_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

# convertor
def to_tensor(data):
    return torch.from_numpy(data).cuda()

def to_np(data):
    try:
        return data.cpu().numpy()
    except:
        return data.detach().cpu().numpy()

def to_np2(data):
    return data.detach().cpu().numpy()

def to_3D_np(data):
    return np.repeat(np.expand_dims(data, 2), 3, 2)

def logger(text, LOGGER_FILE):  # write log
    with open(LOGGER_FILE, 'a') as f:
        f.write(text),
        f.close()


# directory & file
def mkdir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


def rmfile(path):
    if os.path.exists(path):
        os.remove(path)

def rmdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

# pickle
def save_pickle(path, data):
    '''
    :param file_path: ...
    :param data:
    :return:
    '''
    mkdir(os.path.dirname(path))
    with open(path + '.pickle', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    with open(file_path + '.pickle', 'rb') as f:
        data = pickle.load(f)

    return data

def prune_3d_lane_by_range(lane_3d, x_min, x_max):
    # remove points with y out of range
    # 3D label may miss super long straight-line with only two points: Not have to be 200, gt need a min-step
    # 2D dataset requires this to rule out those points projected to ground, but out of meaningful range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 1] > 0, lane_3d[:, 1] < 200), ...]

    # remove lane points out of x range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 0] > x_min,
                                     lane_3d[:, 0] < x_max), ...]
    return lane_3d

def projection_g2im_extrinsic(E, K):
    E_inv = np.linalg.inv(E)
    E_inv = E_inv[0:3, :]
    P_g2im = np.matmul(K, E_inv)
    return P_g2im

def projection_g2im_extrinsic_parallel(E, K):
    E_inv = torch.inverse(E)[:, :3, :]
    P_g2im = torch.bmm(K, E_inv)
    return P_g2im


def homography_crop_resize(org_img_size, crop_y, resize_img_size):
    """
    compute the homography matrix transform original image to cropped and resized image
    :param org_img_size: [org_h, org_w]
    :param crop_y:
    :param resize_img_size: [resize_h, resize_w]
    :return:
    """
    # transform original image region to network input region
    ratio_x = resize_img_size[1] / org_img_size[1]
    ratio_y = resize_img_size[0] / (org_img_size[0] - crop_y)
    H_c = np.array([[ratio_x, 0, 0],
                    [0, ratio_y, -ratio_y*crop_y],
                    [0, 0, 1]])
    return H_c

def resample_laneline_in_y(input_lane, y_steps, mode='linear', degree=3, knots=None):
    """
    Interpolate x, z values at each anchor grid, including those beyond the range of input lane y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # # at least two points are included
    # assert(input_lane.shape[0] >= 2)

    y_min = np.min(input_lane[:, 1])
    y_max = np.max(input_lane[:, 1])
    sampling_pts = np.linspace(y_min,y_max,num=20)

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)


    if mode == 'custom_spline':
        x_values, _, z_values = do_extrapolation(input_lane = input_lane, sampling_pts = y_steps, mode = mode, knots = knots, k = degree) 
    else:
        x_values, _, z_values = do_extrapolation(input_lane = input_lane, sampling_pts = sampling_pts, mode = mode, degree= degree) 
    
    # f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    # f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")

    # x_values = f_x(y_steps)
    # z_values = f_z(y_steps)
    
    # 이거 할 필요 없으면 삭제 y steps 범위 어짜피 고정이라
    # output_visibility = np.ones((len(y_steps)),dtype=np.bool_)
    return x_values, sampling_pts, z_values

def projective_transformation(Matrix, x, y, z):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x4 projection matrix
            x (array): original x coordinates
            y (array): original y coordinates
            z (array): original z coordinates
    """
    ones = np.ones((1, len(z)))
    coordinates = np.vstack((x, y, z, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/(trans[2, :] + 1e-8)
    y_vals = trans[1, :]/(trans[2, :] + 1e-8)

    return x_vals, y_vals

def inverse_projective_transformation(Matrix, x_2d, y_2d,scale_vec):
    # at_a_inv_at = np.matmul(np.linalg.inv(np.matmul(Matrix.T,Matrix)),Matrix.T)
    # b= np.concatenate([x_2d.reshape(1,-1)*(scale_vec+1e-8),y_2d.reshape(1,-1)*(scale_vec+1e-8),scale_vec.reshape(1,-1)],axis=0)
    # real_coord = np.matmul(at_a_inv_at,b)
    # x,y,z= real_coord[0,:]/real_coord[3,:],real_coord[1,:]/real_coord[3,:],real_coord[2,:]/real_coord[3,:]
    b= np.concatenate([x_2d.reshape(1,-1)*(scale_vec+1e-8),y_2d.reshape(1,-1)*(scale_vec+1e-8),scale_vec.reshape(1,-1)],axis=0)
    for i in range(b.shape[0]):
        b[i,:] = b[i,:]-Matrix[i,-1]
    a_inv = np.linalg.inv(Matrix[:,:-1])
    real_coord = np.matmul(a_inv,b)
    x,y,z= real_coord[0,:],real_coord[1,:],real_coord[2,:]

    return x, y, z

def inverse_projective_transformation2(Matrix, x_2d, y_2d,sampling_pts):
    # at_a_inv_at = np.matmul(np.linalg.inv(np.matmul(Matrix.T,Matrix)),Matrix.T)
    # b= np.concatenate([x_2d.reshape(1,-1)*(scale_vec+1e-8),y_2d.reshape(1,-1)*(scale_vec+1e-8),scale_vec.reshape(1,-1)],axis=0)
    # real_coord = np.matmul(at_a_inv_at,b)
    # x,y,z= real_coord[0,:]/real_coord[3,:],real_coord[1,:]/real_coord[3,:],real_coord[2,:]/real_coord[3,:]
    x = np.zeros((0,len(x_2d)))
    y = np.zeros((0,1))
    z = np.zeros((0,len(x_2d)))
    A = Matrix[:,:3]
    k = Matrix[:,3].reshape(-1,1)
    b = np.concatenate([x_2d.reshape(-1,1),y_2d.reshape(-1,1),np.ones((x_2d.shape)).reshape(-1,1)],axis=1).T
    A_inv = np.linalg.inv(A)
    for idx,y_sample in enumerate(sampling_pts):
        cur_b = b.copy()
        cur_b = cur_b*(y_sample)-np.repeat(k,len(x_2d),axis=1)
        ans = np.matmul(A_inv,cur_b)
        x = np.concatenate([x,ans[0,:].reshape(1,-1)],axis=0)
        z = np.concatenate([z,ans[2,:].reshape(1,-1)],axis=0)

    start = 0
    # for i in range(len(x_2d)):
    #     if -20<z[0,i]<20 and -0.5<z[0,i]<0.5:
    #         start = i
    #         break
    end = len(x_2d)-1
    # for i in range(len(x_2d)-1,-1,-1):
    #     if -20<z[len(sampling_pts)-1,i]<20 and -0.5<z[len(sampling_pts)-1,i]<0.5:
    #         end = i
    #         break
    sampling_positions = np.linspace(start,end,len(sampling_pts),endpoint=False,dtype=np.uint8)
    final_x = np.zeros((0,1))
    final_z = np.zeros((0,1))
    for i in range(len(sampling_pts)):
        final_x = np.concatenate([final_x,x[i,sampling_positions[i]].reshape(1,1)],axis = 0)
        final_z = np.concatenate([final_z,z[i,sampling_positions[i]].reshape(1,1)],axis = 0)

    x = final_x
    z = final_z
    y = np.array(sampling_pts).reshape(-1,1)
    return x, y, z

colors = [
    (1.0, 0.0, 0.0),     # Red
    (0.0, 1.0, 0.0),     # Green
    (0.0, 0.0, 1.0),     # Blue
    (1.0, 1.0, 0.0),     # Yellow
    (0.0, 1.0, 1.0),     # Cyan
    (1.0, 0.0, 1.0),     # Magenta
    (0.502, 0.0, 0.0),   # Maroon
    (0.0, 0.502, 0.0),   # Olive
    (0.0, 0.0, 0.502),   # Navy
    (0.502, 0.502, 0.502),  # Gray
    (1.0, 0.647, 0.0),   # Orange
    (0.941, 0.902, 0.549), # Khaki
    (0.118, 0.565, 1.0), # DodgerBlue
    (0.941, 0.502, 0.502), # LightCoral
    (0.133, 0.545, 0.133), # ForestGreen
    (0.576, 0.439, 0.859), # MediumPurple
    (0.824, 0.412, 0.118), # Chocolate
    (0.941, 0.973, 1.0),   # LightCyan
    (1.0, 0.388, 0.278),   # Tomato
    (0.824, 0.706, 0.549)  # Wheat
]